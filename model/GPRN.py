import model.resnet as resnet
import sys
import torch
from torch import nn
import torch.nn.functional as F
import pdb
from SAM2pred import *

import numpy as np



def batch_cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # calculate cosine similarity between a and b
    # a: [batch,num_a,channel]
    # b: [batch,num_b,channel]
    # return: [batch,num_a,num_b]
    assert a.shape[0] == b.shape[0], 'batch size of a and b must be equal'
    assert a.shape[2] == b.shape[2], 'channel of a and b must be equal'
    cos_esp = 1e-8
    a_norm = a.norm(dim=2, keepdim=True)
    b_norm = b.norm(dim=2, keepdim=True)
    cos_sim = torch.bmm(a, b.permute(0, 2, 1))
    cos_sim = cos_sim / (torch.bmm(a_norm, b_norm.permute(0, 2, 1)) + cos_esp)
    return cos_sim


class GraphAttention(nn.Module):
    def __init__(self, h_dim=None):
        super(GraphAttention, self).__init__()
        self.with_projection = h_dim is not None
        if self.with_projection:
            self.linear = nn.Linear(h_dim, h_dim)
            # self.linear_k = nn.Linear(h_dim, h_dim)
            # self.linear_v = nn.Linear(h_dim, h_dim)

    def forward(self, q_node, k_node, v_node):
        assert q_node.shape[0] == k_node.shape[0] and q_node.shape[
            0] == v_node.shape[0]
        assert k_node.shape[1] == v_node.shape[1]
        assert q_node.shape[2] == k_node.shape[2]

        if self.with_projection:
            q_node = self.linear(q_node)
            k_node = self.linear(k_node)
            v_node = self.linear(v_node)

        cos_sim = batch_cos_sim(q_node, k_node)
        sum_sim = cos_sim.sum(dim=2, keepdim=True)
        edge_weight = cos_sim / (sum_sim + 1e-8)
        edge_feature = torch.bmm(edge_weight, v_node)
        return edge_feature
    

class GPRN_Net(nn.Module):
    def __init__(self, backbone, refine=False, shot=1, args = None):
        super(GPRN_Net, self).__init__()
        backbone = resnet.__dict__[backbone](pretrained=True)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3
        self.refine = refine
        self.shot = shot
        self.args = args
        self.iter_refine = False
        self.adapter = nn.Linear(1024, 256, bias = False)
        self.graph_attention = GraphAttention(h_dim=1024)


    def enhance_feature(self, feature_q, qry_sam_masks):
        def l2_normalize(tensor):
            norm = tensor.norm(p=2, dim=(2, 3), keepdim=True)
            return tensor / norm
        
        if self.args.dataset in ('isic', 'lung'):
            qry_sam_masks = qry_sam_masks[:, 1:, :, :]
        b, m, w, h = qry_sam_masks.shape
        index_mask = torch.zeros_like(qry_sam_masks[:, 0]).long() + m # b x h x w
        for i in range(m):
            index_mask[qry_sam_masks[:, i]==1] = i
        masks = torch.nn.functional.one_hot(index_mask)[:, :, :, :m].permute((0, 3, 1, 2))

        #masks = qry_sam_masks

        if self.training:
            target_masks = F.interpolate(masks.float(), feature_q.shape[-2:], mode='nearest')
        else:
            target_masks = masks.float()
        map_features = self.masked_average_pooling2(feature_q, target_masks) # b x m x c, 用mask生成的prototypes 

        graph_prompt = self.graph_attention(map_features, map_features, map_features)

        w = self.args.weight

        map_features = map_features + w * graph_prompt
        b, m, w, h = target_masks.shape
        _, _, c = map_features.shape
        _map_features = map_features.permute(0, 2, 1).contiguous() # b x c x m
        feature_sum = _map_features @ target_masks.view(b, m, -1) # 
        feature_sum = feature_sum.view(b, c, w, h)

        sum_mask = target_masks.sum(dim=1, keepdim=True)
        #enabled_feat = torch.div(feature_sum, sum_mask + 1e-5)
        enabled_feat = feature_sum

        return enabled_feat


    def forward(self, img_s_list, mask_s_list, img_q, mask_q, qry_sam_masks, supp_sam_masks):
        b, c, h, w = img_q.shape
        # keep_num = self.args.keep_num
        # qry_sam_masks = qry_sam_masks[:, :keep_num, :, :]
        # for i in range(self.shot):
        #     supp_sam_masks[i] = supp_sam_masks[i][:, :keep_num, :, :]

        # feature maps of support images
        feature_s_list = []
        origin_feat_s = []
        #supp_dis = []

        for k in range(len(img_s_list)):
            with torch.no_grad():
                s_0 = self.layer0(img_s_list[k])
                s_0 = self.layer1(s_0)
            s_0 = self.layer2(s_0)
            s_0 = self.layer3(s_0)
            origin_feat_s.append(s_0)
            enhance_s= self.enhance_feature(s_0, supp_sam_masks[k])
            s_0 = s_0 + enhance_s
            feature_s_list.append(s_0)
            del s_0

        feature_s_ls = torch.cat(feature_s_list, dim=0)
        origin_feat_s = torch.cat(origin_feat_s, dim=0)

        # feature map of query image
        with torch.no_grad():
            # self.query_feat = query_feat
            q_0 = self.layer0(img_q)
            q_0 = self.layer1(q_0)
        q_0 = self.layer2(q_0)
        feature_q = self.layer3(q_0)
        enhance_q = self.enhance_feature(feature_q, qry_sam_masks)
        z = feature_q.clone()
        feature_q = feature_q + enhance_q 


        # foreground(target class) and background prototypes pooled from K support features
        feature_fg_list = []
        feature_bg_list = []
        supp_out_ls = []

        for k in range(len(img_s_list)):
            feature_fg = self.masked_average_pooling(feature_s_list[k],
                                                               (mask_s_list[k] == 1).float())[None, :]
            feature_bg = self.masked_average_pooling(feature_s_list[k],
                                                               (mask_s_list[k] == 0).float())[None, :]
            
            feature_fg_list.append(feature_fg)
            feature_bg_list.append(feature_bg)

            if self.training:
                supp_similarity_fg = F.cosine_similarity(feature_s_list[k], feature_fg.squeeze(0)[..., None, None], dim=1)
                supp_similarity_bg = F.cosine_similarity(feature_s_list[k], feature_bg.squeeze(0)[..., None, None], dim=1)
                supp_out = torch.cat((supp_similarity_bg[:, None, ...], supp_similarity_fg[:, None, ...]), dim=1) * 10.0

                supp_out = F.interpolate(supp_out, size=(h, w), mode="bilinear", align_corners=True)
                supp_out_ls.append(supp_out)

        # average K foreground prototypes and K background prototypes
        FP = torch.mean(torch.cat(feature_fg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        BP = torch.mean(torch.cat(feature_bg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)



        if self.training:

            ### iter = 1 (BFP)
            if self.refine:
                out_refine, out_1, supp_out_1, new_FP, new_BP, FP_1, FP_2, out_0 = self.iter_BFP(FP, BP, feature_s_ls, feature_q, self.refine)
            else:
                out_1, supp_out_1, new_FP, new_BP = self.iter_BFP(FP, BP, feature_s_ls, feature_q, self.refine)
            out_1 = F.interpolate(out_1, size=(h, w), mode="bilinear", align_corners=True)
            supp_out_1 = F.interpolate(supp_out_1, size=(h, w), mode="bilinear", align_corners=True)
          
        else:
            if self.refine:
                out_refine, out_1,FP_1, FP_2, out_0,BP_1= self.iter_BFP(FP, BP, feature_s_ls, feature_q, self.refine)
            else:
                out_1 = self.iter_BFP(FP, BP, feature_s_ls, feature_q, self.refine)
            out_1 = F.interpolate(out_1, size=(h, w), mode="bilinear", align_corners=True)

        if self.refine:
            out_refine_origin = out_refine.clone()
            out_refine = F.interpolate(out_refine, size=(h, w), mode="bilinear", align_corners=True)
            out_0 = F.interpolate(out_0, size=(h, w), mode="bilinear", align_corners=True)
            out_ls = [out_refine, out_1]
        else:
            out_ls = [out_1]

        if self.training:
            fg_q = self.masked_average_pooling(feature_q, (mask_q == 1).float())[None, :].squeeze(0)
            bg_q = self.masked_average_pooling(feature_q, (mask_q == 0).float())[None, :].squeeze(0)

            self_similarity_fg = F.cosine_similarity(feature_q, fg_q[..., None, None], dim=1)
            self_similarity_bg = F.cosine_similarity(feature_q, bg_q[..., None, None], dim=1)
            self_out = torch.cat((self_similarity_bg[:, None, ...], self_similarity_fg[:, None, ...]), dim=1) * 10.0

            self_out = F.interpolate(self_out, size=(h, w), mode="bilinear", align_corners=True)
            supp_out = torch.cat(supp_out_ls, 0)

            out_ls.append(self_out)
            out_ls.append(supp_out)

        out_ls.append(out_refine_origin)
        out_ls.append(out_0)
     
        return out_ls
    



    def SSP_func(self, feature_q, out, flag = True):
        device = feature_q.device
        bs,c= feature_q.shape[:2]
        pred_1 = out.softmax(1)
        pred_2 = pred_1.view(bs, 2, -1)
        pred_fg = pred_2[:, 1]
        pred_bg = pred_2[:, 0]
        fg_ls = []
        bg_ls = []
        fg_local_ls = []
        bg_local_ls = []
        for epi in range(bs):
            f_h, f_w = feature_q[epi].shape[-2:]
            fg_mask = torch.zeros(f_h, f_w).to(torch.int64).to(device)
            bg_mask = torch.zeros(f_h, f_w).to(torch.int64).to(device=device)

            fg_thres = 0.7
            bg_thres = 0.6
            cur_feat = feature_q[epi].view(c, -1)
            
            if (pred_fg[epi] > fg_thres).sum() > 0:
                fg_feat = cur_feat[:, (pred_fg[epi]>fg_thres)] #.mean(-1)
                fg_mask = pred_1[epi, 1, :, :] > fg_thres
            else:
                topk_fg = torch.topk(pred_fg[epi], 12).indices
                fg_feat = cur_feat[:, topk_fg] #.mean(-1)
                topk_coords = torch.stack((topk_fg// f_w, topk_fg % f_w), dim=1)
                fg_mask[topk_coords[:,0], topk_coords[:, 1]] = 1
            if (pred_bg[epi] > bg_thres).sum() > 0:
                bg_feat = cur_feat[:, (pred_bg[epi]>bg_thres)] #.mean(-1)
                bg_mask = pred_1[epi, 0, :, :] > bg_thres
            else:
                topk_bg = torch.topk(pred_bg[epi], 12).indices
                bg_feat = cur_feat[:, topk_bg] #.mean(-1)
                topk_coords_b = torch.stack((topk_bg// f_w, topk_bg % f_w), dim=1)
                bg_mask[topk_coords_b[:,0], topk_coords_b[:, 1]] = 1
            
            # global proto
            fg_proto = fg_feat.mean(-1)
            bg_proto = bg_feat.mean(-1)
            fg_ls.append(fg_proto.unsqueeze(0))
            bg_ls.append(bg_proto.unsqueeze(0))

            # local proto
            fg_feat_norm = fg_feat / torch.norm(fg_feat, 2, 0, True) # 1024, N1
            bg_feat_norm = bg_feat / torch.norm(bg_feat, 2, 0, True) # 1024, N2
            cur_feat_norm = cur_feat / torch.norm(cur_feat, 2, 0, True) # 1024, N3

            cur_feat_norm_t = cur_feat_norm.t() # N3, 1024
            fg_sim = torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0 # N3, N1
            bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0 # N3, N2

            fg_sim = fg_sim.softmax(-1)
            bg_sim = bg_sim.softmax(-1)

            fg_proto_local = torch.matmul(fg_sim, fg_feat.t()) # N3, 1024
            bg_proto_local = torch.matmul(bg_sim, bg_feat.t()) # N3, 1024

            fg_proto_local = fg_proto_local.t().view(c, f_h, f_w).unsqueeze(0) # 1024, N3
            bg_proto_local = bg_proto_local.t().view(c, f_h, f_w).unsqueeze(0) # 1024, N3

            fg_local_ls.append(fg_proto_local)
            bg_local_ls.append(bg_proto_local)

        # global proto
        new_fg = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)

        # local proto
        new_fg_local = torch.cat(fg_local_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg_local = torch.cat(bg_local_ls, 0)

        return new_fg, new_bg, new_fg_local, new_bg_local
    

    def similarity_func(self, feature_q, fg_proto, bg_proto):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)

        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        return out

    def masked_average_pooling(self, feature, mask):
        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) \
                         / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature
    
    def masked_average_pooling2(self, feature, mask):
        b, c, w, h = feature.shape
        _, m, _, _ = mask.shape

        _mask = mask.view(b, m, -1)
        _feature = feature.view(b, c, -1).permute(0, 2, 1).contiguous() # b, h*w, c
        feature_sum = _mask @ _feature # b x m x c
        masked_sum = torch.sum(_mask, dim=2, keepdim=True) # b x m x 1

        masked_average_pooling = torch.div(feature_sum, masked_sum + 1e-5)
        return masked_average_pooling

    
    def iter_BFP(self, FP, BP, feature_s_ls, feature_q, refine=True):
        ###### input FP and BP are support prototype
        ###### SSP on query side
        ### find the most similar part in query feature
        out_0 = self.similarity_func(feature_q, FP, BP)
        ### SSP in query feature
        SSFP_1, SSBP_1, ASFP_1, ASBP_1 = self.SSP_func(feature_q, out_0, True)
        ### update prototype for query prediction
        FP_1 = FP * 0.5 + SSFP_1 * 0.5
        FP_use = FP_1.clone()
        BP_1 = SSBP_1 * 0.3 + ASBP_1 * 0.7
        BP_use = SSBP_1.clone()
        ### use updated prototype to search target in query feature
        out_1 = self.similarity_func(feature_q, FP_1, BP_1)
        ###### Refine (only for the 1st iter)
        if refine:
            ### use updated prototype to find the most similar part in query feature again
            SSFP_2, SSBP_2, ASFP_2, ASBP_2 = self.SSP_func(feature_q, out_1, True)
            ### update prototype again for query regine
            FP_2 = FP * 0.5 + SSFP_2 * 0.5
            BP_2 = SSBP_2 * 0.3 + ASBP_2 * 0.7
            FP_2 = FP * 0.5 + FP_1 * 0.2 + FP_2 * 0.3
            BP_2 = BP * 0.5 + BP_1 * 0.2 + BP_2 * 0.3
            ### use updated prototype to search target in query feature again
            out_refine = self.similarity_func(feature_q, FP_2, BP_2)
            out_refine = out_refine * 0.7 + out_1 * 0.3

        ###### SSP on support side
        if self.training:
            ### duplicate query prototype for support SSP if shot > 1
            if self.shot > 1:
                FP_nshot = FP.repeat_interleave(self.shot, dim=0)
                FP_1 = FP_1.repeat_interleave(self.shot, dim=0)
                BP_1 = BP_1.repeat_interleave(self.shot, dim=0)
            ### find the most similar part in support feature list
            supp_out_0 = self.similarity_func(feature_s_ls, FP_1, BP_1)
            ### SSP in support feature list
            SSFP_supp, SSBP_supp, ASFP_supp, ASBP_supp = self.SSP_func(feature_s_ls, supp_out_0, False)
            ### update prototype for support prediction
            if self.shot > 1:
                FP_supp = FP_nshot * 0.5 + SSFP_supp * 0.5
            else:
                FP_supp = FP * 0.5 + SSFP_supp * 0.5

            BP_supp = SSBP_supp * 0.3 + ASBP_supp * 0.7
            ### use updated prototype to search target in support feature list
            supp_out_1 = self.similarity_func(feature_s_ls, FP_supp, BP_supp)

            ### process prototype if shot > 1
            if self.shot > 1:
                for i in range(FP_supp.shape[0]//self.shot):
                    for j in range(self.shot):
                        # print("each FP_supp", FP_supp[i*self.shot+j])
                        if j == 0:
                            FP_supp_avg = FP_supp[i*self.shot+j]
                            pass
                            BP_supp_avg = BP_supp[i*self.shot+j]
                        else:
                            FP_supp_avg = FP_supp_avg + FP_supp[i*self.shot+j]
                            BP_supp_avg = BP_supp_avg + BP_supp[i*self.shot+j]

                    FP_supp_avg = FP_supp_avg/self.shot
                    BP_supp_avg = BP_supp_avg/self.shot
                    FP_supp_avg = FP_supp_avg.reshape(1,FP_supp.shape[1],FP_supp.shape[2],FP_supp.shape[3])
                    BP_supp_avg = BP_supp_avg.reshape(1,BP_supp.shape[1],BP_supp.shape[2],BP_supp.shape[3])
                    if i == 0:
                        new_FP_supp = FP_supp_avg
                        new_BP_supp = BP_supp_avg
                    else:
                        new_FP_supp = torch.cat((new_FP_supp,FP_supp_avg), dim=0)
                        new_BP_supp = torch.cat((new_BP_supp,BP_supp_avg), dim=0)

                FP_supp = new_FP_supp
                BP_supp = new_BP_supp          

        if refine:
            if self.training:
                return out_refine, out_1, supp_out_1, FP_supp, BP_supp, FP_use, FP_2, out_0
            else:
                return out_refine, out_1, FP_use, FP_2, out_0, BP_use
        else:
            if self.training:
                return out_1, supp_out_1, FP_supp, BP_supp
            else:
                return out_1
            

