from segment_anything import sam_model_registry
import os
import torch
from torch import nn
import cv2
import torch.nn.functional as F
import numpy as np


from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter



def coff_fuse(SAM_logits, model_logits, alpha):
    SAM_logits = torch.sigmoid(SAM_logits)
    SAM_logits = torch.cat([1-SAM_logits, SAM_logits], dim = 1)
    model_logits = model_logits.softmax(dim = 1)

    return alpha * SAM_logits + (1 - alpha) * model_logits

class SAM_pred(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.sam_model = sam_model_registry[args.sam_type](args.ckpt)
        self.sam_model.eval()
        self.point = args.point 
        self.negative_point = args.negative_point
        self.positive_point = args.positive_point
        self.args = args

    def forward_img_encoder(self, query_img):
        query_img = F.interpolate(query_img, (1024,1024), mode='bilinear', align_corners=True)

        with torch.no_grad():
            query_feats = self.sam_model.image_encoder(query_img)

        return query_feats
    
    
    def get_feat_from_np(self, query_img, query_name, protos):
        np_feat_path = '/root/paddlejob/workspace/env_run/vrp_sam/feats_np/coco/'
        if not os.path.exists(np_feat_path): os.makedirs(np_feat_path)
        files_name = os.listdir(np_feat_path)
        query_feat_list = []
        for idx, name in enumerate(query_name):
            if '/root' in name:
                name = os.path.splitext(name.split('/')[-1])[0]
                
            if name + '.npy' not in files_name:
                query_feats_np = self.forward_img_encoder(query_img[idx, :, :, :].unsqueeze(0))
                query_feat_list.append(query_feats_np)
                query_feats_np = query_feats_np.detach().cpu().numpy()
                np.save(np_feat_path + name + '.npy', query_feats_np)
            else:
                sub_query_feat = torch.from_numpy(np.load(np_feat_path + name + '.npy')).to(protos.device)
                query_feat_list.append(sub_query_feat)
                del sub_query_feat
        query_feats_np = torch.cat(query_feat_list, dim=0)
        return query_feats_np

    def get_pormpt(self, protos, points_mask=None):
        if points_mask is not None :
            point_mask = points_mask

            postivate_pos = (point_mask.squeeze(0).nonzero().unsqueeze(0) + 0.5) * 64 -0.5
            postivate_pos = postivate_pos[:,:,[1,0]]
            point_label = torch.ones(postivate_pos.shape[0], postivate_pos.shape[1]).to(postivate_pos.device)
            point_prompt = (postivate_pos, point_label)
        else:
            point_prompt = None
        protos = protos
        return  protos, point_prompt

    def forward_prompt_encoder(self, points=None, boxes=None, protos=None, masks=None):
        q_sparse_em, q_dense_em = self.sam_model.prompt_encoder(
                points=points,
                boxes=None,
                protos=protos,
                masks=None)
        return q_sparse_em, q_dense_em
    
    def forward_mask_decoder(self, query_feats, q_sparse_em, q_dense_em, ori_size, protos = None, attn_sim=None):
        # if protos is not None: 
        #     protos = torch.mean(protos, dim = 1, keepdim=True)
        
        low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                image_embeddings=query_feats,
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=q_sparse_em,
                dense_prompt_embeddings=q_dense_em,
                protos = protos,
                attn_sim = attn_sim,
                multimask_output=False)
        low_masks = F.interpolate(low_res_masks, size=ori_size, mode='bilinear', align_corners=True)
            
      
        binary_mask = torch.where(low_masks > 0, 1, 0)
        return low_masks, binary_mask
    
    
    def forward(self, query_img, prediction, origin_pred = None):
        # B,C, h, w = query_img.shape
        h, w = 400, 400
       
        coords, labels= self.point_mask_slic(origin_pred) # best 

        with torch.no_grad():
        #     #-------------save_sam_img_feat------------------------- # save_sam_img_feat
             query_feats = self.forward_img_encoder(query_img)

        #     #query_feats = self.get_feat_from_np(query_img, query_name, protos)

        q_sparse_em, q_dense_em = self.forward_prompt_encoder(
                points=(coords, labels),
                boxes=None,
                protos=None,
                masks=None)
            
        low_masks, binary_mask = self.forward_mask_decoder(query_feats, q_sparse_em, q_dense_em, (h, w), protos = None, attn_sim=None)

        pred = coff_fuse(low_masks,prediction, self.args.alpha)

        return pred, low_masks


    
    def transform(self, coords, old_h = 400, old_w = 400, new_h = 1024, new_w = 1024):
        # coords has shape [B x N x 2]
        coords = coords.float()
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords
    

    def mask_slic(self, pred, point_num, avg_sp_area=100):
        '''
        :param mask: the RoI region to do clustering, torch tensor: H x W
        :param down_stride: downsampled stride for RoI region
        :param max_num_sp: the maximum number of superpixels
        :return: segments: the coordinates of the initial seed, max_num_sp x 2
        '''
        assert point_num >= 0
        mask = pred # Binary mask
        h, w = mask.shape
        max_num_sp = point_num

        segments_x = np.zeros(max_num_sp, dtype=np.int64)
        segments_y = np.zeros(max_num_sp, dtype=np.int64)

        m_np = mask.cpu().numpy()
        m_np_down = m_np

        nz = np.nonzero(m_np_down) # 找到所有的非零坐标下标 x坐标，y坐标
        # After transform, there may be no nonzero in the label
        if len(nz[0]) != 0: # 区域内存在掩码区域

            p = [np.min(nz[0]), np.min(nz[1])] # 左上角
            pend = [np.max(nz[0]), np.max(nz[1])] # 右下角

            # cropping to bounding box around ROI
            m_np_roi = np.copy(m_np_down)[p[0]:pend[0] + 1, p[1]:pend[1] + 1] # 感兴趣的区域
            num_sp = max_num_sp

        else:
            num_sp = 0

        if (num_sp != 0) and (num_sp != 1):
            for i in range(num_sp):

                # n seeds are placed as far as possible from every other seed and the edge.

                # STEP 1:  conduct Distance Transform and choose the maximum point
                dtrans = distance_transform_edt(m_np_roi) #  
                dtrans = gaussian_filter(dtrans, sigma=0.1)

                coords1 = np.nonzero(dtrans == np.max(dtrans))
                segments_x[i] = coords1[0][0]
                segments_y[i] = coords1[1][0]

                # STEP 2:  set the point to False and repeat Step 1
                m_np_roi[segments_x[i], segments_y[i]] = False
                segments_x[i] += p[0]
                segments_y[i] += p[1]

        segments = np.concatenate([segments_y[..., np.newaxis], segments_x[..., np.newaxis]], axis=1)  # max_num_sp x 2
        segments = torch.from_numpy(segments)
        segments = segments.to(pred.device)

        return segments
    

    def point_mask_slic(self, pred):
        b = pred.shape[0]
        coords = []
        labels = []
        
        for i in range(b):
            pred_i = pred[i, :, :, :]
            # positive
            seg_p= self.mask_slic(pred_i.argmax(dim = 0), self.positive_point)
            # negatice
            
            seg_n = self.mask_slic(pred_i.flip(dims=[0]).argmax(dim = 0), self.negative_point)
            
            M = seg_p.shape[0] 
            N = seg_n.shape[0]
            label_p = torch.ones(M, dtype=torch.int)
            label_n = torch.zeros(N, dtype=torch.int)

            coords.append(torch.cat([seg_p, seg_n], dim=0))
            labels.append(torch.cat([label_p, label_n], dim = 0))
         

        coords = torch.stack(coords, dim = 0) # B X N X 2
        labels = torch.stack(labels, dim = 0)

        labels = labels.to(pred.device)
        coords = self.transform(coords, 50, 50)
       
        return coords, labels
    




    


    