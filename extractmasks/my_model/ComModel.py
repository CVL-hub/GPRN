import math
import torch
import numpy as np
from torch import Tensor, nn
from typing import Tuple, Type
from alisuretool.Tools import Tools
from torch.nn import functional as F


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int,
                 act: Type[nn.Module] = nn.GELU, embedding_dim_out: int = None) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim if embedding_dim_out is None else embedding_dim_out)
        self.act = act()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

    pass


class Attention(nn.Module):

    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        pass

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

    pass


class MyCrossAttention(nn.Module):

    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim: int, attention_downsample_rate: int = 2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim

        self.cross_attn = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.cross_attn_norm = nn.LayerNorm(embedding_dim)
        self.cross_attn_mlp1 = MLPBlock(embedding_dim, mlp_dim, act=nn.ReLU)
        self.cross_attn_mlp2 = MLPBlock(embedding_dim, mlp_dim, act=nn.ReLU)
        self.cross_attn_mlp1_norm = nn.LayerNorm(embedding_dim)
        self.cross_attn_mlp2_norm = nn.LayerNorm(embedding_dim)
        pass

    def forward(self, ques: Tensor, keys: Tensor) -> Tuple[Tensor, Tensor]:
        ques_out = self.cross_attn_norm(ques + self.cross_attn(q=ques, k=keys, v=keys))
        keys_out = self.cross_attn_norm(keys + self.cross_attn(q=keys, k=ques, v=ques))
        ques_out = self.cross_attn_mlp1_norm(ques_out + self.cross_attn_mlp1(ques_out))
        keys_out = self.cross_attn_mlp2_norm(keys_out + self.cross_attn_mlp2(keys_out))
        return ques_out, keys_out

    pass


class OurComModel(nn.Module):

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.project_cross = MyCrossAttention(embedding_dim=embedding_dim, mlp_dim=2048, num_heads=8)
        pass

    def load_checkpoint(self, checkpoint):
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location=None)
                pass
            self.load_state_dict(state_dict, strict=True)
            Tools.print("load weights from {}".format(checkpoint))
        else:
            Tools.print("load weights error due to no checkpoint")
        pass

    def forward(self, query_part_masked_average_pooling_feature, support_masked_average_pooling_features,
                support_part_masked_average_pooling_features, support_part_mask_ratio, query_part_mask_segmentation):
        b, query_p, c = query_part_masked_average_pooling_feature.shape
        _, support_k, support_p, _ = support_masked_average_pooling_features.shape

        # project
        query_part_map_feature, support_map_feature = self.project_cross(
            query_part_masked_average_pooling_feature, support_masked_average_pooling_features.view(b, -1, c))
        query_part_map_feature_2, support_part_map_feature_2 = self.project_cross(
            query_part_masked_average_pooling_feature, support_part_masked_average_pooling_features.view(b, -1, c))

        # sim
        # sim = torch.cosine_similarity(support_map_feature, query_part_map_feature, dim=2)
        sim_dot = support_map_feature @ query_part_map_feature.permute(0, 2, 1).contiguous()
        sim_norm = torch.norm(support_map_feature, dim=2) * torch.norm(query_part_map_feature, dim=2)
        sim = sim_dot / (sim_norm.unsqueeze(1) + 1e-8)

        sim2_dot = support_part_map_feature_2 @ query_part_map_feature_2.permute(0, 2, 1).contiguous()
        sim2_norm = torch.norm(support_part_map_feature_2, dim=2).unsqueeze(2) \
                   @ torch.norm(query_part_map_feature_2, dim=2).unsqueeze(1)
        sim2 = sim2_dot / (sim2_norm + 1e-8)

        # score
        score = sim.mean(dim=1)
        score2 = sim2.view(b, support_k, -1, query_p)

        # predict
        query_part_mask_segmentation_score = query_part_mask_segmentation * score.view(b, -1, 1, 1)
        predict1 = query_part_mask_segmentation_score.sum(dim=1)

        score2_ratio = support_part_mask_ratio.unsqueeze(-1) * score2
        score2_ratio_sum = score2_ratio.sum(dim=2) / (support_part_mask_ratio.sum(dim=2).unsqueeze(-1) + 1e-8)
        score2_ratio_sum = score2_ratio_sum.mean(dim=1)
        query_part_mask_segmentation_score2 = query_part_mask_segmentation * score2_ratio_sum.view(b, -1, 1, 1)
        predict2 = query_part_mask_segmentation_score2.sum(dim=1)

        predict = (predict1 + predict2) / 2
        return score, score2, predict, predict1, predict2

    def get_postprocess_masks(self, masks, input_size, original_size):
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    pass


class OurComModelMax(nn.Module):

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.project_cross = MyCrossAttention(embedding_dim=embedding_dim, mlp_dim=2048, num_heads=8)
        pass

    def load_checkpoint(self, checkpoint):
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location=None)
                pass
            self.load_state_dict(state_dict, strict=True)
            Tools.print("load weights from {}".format(checkpoint))
        else:
            Tools.print("load weights error due to no checkpoint")
        pass

    def forward(self, query_part_masked_average_pooling_feature, support_masked_average_pooling_features,
                support_part_masked_average_pooling_features, support_part_mask_ratio, query_part_mask_segmentation):
        b, query_p, c = query_part_masked_average_pooling_feature.shape
        _, support_k, support_p, _ = support_masked_average_pooling_features.shape

        # project
        query_part_map_feature, support_map_feature = self.project_cross(
            query_part_masked_average_pooling_feature, support_masked_average_pooling_features.view(b, -1, c))
        query_part_map_feature_2, support_part_map_feature_2 = self.project_cross(
            query_part_masked_average_pooling_feature, support_part_masked_average_pooling_features.view(b, -1, c))

        # sim
        # sim = torch.cosine_similarity(support_map_feature, query_part_map_feature, dim=2)
        sim_dot = support_map_feature @ query_part_map_feature.permute(0, 2, 1).contiguous()
        sim_norm = torch.norm(support_map_feature, dim=2) * torch.norm(query_part_map_feature, dim=2)
        sim = sim_dot / (sim_norm.unsqueeze(1) + 1e-8)

        sim2_dot = support_part_map_feature_2 @ query_part_map_feature_2.permute(0, 2, 1).contiguous()
        sim2_norm = torch.norm(support_part_map_feature_2, dim=2).unsqueeze(2) \
                   @ torch.norm(query_part_map_feature_2, dim=2).unsqueeze(1)
        sim2 = sim2_dot / (sim2_norm + 1e-8)

        # score
        score = sim.mean(dim=1)
        score2 = sim2.view(b, support_k, -1, query_p)

        # predict
        query_part_mask_segmentation_score = query_part_mask_segmentation * score.view(b, -1, 1, 1)
        predict1 = query_part_mask_segmentation_score.max(dim=1)[0]

        score2_ratio = support_part_mask_ratio.unsqueeze(-1) * score2
        score2_ratio_sum = score2_ratio.sum(dim=2) / (support_part_mask_ratio.sum(dim=2).unsqueeze(-1) + 1e-8)
        score2_ratio_sum = score2_ratio_sum.mean(dim=1)
        query_part_mask_segmentation_score2 = query_part_mask_segmentation * score2_ratio_sum.view(b, -1, 1, 1)
        predict2 = query_part_mask_segmentation_score2.max(dim=1)[0]

        predict = (predict1 + predict2) / 2
        return score, score2, predict, predict1, predict2

    def get_postprocess_masks(self, masks, input_size, original_size):
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    pass


# OK
class OurComModelSimple(nn.Module):

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.project_cross = MyCrossAttention(embedding_dim=embedding_dim, mlp_dim=2048, num_heads=8)
        pass

    def load_checkpoint(self, checkpoint):
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location=None)
                pass
            self.load_state_dict(state_dict, strict=True)
            Tools.print("load weights from {}".format(checkpoint))
        else:
            Tools.print("load weights error due to no checkpoint")
        pass

    def forward(self, query_part_masked_average_pooling_feature,
                support_masked_average_pooling_feature, query_part_mask_segmentation):
        b, query_p, c = query_part_masked_average_pooling_feature.shape

        # project
        query_part_map_feature, support_map_feature = self.project_cross(
            query_part_masked_average_pooling_feature, support_masked_average_pooling_feature)

        # sim
        # sim = torch.cosine_similarity(support_map_feature, query_part_map_feature, dim=2)
        sim_dot = support_map_feature @ query_part_map_feature.permute(0, 2, 1).contiguous()
        sim_norm = torch.norm(support_map_feature, dim=2) * torch.norm(query_part_map_feature, dim=2)
        sim = sim_dot / (sim_norm.unsqueeze(1) + 1e-8)

        # score
        # score = (sim.mean(dim=1) + 1) / 2
        score = sim.mean(dim=1)

        # predict
        query_part_mask_segmentation_score = query_part_mask_segmentation * score.view(b, -1, 1, 1)
        predict = query_part_mask_segmentation_score.max(dim=1)[0]

        return score, predict

    def get_postprocess_masks(self, masks, input_size, original_size):
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    pass


# OK
class OurComModelSimpleEmbedding(nn.Module):

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.project_cross = MyCrossAttention(embedding_dim=embedding_dim, mlp_dim=2048, num_heads=8)
        self.null_token = nn.Embedding(1, embedding_dim)
        pass

    def load_checkpoint(self, checkpoint):
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location=None)
                pass
            self.load_state_dict(state_dict, strict=True)
            Tools.print("load weights from {}".format(checkpoint))
        else:
            Tools.print("load weights error due to no checkpoint")
        pass

    def forward(self, query_part_masked_average_pooling_feature,
                support_masked_average_pooling_feature, query_part_mask_segmentation):
        b, query_p, c = query_part_masked_average_pooling_feature.shape

        # query_part_masked_average_pooling_feature[
        #     query_part_masked_average_pooling_feature.sum(dim=-1) == 0.0] = self.null_token.weight
        # support_masked_average_pooling_feature[
        #     support_masked_average_pooling_feature.sum(dim=-1) == 0.0] = self.null_token.weight
        query_part_masked_average_pooling_feature += self.null_token.weight
        support_masked_average_pooling_feature += self.null_token.weight

        # project
        query_part_map_feature, support_map_feature = self.project_cross(
            query_part_masked_average_pooling_feature, support_masked_average_pooling_feature)

        # sim
        # sim = torch.cosine_similarity(support_map_feature, query_part_map_feature, dim=2)
        sim_dot = support_map_feature @ query_part_map_feature.permute(0, 2, 1).contiguous()
        sim_norm = torch.norm(support_map_feature, dim=2) * torch.norm(query_part_map_feature, dim=2)
        sim = sim_dot / (sim_norm.unsqueeze(1) + 1e-8)

        # score
        # score = (sim.mean(dim=1) + 1) / 2
        score = sim.mean(dim=1)

        # predict
        query_part_mask_segmentation_score = query_part_mask_segmentation * score.view(b, -1, 1, 1)
        predict = query_part_mask_segmentation_score.max(dim=1)[0]

        return score, predict

    def get_postprocess_masks(self, masks, input_size, original_size):
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    pass


class OurComModelSimplePart(nn.Module):

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.project_cross = MyCrossAttention(embedding_dim=embedding_dim, mlp_dim=2048, num_heads=8)
        pass

    def load_checkpoint(self, checkpoint):
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location=None)
                pass
            self.load_state_dict(state_dict, strict=True)
            Tools.print("load weights from {}".format(checkpoint))
        else:
            Tools.print("load weights error due to no checkpoint")
        pass

    def forward(self, query_part_masked_average_pooling_feature, support_part_map_feature,
                support_part_mask_ratio, query_part_mask_segmentation):
        b, query_p, c = query_part_masked_average_pooling_feature.shape

        # project
        query_part_map_feature, support_part_map_feature = self.project_cross(
            query_part_masked_average_pooling_feature, support_part_map_feature)

        # sim
        # sim = torch.cosine_similarity(support_map_feature, query_part_map_feature, dim=2)
        sim_dot = support_part_map_feature @ query_part_map_feature.permute(0, 2, 1).contiguous()
        sim_norm = torch.norm(support_part_map_feature, dim=2).unsqueeze(2) \
                   @ torch.norm(query_part_map_feature, dim=2).unsqueeze(1)
        sim = sim_dot / (sim_norm + 1e-8)

        # score
        score = sim

        # predict
        score_ratio = support_part_mask_ratio.unsqueeze(-1) * score
        score_ratio_sum = score_ratio.sum(dim=1) / (support_part_mask_ratio.sum(dim=1).unsqueeze(-1) + 1e-8)
        query_part_mask_segmentation_score = query_part_mask_segmentation * score_ratio_sum.view(b, -1, 1, 1)
        predict = query_part_mask_segmentation_score.max(dim=1)[0]

        # score_ratio = support_part_mask_ratio.unsqueeze(-1) * score
        # score_ratio_sum = score_ratio.max(dim=1)[0]
        # query_part_mask_segmentation_score = query_part_mask_segmentation * score_ratio_sum.view(b, -1, 1, 1)
        # predict = query_part_mask_segmentation_score.max(dim=1)[0]

        return score, predict

    def get_postprocess_masks(self, masks, input_size, original_size):
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    pass


class OurComModelSimplePart2(nn.Module):

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.project_cross = MyCrossAttention(embedding_dim=embedding_dim, mlp_dim=2048, num_heads=8)
        pass

    def load_checkpoint(self, checkpoint):
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location=None)
                pass
            self.load_state_dict(state_dict, strict=True)
            Tools.print("load weights from {}".format(checkpoint))
        else:
            Tools.print("load weights error due to no checkpoint")
        pass

    def forward(self, query_part_masked_average_pooling_feature, support_part_map_feature,
                support_part_mask_ratio, query_part_mask_segmentation):
        b, query_p, c = query_part_masked_average_pooling_feature.shape

        # project
        query_part_map_feature, support_map_feature = self.project_cross(
            query_part_masked_average_pooling_feature, support_part_map_feature.unsqueeze(1))

        # sim
        # sim = torch.cosine_similarity(support_map_feature, query_part_map_feature, dim=2)
        sim_dot = support_map_feature @ query_part_map_feature.permute(0, 2, 1).contiguous()
        sim_norm = torch.norm(support_map_feature, dim=2) * torch.norm(query_part_map_feature, dim=2)
        sim = sim_dot / (sim_norm.unsqueeze(1) + 1e-8)

        # score
        score = sim.mean(dim=1)

        # predict
        query_part_mask_segmentation_score = query_part_mask_segmentation * score.view(b, -1, 1, 1)
        predict = query_part_mask_segmentation_score.max(dim=1)[0]
        return score, predict

    def get_postprocess_masks(self, masks, input_size, original_size):
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    pass


# OK
class OurComModel4Merge(nn.Module):

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.project_cross = MyCrossAttention(embedding_dim=embedding_dim, mlp_dim=2048,
                                              num_heads=8, attention_downsample_rate=2)
        self.null_token = nn.Embedding(1, embedding_dim)
        pass

    def load_checkpoint(self, checkpoint):
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location=None)
                pass
            self.load_state_dict(state_dict, strict=True)
            Tools.print("load weights from {}".format(checkpoint))
        else:
            Tools.print("load weights error due to no checkpoint")
        pass

    def forward(self, query_part_map_feature, support_feature, support_ratio, query_part_mask_segmentation):
        b, query_p, c = query_part_map_feature.shape

        # 2. Point
        # query_part_map_feature[query_part_map_feature.sum(dim=-1) == 0.0] = null_token
        # support_feature[support_feature.sum(dim=-1) == 0.0] = null_token
        # query_part_map_feature += self.null_token.weight
        # support_feature += self.null_token.weight

        # project
        query_feature, support_feature = self.project_cross(query_part_map_feature, support_feature)

        # sim
        sim_dot = support_feature @ query_feature.permute(0, 2, 1).contiguous()
        sim_norm = torch.norm(support_feature, dim=2).unsqueeze(2) \
                   @ torch.norm(query_feature, dim=2).unsqueeze(1)
        score = sim_dot / (sim_norm + 1e-8)

        # 3. Point
        # predict
        score_ratio = support_ratio.unsqueeze(-1) * score
        # 方案1，求平均
        score_ratio_sum = score_ratio.sum(dim=1) / (support_ratio.sum(dim=1).unsqueeze(-1) + 1e-8)
        # 方案2，取最大
        # score_ratio_sum = score_ratio.max(dim=1)[0]
        query_part_mask_segmentation_score = query_part_mask_segmentation * score_ratio_sum.view(b, -1, 1, 1)
        predict = query_part_mask_segmentation_score.max(dim=1)[0]

        return score, predict

    def get_postprocess_masks(self, masks, input_size, original_size):
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    pass


# OK
class OurComModel4MergeSimple(nn.Module):

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.project_cross = MyCrossAttention(embedding_dim=embedding_dim, mlp_dim=2048,
                                              num_heads=8, attention_downsample_rate=2)
        pass

    def load_checkpoint(self, checkpoint):
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location=None)
                pass
            self.load_state_dict(state_dict, strict=True)
            Tools.print("load weights from {}".format(checkpoint))
        else:
            Tools.print("load weights error due to no checkpoint")
        pass

    def forward(self, query_part_map_feature, support_feature, support_ratio, query_part_mask_segmentation):
        b, query_p, c = query_part_map_feature.shape

        # project
        query_feature, support_feature = self.project_cross(query_part_map_feature, support_feature)

        # sim
        sim_dot = support_feature @ query_feature.permute(0, 2, 1).contiguous()
        sim_norm = torch.norm(support_feature, dim=2).unsqueeze(2) \
                   @ torch.norm(query_feature, dim=2).unsqueeze(1)
        sim = sim_dot / (sim_norm + 1e-8)

        # 3. Point
        # predict
        score_ratio = support_ratio.unsqueeze(-1) * sim
        # 方案1，求平均
        score = score_ratio.sum(dim=1) / (support_ratio.sum(dim=1).unsqueeze(-1) + 1e-8)
        # 方案2，取最大
        # score = score_ratio.max(dim=1)[0]
        query_part_mask_segmentation_score = query_part_mask_segmentation * score.view(b, -1, 1, 1)
        predict = query_part_mask_segmentation_score.max(dim=1)[0]

        return score, predict

    def get_postprocess_masks(self, masks, input_size, original_size):
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    pass


#
class OurComModel4MergeSimpleNorm(nn.Module):

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.project_cross = MyCrossAttention(embedding_dim=embedding_dim, mlp_dim=2048,
                                              num_heads=8, attention_downsample_rate=2)
        self.norm = nn.LayerNorm(embedding_dim) if embedding_dim == 1280 else None
        pass

    def load_checkpoint(self, checkpoint):
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location=None)
                pass
            self.load_state_dict(state_dict, strict=True)
            Tools.print("load weights from {}".format(checkpoint))
        else:
            Tools.print("load weights error due to no checkpoint")
        pass

    def forward(self, query_part_map_feature, support_feature, support_ratio, query_part_mask_segmentation):
        b, query_p, c = query_part_map_feature.shape

        # project
        if self.norm:
            query_part_map_feature = self.norm(query_part_map_feature)
            support_feature = self.norm(support_feature)
            pass
        query_feature, support_feature = self.project_cross(query_part_map_feature, support_feature)

        # sim
        sim_dot = support_feature @ query_feature.permute(0, 2, 1).contiguous()
        sim_norm = torch.norm(support_feature, dim=2).unsqueeze(2) \
                   @ torch.norm(query_feature, dim=2).unsqueeze(1)
        sim = sim_dot / (sim_norm + 1e-8)

        # 3. Point
        # predict
        score_ratio = support_ratio.unsqueeze(-1) * sim
        # 方案1，求平均
        score = score_ratio.sum(dim=1) / (support_ratio.sum(dim=1).unsqueeze(-1) + 1e-8)
        # 方案2，取最大
        # score = score_ratio.max(dim=1)[0]
        query_part_mask_segmentation_score = query_part_mask_segmentation * score.view(b, -1, 1, 1)
        predict = query_part_mask_segmentation_score.max(dim=1)[0]

        return score, predict

    def get_postprocess_masks(self, masks, input_size, original_size):
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    pass


#
class OurComModel4MergePyramidal(nn.Module):

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.project_cross = MyCrossAttention(embedding_dim=256+1280, mlp_dim=2048,
                                              num_heads=8, attention_downsample_rate=2)
        pass

    def load_checkpoint(self, checkpoint):
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location=None)
                pass
            self.load_state_dict(state_dict, strict=True)
            Tools.print("load weights from {}".format(checkpoint))
        else:
            Tools.print("load weights error due to no checkpoint")
        pass

    def forward(self, query_part_map_features, support_features,
                support_ratio, query_part_mask_segmentation):
        # project
        query_feature = torch.cat(query_part_map_features, dim=-1)
        support_feature = torch.cat(support_features, dim=-1)
        query_feature, support_feature = self.project_cross(query_feature, support_feature)

        # sim
        sim_dot = support_feature @ query_feature.permute(0, 2, 1).contiguous()
        sim_norm = torch.norm(support_feature, dim=2).unsqueeze(2) \
                   @ torch.norm(query_feature, dim=2).unsqueeze(1)
        score = sim_dot / (sim_norm + 1e-8)

        # 3. Point
        # predict
        score_ratio = support_ratio.unsqueeze(-1) * score
        # 方案1，求平均
        score_ratio_sum = score_ratio.sum(dim=1) / (support_ratio.sum(dim=1).unsqueeze(-1) + 1e-8)
        # 方案2，取最大
        # score_ratio_sum = score_ratio.max(dim=1)[0]
        query_predict = query_part_mask_segmentation * score_ratio_sum.unsqueeze(-1).unsqueeze(-1)
        predict = query_predict.max(dim=1)[0]

        return score, predict

    def get_postprocess_masks(self, masks, input_size, original_size):
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    pass


class OurComModel4MergePyramidal2(nn.Module):

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.project_cross = MyCrossAttention(embedding_dim=256+1280, mlp_dim=2048,
                                              num_heads=8, attention_downsample_rate=2)
        pass

    def load_checkpoint(self, checkpoint):
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location=None)
                pass
            self.load_state_dict(state_dict, strict=True)
            Tools.print("load weights from {}".format(checkpoint))
        else:
            Tools.print("load weights error due to no checkpoint")
        pass

    def forward(self, query_part_map_features, support_features,
                support_ratio, query_part_mask_segmentation):
        # project
        query_feature = torch.cat(query_part_map_features, dim=-1)
        support_feature = torch.cat(support_features, dim=-1)
        query_feature, support_feature = self.project_cross(query_feature, support_feature)

        # sim
        sim_dot = support_feature @ query_feature.permute(0, 2, 1).contiguous()
        sim_norm = torch.norm(support_feature, dim=2).unsqueeze(2) \
                   @ torch.norm(query_feature, dim=2).unsqueeze(1)
        sim = sim_dot / (sim_norm + 1e-8)

        # 3. Point
        # predict
        score_ratio = support_ratio.unsqueeze(-1) * sim
        # 方案1，求平均
        score_ratio_sum = score_ratio.sum(dim=1) / (support_ratio.sum(dim=1).unsqueeze(-1) + 1e-8)
        # 方案2，取最大
        # score_ratio_sum = score_ratio.max(dim=1)[0]
        query_predict = query_part_mask_segmentation * score_ratio_sum.unsqueeze(-1).unsqueeze(-1)
        predict = query_predict.max(dim=1)[0]

        return score_ratio_sum, predict

    def get_postprocess_masks(self, masks, input_size, original_size):
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    pass


class OurComModel4MergePyramidalNorm(nn.Module):

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.project_cross = MyCrossAttention(embedding_dim=256+1280, mlp_dim=2048,
                                              num_heads=8, attention_downsample_rate=2)
        self.norm = nn.LayerNorm(1280)
        pass

    def load_checkpoint(self, checkpoint):
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location=None)
                pass
            self.load_state_dict(state_dict, strict=True)
            Tools.print("load weights from {}".format(checkpoint))
        else:
            Tools.print("load weights error due to no checkpoint")
        pass

    def forward(self, query_part_map_features, support_features,
                support_ratio, query_part_mask_segmentation):
        # project
        query_feature = torch.cat([
            query_part_map_features[0], self.norm(query_part_map_features[1])], dim=-1)
        support_feature = torch.cat([
            support_features[0], self.norm(support_features[1])], dim=-1)
        query_feature, support_feature = self.project_cross(query_feature, support_feature)

        # sim
        sim_dot = support_feature @ query_feature.permute(0, 2, 1).contiguous()
        sim_norm = torch.norm(support_feature, dim=2).unsqueeze(2) \
                   @ torch.norm(query_feature, dim=2).unsqueeze(1)
        sim = sim_dot / (sim_norm + 1e-8)

        # 3. Point
        # predict
        score_ratio = support_ratio.unsqueeze(-1) * sim
        # 方案1，求平均
        score_ratio_sum = score_ratio.sum(dim=1) / (support_ratio.sum(dim=1).unsqueeze(-1) + 1e-8)
        # 方案2，取最大
        # score_ratio_sum = score_ratio.max(dim=1)[0]
        query_predict = query_part_mask_segmentation * score_ratio_sum.unsqueeze(-1).unsqueeze(-1)
        predict = query_predict.max(dim=1)[0]

        return score_ratio_sum, predict

    def get_postprocess_masks(self, masks, input_size, original_size):
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    pass

