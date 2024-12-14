import torch
import numpy as np


def my_masked_average_pooling(feature, mask):
    # b, d, w, h / b, n, w, h
    _feature = feature.view(feature.shape[0], feature.shape[1], -1).permute(0, 2, 1).contiguous()
    _mask = mask.view(mask.shape[0], mask.shape[1], -1)
    featured_sum = _mask @ _feature

    masked_sum = torch.repeat_interleave(torch.sum(mask, dim=[2, 3]).unsqueeze(2), featured_sum.shape[-1], dim=2)

    masked_average_pooling = torch.div(featured_sum, masked_sum + 1e-8)
    return masked_average_pooling


def my_masked_average_pooling_bbox(feature, bbox):
    x, y, w, h = bbox
    if w > 0 and h > 0:
        bbox_feature = feature[:, :, x: x + w, y: y + h]
        masked_average_pooling = bbox_feature.mean(dim=[-1, -2])
    else:
        masked_average_pooling = torch.tensor(np.zeros(feature.shape[:2]), dtype=feature.dtype)
    return masked_average_pooling

