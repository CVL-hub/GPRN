import torch
import numpy as np
from typing import Tuple
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image, InterpolationMode  # type: ignore


class MyResizeLongestSide:

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray, is_pad=False) -> np.ndarray:
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        image_result = np.array(resize(to_pil_image(image), target_size))
        if is_pad:
            result = np.zeros(shape=(self.target_length, self.target_length, image_result.shape[2]), dtype=np.uint8)
            result[0: image_result.shape[0], 0: image_result.shape[1]] = image_result
            return result
        return image_result

    def apply_feature(self, feature: torch.Tensor):
        target_size = self.get_preprocess_shape(feature.shape[-2], feature.shape[-1], self.target_length)
        feature_result = F.interpolate(feature, target_size, mode="bilinear", align_corners=False)
        return feature_result

    def apply_mask(self, mask: np.ndarray, is_pad=False, pad_value=0):
        target_size = self.get_preprocess_shape(mask.shape[0], mask.shape[1], self.target_length)
        mask_result = np.array(resize(to_pil_image(mask), size=target_size, interpolation=InterpolationMode.NEAREST))
        if is_pad:
            result = np.zeros(shape=(self.target_length, self.target_length), dtype=np.uint8) + pad_value
            result[0: mask_result.shape[0], 0: mask_result.shape[1]] = mask_result
            return result
        return mask_result

    def apply_xs(self, xs, size_x, size_y):
        scale = self.target_length * 1.0 / max(size_x, size_y)
        new_xs = [int(x * scale + 0.5) for x in xs]
        return new_xs

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        newh = int(newh + 0.5)
        neww = int(neww + 0.5)
        return (newh, neww)

    @staticmethod
    def pad(input_data, image_size):
        h, w = input_data.shape[-2:]
        return F.pad(input_data, (0, image_size - w, 0, image_size - h))

    pass
