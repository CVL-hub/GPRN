import os
import cv2
import time
import torch
import pickle
import numpy as np
from tqdm import tqdm
from alisuretool.Tools import Tools
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from src.my_data.utils_rle import rle_to_mask
from src.my_transforms import MyResizeLongestSide


class VOCAll(Dataset):

    def __init__(self, root='data', data_split_root=None):
        self.data_path = root
        self.img_path = os.path.join(self.data_path, 'VOCdevkit/VOC2012/JPEGImages/')
        self.ann_path = os.path.join(self.data_path, 'VOCdevkit/VOC2012/SegmentationClassAug/')
        self.data_split_root = data_split_root

        self.img_meta_data = self.build_img_metadata()
        self.dataset = []
        for img_name, img_class in self.img_meta_data:
            image_path = os.path.join(self.img_path, img_name) + '.jpg'
            mask_path = os.path.join(self.ann_path, img_name) + '.png'
            self.dataset.append((img_class, img_name, image_path, mask_path))
            pass
        pass

    def build_img_metadata(self):

        def read_metadata(_split, _fold_id):
            data_split_root = "../../data/splits/pascal" if self.data_split_root is None else self.data_split_root
            fold_n_metadata = os.path.join(f'{data_split_root}/{_split}/fold{_fold_id}.txt')
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        img_metadata += read_metadata('trn', 0)
        img_metadata += read_metadata('trn', 1)
        img_metadata += read_metadata('trn', 2)
        img_metadata += read_metadata('trn', 3)

        img_metadata += read_metadata('val', 0)
        img_metadata += read_metadata('val', 1)
        img_metadata += read_metadata('val', 2)
        img_metadata += read_metadata('val', 3)
        return img_metadata

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        name, image_name, image_path, mask_path = self.dataset[idx]
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)
        return name, image_name, image, mask

    @staticmethod
    def check(one):
        """
        """
        name, image_name, image, mask = one
        # 检查图片和mask是否匹配
        if image.shape[1] != mask.shape[1] or image.shape[2] != mask.shape[2]:
            print(name, image_name, image.shape, mask.shape)
        pass

    @staticmethod
    def has_error(image, mask):
        # 检查图片和mask是否匹配
        return image.shape[0] != mask.shape[0] or image.shape[1] != mask.shape[1]

    pass


class VOCFewShotFromMask(Dataset):

    def __init__(self, data_path, fold, split, shot, data_split_root=None, is_test=False):
        self.data_path = data_path
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.shot = shot
        self.fold = fold
        self.fold_num = 4
        self.n_class = 20
        self.eval_class_num = 5
        self.data_split_root = data_split_root

        self.img_meta_data = self.build_img_metadata()
        self.img_meta_data_class_wise = self.build_img_metadata_classwise()

        self.which_segmentation = "part_masks_info" if is_test else "part_mask_target_size_info"
        pass

    def build_img_metadata(self):

        def read_metadata(_split, _fold_id):
            data_split_root = "../../data/splits/pascal" if self.data_split_root is None else self.data_split_root
            fold_n_metadata = os.path.join(f'{data_split_root}/{_split}/fold{_fold_id}.txt')
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.fold_num):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)
            pass
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_class_wise = {}
        for class_id in range(self.n_class):
            img_metadata_class_wise[class_id] = []

        for img_name, img_class in self.img_meta_data:
            img_metadata_class_wise[img_class] += [img_name]
        return img_metadata_class_wise

    def __len__(self):
        # return len(self.img_meta_data) if self.split == 'trn' else 1000
        return len(self.img_meta_data)

    def __getitem__(self, idx):
        if self.split == 'val' and len(self) == 1000:
            idx = np.random.randint(len(self.img_meta_data))
        query_image_path, support_images_path, class_id = self.sample_episode(idx)
        try:
            now_data = self.load_frame(query_image_path, support_images_path)

            query_mask, query_ignore_idx = self.extract_ignore_idx(
                torch.as_tensor(now_data["query_label_mask"]), class_id)
            now_data['class_id'] = class_id
            now_data['query_label_mask'] = query_mask
            now_data['query_ignore_idx'] = query_ignore_idx
        except Exception:
            Tools.print("Error {} {}".format(query_image_path, support_images_path))
            return self.__getitem__(idx=np.random.randint(len(self.img_meta_data)))
        return now_data

    def load_frame(self, query_image_path, support_images_path):
        # query_image_path
        with open(query_image_path, "rb") as f:
            result = pickle.load(f)

        query_masked_average_pooling_feature = result["masked_average_pooling"].squeeze(axis=0)
        query_part_masked_average_pooling_feature = result["part_masked_average_pooling"].squeeze(axis=0)
        if isinstance(result[self.which_segmentation][0]["segmentation"], dict):
            query_part_mask_segmentation = np.concatenate(
                [rle_to_mask(one["segmentation"])[np.newaxis, :, :]
                 for one in result[self.which_segmentation]], axis=0)
        else:
            query_part_mask_segmentation = np.concatenate(
                [one["segmentation"][np.newaxis, :, :] for one in result[self.which_segmentation]], axis=0)
            pass
        query_part_mask_segmentation = np.array(query_part_mask_segmentation, dtype=np.float32)
        query_part_mask_ratio = np.array([one["ratio"] for one in result[self.which_segmentation]])
        query_part_mask_ratio = np.array(query_part_mask_ratio, dtype=np.float32)
        query_part_mask_area = np.array([one["area"] for one in result[self.which_segmentation]])
        query_part_mask_area = np.array(query_part_mask_area, dtype=np.float32)
        query_original_mask = np.array(result["original_mask"], dtype=np.float32)

        _transform_mask = MyResizeLongestSide(target_length=query_part_mask_segmentation.shape[-1])
        query_label_mask = _transform_mask.apply_mask(result["original_mask"], is_pad=True)
        query_label_mask = np.array(query_label_mask, dtype=np.float32)

        support_results = []
        for name in support_images_path:
            with open(name, "rb") as f:
                support_results.append(pickle.load(f))
            pass

        support_masked_average_pooling_features = np.concatenate(
            [one["masked_average_pooling"] for one in support_results], axis=0)
        support_part_masked_average_pooling_features = np.concatenate(
            [one["part_masked_average_pooling"] for one in support_results], axis=0)

        if isinstance(support_results[0][self.which_segmentation][0]["segmentation"], dict):
            support_part_mask_segmentation = np.concatenate([np.concatenate(
                [rle_to_mask(one["segmentation"])[np.newaxis, :, :] for one in support_result[self.which_segmentation]
                 ], axis=0)[None, :] for support_result in support_results], axis=0)
        else:
            support_part_mask_segmentation = np.concatenate([np.concatenate(
                [one["segmentation"][np.newaxis, :, :] for one in support_result[self.which_segmentation]
                 ], axis=0)[None, :] for support_result in support_results], axis=0)
            pass

        support_part_mask_segmentation = np.array(support_part_mask_segmentation, dtype=np.float32)
        support_part_mask_ratio = np.concatenate([
            np.array([one["ratio"] for one in support_result[self.which_segmentation]])[None, :]
            for support_result in support_results], axis=0)
        support_part_mask_ratio = np.array(support_part_mask_ratio, dtype=np.float32)
        support_part_mask_area = np.concatenate([
            np.array([one["area"] for one in support_result[self.which_segmentation]])[None, :]
            for support_result in support_results], axis=0)
        support_part_mask_area = np.array(support_part_mask_area, dtype=np.float32)
        support_original_mask = np.concatenate(
            [support_result["original_mask"][None, :] for support_result in support_results], axis=0)
        support_original_mask = np.array(support_original_mask, dtype=np.float32)

        information = list(result[self.which_segmentation][0]["crop_box"][-2:]) + \
                      list(query_original_mask.shape) + \
                      [result["original_image_name"], result["original_name"]] + \
                      [one["original_image_name"] for one in support_results]

        result = {"query_masked_average_pooling_feature": query_masked_average_pooling_feature,
                  "query_part_masked_average_pooling_feature": query_part_masked_average_pooling_feature,
                  "query_part_mask_segmentation": query_part_mask_segmentation,
                  "query_part_mask_ratio": query_part_mask_ratio,
                  "query_part_mask_area": query_part_mask_area,
                  "query_label_mask": query_label_mask,
                  # "query_original_mask": query_original_mask,

                  "support_masked_average_pooling_features": support_masked_average_pooling_features,
                  "support_part_masked_average_pooling_features": support_part_masked_average_pooling_features,
                  "support_part_mask_segmentation": support_part_mask_segmentation,
                  "support_part_mask_ratio": support_part_mask_ratio,
                  "support_part_mask_area": support_part_mask_area,
                  # "support_original_mask": support_original_mask,

                  "information": information,
                  "query_image_path": query_image_path,
                  "support_images_path": support_images_path,
                  }

        if not self.split == "trn":
            result["query_original_mask"] = query_original_mask
            result["support_original_mask"] = support_original_mask
            pass
        return result

    @staticmethod
    def extract_ignore_idx(mask, class_id):
        boundary = (mask / 255).floor()
        mask[mask != class_id + 1] = 0
        mask[mask == class_id + 1] = 1
        return mask, boundary

    def sample_episode(self, idx):
        query_name, class_id = self.img_meta_data[idx]

        support_names = []
        while True:
            support_name = np.random.choice(self.img_meta_data_class_wise[class_id], 1, replace=False)[0]
            if query_name != support_name and support_name not in support_names:
                support_names.append(support_name)
            if len(support_names) == self.shot:
                break
            pass

        query_pkl_path = os.path.join(self.data_path, str(class_id), query_name) + ".pkl"
        support_pkl_path = [os.path.join(self.data_path, str(class_id), one) + ".pkl" for one in support_names]
        return query_pkl_path, support_pkl_path, class_id

    pass


class VOCOneShotFromFinal(Dataset):

    def __init__(self, data_final_path, fold, split, data_split_root=None, is_test=False):
        self.dataset_name = "voc"
        self.data_final_path = data_final_path
        self.data_split_root = data_split_root
        self.fold = fold  # not used
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.is_test = is_test

        self.fold_num = 4
        self.n_class = 20
        self.eval_class_num = 5
        if self.split == "trn" and self.is_test:
            self.eval_class_num = 15

        self.img_meta_data = self.build_img_metadata()
        self.img_meta_data_class_wise = self.build_img_metadata_classwise()
        pass

    def build_img_metadata(self):

        def read_metadata(_split, _fold_id):
            data_split_root = "../../data/splits/pascal" if self.data_split_root is None else self.data_split_root
            fold_n_metadata = os.path.join(f'{data_split_root}/{_split}/fold{_fold_id}.txt')
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.fold_num):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)
            pass
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_class_wise = {}
        for class_id in range(self.n_class):
            img_metadata_class_wise[class_id] = []

        for img_name, img_class in self.img_meta_data:
            img_metadata_class_wise[img_class] += [img_name]
        return img_metadata_class_wise

    def __len__(self):
        return 1000 if self.is_test else len(self.img_meta_data)

    def __getitem__(self, idx):
        if self.is_test:
            idx = np.random.randint(len(self.img_meta_data))

        query_image_path, support_image_path, class_id = self.sample_episode(idx)

        with open(query_image_path, "rb") as f:
            query_final_data = pickle.load(f)
        with open(support_image_path, "rb") as f:
            support_final_data = pickle.load(f)

        which_rle = "part_mask_segmentation_rle_original" if self.is_test else "part_mask_segmentation_rle"
        which_label = "label_original" if self.is_test else "label_target"
        query_part_mask_segmentation = np.asarray(np.concatenate(
            [rle_to_mask(one)[None, :] for one in query_final_data[which_rle]]), dtype=np.float32)
        support_part_mask_segmentation = np.asarray(np.concatenate(
            [rle_to_mask(one)[None, :] for one in support_final_data[which_rle]]), dtype=np.float32)

        # 计算错误，重新计算ratio
        query_label_original = np.array(query_final_data[which_label])
        query_label_original[query_label_original != class_id + 1] = 0
        query_label_original[query_label_original == class_id + 1] = 1
        query_part_mask_ratio = np.sum(np.sum(query_part_mask_segmentation *
                                              query_label_original[None, :], axis=1), axis=1)\
                                / (np.sum(np.sum(query_part_mask_segmentation, axis=1), axis=1) + 1e-8)
        support_label_original = np.array(support_final_data[which_label])
        support_label_original[support_label_original != class_id + 1] = 0
        support_label_original[support_label_original == class_id + 1] = 1
        support_part_mask_ratio = np.sum(np.sum(support_part_mask_segmentation *
                                                support_label_original[None, :], axis=1), axis=1)\
                                  / (np.sum(np.sum(support_part_mask_segmentation, axis=1), axis=1) + 1e-8)

        now_data = {"query_label_target": query_final_data[which_label],
                    "support_label_target": support_final_data[which_label],

                    "query_part_mask_ratio": query_part_mask_ratio,
                    "support_part_mask_ratio": support_part_mask_ratio,

                    "query_map_feature": query_final_data["map_feature"],
                    "support_map_feature": support_final_data["map_feature"],

                    "query_part_map_feature": query_final_data["part_map_feature"],
                    "support_part_map_feature": support_final_data["part_map_feature"],

                    'query_information': query_final_data['information'],
                    'support_information': support_final_data['information'],

                    "query_part_mask_segmentation": query_part_mask_segmentation,
                    "support_part_mask_segmentation": support_part_mask_segmentation,

                    'query_image_path': query_image_path,
                    'support_image_path': support_image_path,
                    'class_id': class_id}
        return now_data

    def sample_episode(self, idx):
        query_name, class_id = self.img_meta_data[idx]
        suppo_name = np.random.choice(self.img_meta_data_class_wise[class_id], 1, replace=False)[0]

        query_image_path = os.path.join(self.data_final_path, str(class_id), query_name) + ".pkl"
        suppo_image_path = os.path.join(self.data_final_path, str(class_id), suppo_name) + ".pkl"
        return query_image_path, suppo_image_path, class_id

    pass


if __name__ == '__main__':
    # dataset = VOCAll("/mnt/4T/Data/VOC2012")
    # train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    # for one in tqdm(train_dataloader):
    #     dataset.check(one)
    #     pass

    # dataset = VOCFewShotFromMask("/mnt/4T/Data/VOC2012/sam_feature", fold=0, split="trn", shot=1, image_size=1024)
    dataset = VOCFewShotFromMask(
        "/media/ubuntu/405411e3-8fa3-4178-8754-55090363734b/ALISURE/VOC2012/sam_mask_vit_h_t64_n128_p32",
        fold=0, split="trn", shot=1)
    dataloader = DataLoader(dataset, batch_size=6, shuffle=False, num_workers=4)
    for index, one in enumerate(dataloader):
        print(index)
        pass
    pass
