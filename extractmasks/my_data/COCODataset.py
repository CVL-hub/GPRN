import os
import cv2
import torch
import pickle
import numpy as np
from tqdm import tqdm
from alisuretool.Tools import Tools
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src.my_transforms import MyResizeLongestSide


class COCOAll(Dataset):

    def __init__(self, root='data', data_split_root=None):
        self.data_path = root
        self.base_path = self.data_path
        self.data_split_root = data_split_root

        self.img_meta_data = self.build_img_metadata()
        self.img_meta_data = list(set(self.img_meta_data))
        self.dataset = []
        for img_name in self.img_meta_data:
            image_path = os.path.join(self.base_path, img_name)
            mask_path = os.path.join(self.base_path, 'seg_anno', img_name).replace(".jpg", ".png")
            self.dataset.append((img_name, image_path, mask_path))
            pass
        pass

    def build_img_metadata(self):

        def read_metadata(_split, _fold_id):
            data_split_root = "../../data/splits/coco" if self.data_split_root is None else self.data_split_root
            fold_n_metadata = os.path.join(f'{data_split_root}/{_split}/fold{_fold_id}.pkl')
            with open(fold_n_metadata, 'rb') as f:
                fold_n_metadata = pickle.load(f)
            return fold_n_metadata

        img_metadata = []
        for fold_id in [0, 1, 2, 3]:
            img_metadata_classwise= read_metadata('trn', fold_id)
            for k in img_metadata_classwise.keys():
                img_metadata += img_metadata_classwise[k]

            img_metadata_classwise= read_metadata('val', fold_id)
            for k in img_metadata_classwise.keys():
                img_metadata += img_metadata_classwise[k]
            pass
        return img_metadata

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_name, image_path, mask_path = self.dataset[idx]
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)
        return "feature", image_name, image, mask

    def getitem__image_name(self, idx):
        image_name, image_path, mask_path = self.dataset[idx]
        return "feature", image_name

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


class COCOFewShotFromMask(Dataset):

    def __init__(self, data_path, fold, split, shot, data_split_root=None, image_size=None):
        self.data_path = data_path
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.shot = shot
        self.fold = fold
        self.fold_num = 4
        self.n_class = 80
        self.eval_class_num = 20
        self.data_split_root = data_split_root

        self.img_metadata_classwise, self.img_meta_data = self.build_img_metadata()
        self.class_ids = [one for one in self.img_metadata_classwise if len(self.img_metadata_classwise[one]) > 0]

        self.image_size = image_size
        if self.image_size is not None:
            self.transform = MyResizeLongestSide(self.image_size)
        pass

    def build_img_metadata(self):

        def read_metadata(_split, _fold_id):
            data_split_root = "../../data/splits/coco" if self.data_split_root is None else self.data_split_root
            fold_n_metadata = os.path.join(f'{data_split_root}/{_split}/fold{_fold_id}.pkl')
            with open(fold_n_metadata, 'rb') as f:
                fold_n_metadata = pickle.load(f)
            return fold_n_metadata

        img_metadata = []
        img_metadata_classwise = read_metadata(self.split, self.fold)
        for k in img_metadata_classwise.keys():
            img_metadata += img_metadata_classwise[k]

        return img_metadata_classwise, sorted(list(set(img_metadata)))

    def __len__(self):
        return len(self.img_meta_data) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        query_image_path, support_images_path, class_id = self.sample_episode()
        try:
            query_feature, query_mask, support_features, support_masks, information = self.load_frame(
                query_image_path, support_images_path)
        except Exception:
            Tools.print(query_image_path)
            Tools.print(support_images_path)
            query_image_path, support_images_path, class_id = self.sample_episode()
            query_feature, query_mask, support_features, support_masks, information = self.load_frame(
                query_image_path, support_images_path)
            pass

        query_mask = self.extract_ignore_idx(torch.as_tensor(query_mask), class_id)
        support_masks = self.extract_ignore_idx(torch.as_tensor(support_masks), class_id)

        now_data = {'query_feature': query_feature, 'query_mask': query_mask, 'query_image_path': query_image_path,
                    'support_features': support_features, 'support_masks': support_masks,
                    'support_images_path': support_images_path,
                    'class_id': class_id, "information": information}
        return now_data

    def load_frame(self, query_image_path, support_images_path):
        # query_image_path
        with open(query_image_path, "rb") as f:
            result = pickle.load(f)
        query_feature, query_mask = result["image_feature"], result["original_mask"]
        query_mask = self.processing_image_and_pad_to_input(mask=query_mask)

        support_results = []
        for name in support_images_path:
            with open(name, "rb") as f:
                support_results.append(pickle.load(f))
            pass
        support_features = np.concatenate([one["image_feature"] for one in support_results], axis=0)
        support_masks = [self.processing_image_and_pad_to_input(
            mask=one["original_mask"])[None, :, :] for one in support_results]
        support_masks = np.concatenate(support_masks, axis=0)

        information = list(result["input_size"]) + list(result["original_size"]) + \
                      [result["original_image_name"], result["original_name"]] + \
                      [one["original_image_name"] for one in support_results]

        return query_feature, query_mask, support_features, support_masks, information

    @staticmethod
    def extract_ignore_idx(mask, class_id):
        mask[mask != class_id + 1] = 0
        mask[mask == class_id + 1] = 1
        return mask

    def processing_image_and_pad_to_input(self, mask):
        if self.image_size is not None:
            mask = self.transform.apply_mask(mask)[None, :, :]
            mask = torch.as_tensor(mask, dtype=torch.float)
            h, w = mask.shape[-2:]
            mask = F.pad(mask, (0, self.image_size - w, 0, self.image_size - h))
            pass
        return mask.squeeze()

    def sample_episode(self):
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]

        support_names = []
        while True:
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name and support_name not in support_names:
                support_names.append(support_name)
            if len(support_names) == self.shot:
                break
            pass

        query_pkl_path = os.path.join(self.data_path, "feature", query_name) + ".pkl"
        support_pkl_path = [os.path.join(self.data_path, "feature", one) + ".pkl" for one in support_names]
        return query_pkl_path, support_pkl_path, class_sample

    pass


if __name__ == '__main__':
    # dataset = COCOAll("/mnt/4T/Data/COCO")
    # train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    # for one in tqdm(train_dataloader):
    #     dataset.check(one)
    #     break
    #     pass

    dataset = COCOFewShotFromMask("/mnt/4T/Data/COCO/sam_feature", fold=0, split="trn", shot=5, image_size=1024)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)
    for one in dataloader:
        print(one)
        break
    pass
