import os
import cv2
import torch
import pickle
import numpy as np
from tqdm import tqdm
from glob import glob
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from src.my_data.utils_rle import rle_to_mask
from src.my_transforms import MyResizeLongestSide


"""
1. 加载全部数据，用于提取特征  OK
2. 加载训练数据，用于训练 OK
3. 加载测试数据，用于测试 OK
"""


class FSS1000All(Dataset):

    def __init__(self, root='data', data_split_root=None):
        self.root = root
        self.fss1000_dir = os.path.join(self.root, 'fewshot_data')
        self.classes_name = sorted(os.listdir(self.fss1000_dir))
        self.data_split_root = data_split_root

        self.dataset = []
        for name in self.classes_name:
            class_dir = os.path.join(self.fss1000_dir, name)
            image_paths = sorted(glob(os.path.join(class_dir, "*.jpg")))
            for image_path in image_paths:
                image_name = os.path.basename(image_path).split(".jpg")[0]
                mask_path = os.path.join(class_dir, "{}.png".format(image_name))
                self.dataset.append((name, image_name, image_path, mask_path))
            pass
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        name, image_name, image_path, mask_path = self.dataset[idx]
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY) // 128
        return name, image_name, image, mask

    @staticmethod
    def check(one):
        """
            ['crt_screen'] ['6'] torch.Size([1, 500, 500, 3]) torch.Size([1, 500, 500])
            ['shower_curtain'] ['3'] torch.Size([1, 450, 450, 3]) torch.Size([1, 450, 450])
            ['rocking_chair'] ['9'] torch.Size([1, 550, 640, 3]) torch.Size([1, 550, 640])
            ['peregine_falcon'] ['8'] torch.Size([1, 224, 224, 3]) torch.Size([1, 480, 960])
            ['ruler'] ['10'] torch.Size([1, 700, 1024, 3]) torch.Size([1, 700, 1024])
            ['screw'] ['8'] torch.Size([1, 424, 562, 3]) torch.Size([1, 424, 562])
            ['rubber_eraser'] ['8'] torch.Size([1, 622, 629, 3]) torch.Size([1, 622, 629])
        """
        name, image_name, image, mask = one
        # 检查图片和mask是否匹配
        if image.shape[1] != mask.shape[1] or image.shape[2] != mask.shape[2] \
                or image.shape[1] != 224 or image.shape[2] != 224:
            print(name, image_name, image.shape, mask.shape)
        pass

    @staticmethod
    def has_error(image, mask):
        """
            ['peregine_falcon'] ['8'] torch.Size([1, 224, 224, 3]) torch.Size([1, 480, 960])
        """
        # 检查图片和mask是否匹配
        return image.shape[0] != mask.shape[0] or image.shape[1] != mask.shape[1]

    pass


class FSS1000FewShotFromMask(Dataset):

    def __init__(self, data_path, fold, split, shot, data_split_root=None, is_test=False):
        self.dataset_name = "fss"
        self.data_path = data_path
        self.fold = fold  # not used
        self.split = split
        self.shot = shot
        self.data_split_root = data_split_root

        data_split_root = "../../data/splits/fss" if self.data_split_root is None else self.data_split_root
        with open(f'{data_split_root}/{split}.txt', 'r') as f:
            self.categories = f.read().split('\n')[:-1]
        self.categories = sorted(self.categories)
        self.eval_class_num = len(self.categories)

        self.feature_meta_data = self.build_feature_metadata()

        self.which_segmentation = "part_masks_info" if is_test else "part_mask_target_size_info"
        pass

    def build_feature_metadata(self):
        feature_meta_data = []
        for cat in self.categories:
            feature_meta_data.extend(sorted([path for path in glob(os.path.join(self.data_path, cat, "*.pkl"))]))
            pass
        # filter error
        feature_meta_data = [one for one in feature_meta_data if "peregine_falcon/8" not in one]
        feature_meta_data = [one for one in feature_meta_data if "bamboo_slip/7" not in one]
        return feature_meta_data

    def __len__(self):
        return len(self.feature_meta_data)

    def __getitem__(self, idx):
        query_image_path = idx
        try:
            query_image_path, support_images_path, class_sample = self.sample_episode(idx)
            now_data = self.load_frame(query_image_path, support_images_path)
            now_data['class_id'] = class_sample
        except Exception:
            print("error {}".format(query_image_path))
            return self.__getitem__(idx=np.random.randint(self.__len__()))
            pass
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

    def sample_episode(self, idx):
        query_image_path = self.feature_meta_data[idx]
        class_id = self.categories.index(query_image_path.split('/')[-2])

        # support images
        # --------------------------------------------------------------------------------------------------------
        # 过滤掉一个不好的图像：peregine_falcon/8
        all_support_image_path = sorted(glob(os.path.join(os.path.dirname(query_image_path), "*.pkl")))
        all_support_image_path = [one for one in all_support_image_path if "peregine_falcon/8" not in one]
        all_support_image_path = [one for one in all_support_image_path if "bamboo_slip/7" not in one]
        # --------------------------------------------------------------------------------------------------------
        del all_support_image_path[all_support_image_path.index(query_image_path)]
        choice_index = np.random.choice(len(all_support_image_path), self.shot, replace=False)
        support_images_path = [all_support_image_path[i] for i in choice_index]

        return query_image_path, support_images_path, class_id

    pass


# OK
class FSS1000OneShotFromFinal(Dataset):

    def __init__(self, data_final_path, fold, split, data_split_root=None, is_test=False):
        self.dataset_name = "fss"
        self.data_final_path = data_final_path
        self.data_split_root = data_split_root
        self.fold = fold  # not used
        self.split = split
        self.is_test = is_test

        data_split_root = "../../data/splits/fss" if self.data_split_root is None else self.data_split_root
        with open(f'{data_split_root}/{split}.txt', 'r') as f:
            self.categories = f.read().split('\n')[:-1]
        self.categories = sorted(self.categories)
        self.eval_class_num = len(self.categories)

        self.mask_meta_data = self.build_mask_metadata(self.data_final_path)
        pass

    def build_mask_metadata(self, data_mask_path):
        mask_meta_data = []
        for cat in self.categories:
            mask_meta_data.extend(sorted([path for path in glob(os.path.join(data_mask_path, cat, "*.pkl"))]))
            pass
        # filter error
        mask_meta_data = [one for one in mask_meta_data if "peregine_falcon/8" not in one]
        mask_meta_data = [one for one in mask_meta_data if "bamboo_slip/7" not in one]
        return mask_meta_data

    def __len__(self):
        if self.is_test and self.split == "trn":
            return 2000
        return len(self.mask_meta_data)

    def __getitem__(self, idx):
        if self.is_test and self.split == "trn":
            idx = np.random.randint(len(self.mask_meta_data))
            pass

        query_image_path, support_image_path, class_id = self.sample_episode(idx)

        with open(query_image_path, "rb") as f:
            query_final_data = pickle.load(f)
        with open(support_image_path, "rb") as f:
            support_final_data = pickle.load(f)

        which_rle = "part_mask_segmentation_rle_original" if self.is_test else "part_mask_segmentation_rle"
        which_label = "label_original" if self.is_test else "label_target"
        query_part_mask_segmentation = np.asarray(np.concatenate(
            [rle_to_mask(one)[None, :] for one in query_final_data[which_rle]]), dtype=np.float32)
        # 为了加速，过滤下面的掩码
        # support_part_mask_segmentation = np.asarray(np.concatenate(
        #     [rle_to_mask(one)[None, :] for one in support_final_data[which_rle]]), dtype=np.float32)

        now_data = {"query_label_target": query_final_data[which_label],
                    "support_label_target": support_final_data[which_label],

                    "query_part_mask_ratio": query_final_data["part_mask_ratio"],
                    "support_part_mask_ratio": support_final_data["part_mask_ratio"],

                    "query_map_feature": query_final_data["map_feature"],
                    "support_map_feature": support_final_data["map_feature"],

                    "query_part_map_feature": query_final_data["part_map_feature"],
                    "support_part_map_feature": support_final_data["part_map_feature"],

                    'query_information': query_final_data['information'],
                    'support_information': support_final_data['information'],

                    "query_part_mask_segmentation": query_part_mask_segmentation,
                    # "support_part_mask_segmentation": support_part_mask_segmentation,
                    "support_part_mask_segmentation": support_final_data["part_map_feature"],

                    'query_image_path': query_image_path,
                    'support_image_path': support_image_path,
                    'class_id': class_id}
        return now_data

    def sample_episode(self, idx):
        query_image_path = self.mask_meta_data[idx]
        class_id = self.categories.index(query_image_path.split('/')[-2])

        # support image
        # --------------------------------------------------------------------------------------------------------
        # 过滤掉一个不好的图像：peregine_falcon/8
        all_support_image_path = sorted(glob(os.path.join(os.path.dirname(query_image_path), "*.pkl")))
        all_support_image_path = [one for one in all_support_image_path if "peregine_falcon/8" not in one]
        all_support_image_path = [one for one in all_support_image_path if "bamboo_slip/7" not in one]
        # --------------------------------------------------------------------------------------------------------
        del all_support_image_path[all_support_image_path.index(query_image_path)]
        choice_index = np.random.choice(len(all_support_image_path), 1, replace=False)
        support_images_path = all_support_image_path[choice_index[0]]

        return query_image_path, support_images_path, class_id

    pass


class FSS1000OneShotFromFinalPart(Dataset):

    def __init__(self, data_final_path, fold, split, data_split_root=None, is_test=False):
        self.dataset_name = "fss"
        self.data_final_path = data_final_path
        self.data_split_root = data_split_root
        self.fold = fold  # not used
        self.split = split
        self.is_test = is_test

        data_split_root = "../../data/splits/fss" if self.data_split_root is None else self.data_split_root
        with open(f'{data_split_root}/{split}.txt', 'r') as f:
            self.categories = f.read().split('\n')[:-1]
        self.categories = sorted(self.categories)
        self.eval_class_num = len(self.categories)

        self.mask_meta_data = self.build_mask_metadata(self.data_final_path)
        pass

    def build_mask_metadata(self, data_mask_path):
        mask_meta_data = []
        for cat in self.categories:
            mask_meta_data.extend(sorted([path for path in glob(os.path.join(data_mask_path, cat, "*.pkl"))]))
            pass
        # filter error
        mask_meta_data = [one for one in mask_meta_data if "peregine_falcon/8" not in one]
        mask_meta_data = [one for one in mask_meta_data if "bamboo_slip/7" not in one]
        return mask_meta_data

    def __len__(self):
        if self.is_test and self.split == "trn":
            return 2000
        return len(self.mask_meta_data)

    def __getitem__(self, idx):
        if self.is_test and self.split == "trn":
            idx = np.random.randint(len(self.mask_meta_data))
            pass

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

        #############################################################################################
        # 过滤小的部件
        # seg_shape = support_final_data["part_mask_segmentation_rle"][0]["size"]
        # seg_size_th = (seg_shape[0] * seg_shape[1]) // 16
        # seg_size = [one.sum() < seg_size_th for one in support_part_mask_segmentation]
        # support_final_data["part_mask_ratio"][seg_size] = 0.0
        # support_final_data["part_map_feature"][seg_size] = support_final_data["part_map_feature"][seg_size] * 0.0
        #############################################################################################

        now_data = {"query_label_target": query_final_data[which_label],
                    "support_label_target": support_final_data[which_label],

                    "query_part_mask_ratio": query_final_data["part_mask_ratio"],
                    "support_part_mask_ratio": support_final_data["part_mask_ratio"],

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
        query_image_path = self.mask_meta_data[idx]
        class_id = self.categories.index(query_image_path.split('/')[-2])

        # support image
        # --------------------------------------------------------------------------------------------------------
        # 过滤掉一个不好的图像：peregine_falcon/8
        all_support_image_path = sorted(glob(os.path.join(os.path.dirname(query_image_path), "*.pkl")))
        all_support_image_path = [one for one in all_support_image_path if "peregine_falcon/8" not in one]
        all_support_image_path = [one for one in all_support_image_path if "bamboo_slip/7" not in one]
        # --------------------------------------------------------------------------------------------------------
        del all_support_image_path[all_support_image_path.index(query_image_path)]
        choice_index = np.random.choice(len(all_support_image_path), 1, replace=False)
        support_images_path = all_support_image_path[choice_index[0]]

        return query_image_path, support_images_path, class_id

    pass


# OK
class FSS1000OneShotFromFinalMerge(Dataset):

    def __init__(self, data_final_path, fold, split, data_split_root=None, is_test=False):
        self.dataset_name = "fss"
        self.data_final_path = data_final_path
        self.data_split_root = data_split_root
        self.fold = fold  # not used
        self.split = split
        self.is_test = is_test

        data_split_root = "../../data/splits/fss" if self.data_split_root is None else self.data_split_root
        with open(f'{data_split_root}/{split}.txt', 'r') as f:
            self.categories = f.read().split('\n')[:-1]
        self.categories = sorted(self.categories)
        self.eval_class_num = len(self.categories)

        self.mask_meta_data = self.build_mask_metadata(self.data_final_path)
        pass

    def build_mask_metadata(self, data_mask_path):
        mask_meta_data = []
        for cat in self.categories:
            mask_meta_data.extend(sorted([path for path in glob(os.path.join(data_mask_path, cat, "*.pkl"))]))
            pass
        # filter error
        mask_meta_data = [one for one in mask_meta_data if "peregine_falcon/8" not in one]
        mask_meta_data = [one for one in mask_meta_data if "bamboo_slip/7" not in one]
        return mask_meta_data

    def __len__(self):
        # 4. Point
        if self.is_test:
            return 2400
        if self.is_test and self.split == "trn":
            return 500
        return len(self.mask_meta_data)

    def __getitem__(self, idx):
        if self.is_test:
        # if self.is_test and self.split == "trn":
            idx = np.random.randint(len(self.mask_meta_data))
            pass

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

        now_data = {"query_label_target": query_final_data[which_label],
                    "support_label_target": support_final_data[which_label],

                    "query_part_mask_ratio": query_final_data["part_mask_ratio"],
                    "support_part_mask_ratio": support_final_data["part_mask_ratio"],

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
        query_image_path = self.mask_meta_data[idx]
        class_id = self.categories.index(query_image_path.split('/')[-2])

        # support image
        # --------------------------------------------------------------------------------------------------------
        # 过滤掉一个不好的图像：peregine_falcon/8
        all_support_image_path = sorted(glob(os.path.join(os.path.dirname(query_image_path), "*.pkl")))
        all_support_image_path = [one for one in all_support_image_path if "peregine_falcon/8" not in one]
        all_support_image_path = [one for one in all_support_image_path if "bamboo_slip/7" not in one]
        # --------------------------------------------------------------------------------------------------------
        del all_support_image_path[all_support_image_path.index(query_image_path)]
        choice_index = np.random.choice(len(all_support_image_path), 1, replace=False)
        support_images_path = all_support_image_path[choice_index[0]]

        return query_image_path, support_images_path, class_id

    pass


class FSS1000OneShotFromFinalMergePyramidal(Dataset):

    def __init__(self, data_final_path, fold, split, data_split_root=None, is_test=False):
        self.dataset_name = "fss"
        self.data_final_path = data_final_path
        self.data_split_root = data_split_root
        self.fold = fold  # not used
        self.split = split
        self.is_test = is_test

        data_split_root = "../../data/splits/fss" if self.data_split_root is None else self.data_split_root
        with open(f'{data_split_root}/{split}.txt', 'r') as f:
            self.categories = f.read().split('\n')[:-1]
        self.categories = sorted(self.categories)
        self.eval_class_num = len(self.categories)

        self.mask_meta_data = self.build_mask_metadata(self.data_final_path)
        pass

    def build_mask_metadata(self, data_mask_path):
        mask_meta_data = []
        for cat in self.categories:
            mask_meta_data.extend(sorted([path for path in glob(os.path.join(data_mask_path, cat, "*.pkl"))]))
            pass
        # filter error
        mask_meta_data = [one for one in mask_meta_data if "peregine_falcon/8" not in one]
        mask_meta_data = [one for one in mask_meta_data if "bamboo_slip/7" not in one]
        return mask_meta_data

    def __len__(self):
        # 4. Point
        if self.is_test and self.split == "trn":
            return 500
        return len(self.mask_meta_data)

    def __getitem__(self, idx):
        if self.is_test and self.split == "trn":
            idx = np.random.randint(len(self.mask_meta_data))
            pass

        query_image_path, support_image_path, class_id = self.sample_episode(idx)
        with open(query_image_path, "rb") as f:
            query_final_data = pickle.load(f)
        with open(support_image_path, "rb") as f:
            support_final_data = pickle.load(f)
        with open(query_image_path.replace("1280", "256"), "rb") as f:
            query_final_data_256 = pickle.load(f)
        with open(support_image_path.replace("1280", "256"), "rb") as f:
            support_final_data_256 = pickle.load(f)

        which_rle = "part_mask_segmentation_rle_original" if self.is_test else "part_mask_segmentation_rle"
        which_label = "label_original" if self.is_test else "label_target"
        query_part_mask_segmentation = np.asarray(np.concatenate(
            [rle_to_mask(one)[None, :] for one in query_final_data[which_rle]]), dtype=np.float32)
        support_part_mask_segmentation = np.asarray(np.concatenate(
            [rle_to_mask(one)[None, :] for one in support_final_data[which_rle]]), dtype=np.float32)

        now_data = {"query_label_target": query_final_data[which_label],
                    "support_label_target": support_final_data[which_label],

                    "query_part_mask_ratio": query_final_data["part_mask_ratio"],
                    "support_part_mask_ratio": support_final_data["part_mask_ratio"],

                    "query_map_feature_1280": query_final_data["map_feature"],
                    "query_map_feature_256": query_final_data_256["map_feature"],
                    "support_map_feature_1280": support_final_data["map_feature"],
                    "support_map_feature_256": support_final_data_256["map_feature"],

                    "query_part_map_feature_1280": query_final_data["part_map_feature"],
                    "query_part_map_feature_256": query_final_data_256["part_map_feature"],
                    "support_part_map_feature_1280": support_final_data["part_map_feature"],
                    "support_part_map_feature_256": support_final_data_256["part_map_feature"],

                    'query_information': query_final_data['information'],
                    'support_information': support_final_data['information'],

                    "query_part_mask_segmentation": query_part_mask_segmentation,
                    "support_part_mask_segmentation": support_part_mask_segmentation,

                    'query_image_path': query_image_path,
                    'support_image_path': support_image_path,
                    'class_id': class_id}
        return now_data

    def sample_episode(self, idx):
        query_image_path = self.mask_meta_data[idx]
        class_id = self.categories.index(query_image_path.split('/')[-2])

        # support image
        # --------------------------------------------------------------------------------------------------------
        # 过滤掉一个不好的图像：peregine_falcon/8
        all_support_image_path = sorted(glob(os.path.join(os.path.dirname(query_image_path), "*.pkl")))
        all_support_image_path = [one for one in all_support_image_path if "peregine_falcon/8" not in one]
        all_support_image_path = [one for one in all_support_image_path if "bamboo_slip/7" not in one]
        # --------------------------------------------------------------------------------------------------------
        del all_support_image_path[all_support_image_path.index(query_image_path)]
        choice_index = np.random.choice(len(all_support_image_path), 1, replace=False)
        support_images_path = all_support_image_path[choice_index[0]]

        return query_image_path, support_images_path, class_id

    pass


if __name__ == '__main__':
    # dataset = FSS1000All("/mnt/4T/Data/FSS-1000")
    # train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    # for one in tqdm(train_dataloader):
    #     dataset.check(one)
    #     pass

    # dataset = FSS1000FewShotFromMask("/mnt/4T/Data/FSS-1000/sam_mask_vit_h_t64_n32_p8", fold=0, split="trn", shot=1)
    dataset = FSS1000FewShotFromMask("/mnt/4T/Data/FSS-1000/sam_mask_vit_h_t64_n32_p8", fold=0, split="test", shot=1)
    # dataset = FSS1000FewShotFromMask("/mnt/4T/Data/FSS-1000/sam_mask_vit_h_t64_n128_p32_", fold=0, split="trn", shot=1)
    # dataset = FSS1000FewShotFromMask("/mnt/4T/Data/FSS-1000/sam_mask_vit_h_t64_n128_p32_", fold=0, split="test", shot=1)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    for one in tqdm(dataloader):
        a = one["information"][5][0] == os.path.split(os.path.split(one["query_image_path"][0])[0])[1]
        b = one["information"][5][0] == os.path.split(os.path.split(one["support_images_path"][0][0])[0])[1]
        pass
    pass
