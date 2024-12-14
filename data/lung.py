r""" Chest X-ray few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import pickle
from util.util_tools import MyCommon
import numpy as np


class DatasetLung(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, num=600):
        self.split = split
        self.benchmark = 'lung'
        self.shot = shot
        self.num = num
        self.file_sets = set()

        self.base_path = os.path.join(datapath, 'LungSegmentation')
        self.img_path = os.path.join(self.base_path, 'CXR_png')
        self.ann_path = os.path.join(self.base_path, 'masks')

        self.categories = ['1']

        self.class_ids = range(0, 1)
        self.img_metadata_classwise = self.build_img_metadata_classwise()

        self.transform = transform

    def __len__(self):
        return self.num
    

    def set_exclude_file(self, file_sets):
        self.file_sets = file_sets
    
    def get_data_and_mask(self, pkl_name):
        pkl = pkl_name +  '.pkl'
        with open(pkl, "rb") as f:
            image_label_mask = pickle.load(f)

        # query_img = Image.fromarray(image_label_mask["image"])
        # query_label = Image.fromarray(image_label_mask["label"])
        query_masks = np.asarray([MyCommon.rle_to_mask(one)
                                 for one in image_label_mask["masks_target"]], dtype=np.int8)
        return query_masks

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks,\
             query_sam, support_sams = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        qry_sam_masks = torch.tensor(query_sam)
        #qry_sam_masks = F.interpolate(qry_sam_masks.unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_sam_masks = [torch.tensor(support_sam) for support_sam in support_sams]
        support_sam_masks = torch.stack(support_sam_masks, dim = 0) # shot x 40 x H X W
        #support_sam_masks = F.interpolate(support_sam_masks.float(), query_img.size()[-2:], mode='nearest')

        return support_imgs, support_masks, query_img, query_mask, class_sample, support_names, query_name, qry_sam_masks, support_sam_masks

    def load_frame(self, query_name, support_names):
        query_mask = self.read_mask(query_name)
        support_masks = [self.read_mask(name) for name in support_names]

        query_id = query_name[:-9] + '.png'
        #print(query_id)
        query_img = Image.open(os.path.join(self.img_path, os.path.basename(query_id))).convert('RGB')
        support_ids = [os.path.basename(name)[:-9] + '.png' for name in support_names]
        support_names = [os.path.join(self.img_path, sid) for sid in support_ids]
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        query_sam = self.get_data_and_mask(os.path.join(self.base_path, 'Lung-SAM-Mask', 'train', 'sam_mask_vit_b_t50_p32_s50', query_name[:-9].split('/')[-1]))
        support_sams =[self.get_data_and_mask(os.path.join(self.base_path, 'Lung-SAM-Mask', 'train', 'sam_mask_vit_b_t50_p32_s50'
                                                            ,supp_id.split('.')[0]))
                            for supp_id in support_ids]

        return query_img, query_mask, support_imgs, support_masks, query_sam, support_sams

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        class_id = idx % len(self.class_ids) # 
        class_sample = self.categories[class_id]

        while True:
            query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name not in self.file_sets: break
        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_id

    def build_img_metadata(self):
        img_metadata = []
        for cat in self.categories:
            os.path.join(self.base_path, cat)
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.img_path, cat))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'png':
                    img_metadata.append(img_path)
        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for cat in self.categories:
            img_metadata_classwise[cat] = []

        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob('%s/*' % self.ann_path)])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'png':
                    img_metadata_classwise[cat] += [img_path]
        return img_metadata_classwise
