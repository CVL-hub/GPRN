r""" FSS-1000 few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import pickle
from util.util_tools import MyCommon
import albumentations as A
import random


class DatasetFSSIFA(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, num=1):
        self.split = split
        self.benchmark = 'fss'
        self.shot = shot

        self.base_path = os.path.join(datapath, 'FSS-1000')

        with open('./data/splits/fss/%s.txt' % split, 'r') as f:
            self.categories = f.read().split('\n')[:-1]
        self.categories = sorted(self.categories)

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()

        self.transform = transform

        self.q_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90()
        ])

        self.q_transform2 = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(hue_shift_limit=20,sat_shift_limit=30,val_shift_limit=20,always_apply=False,p=1)
        ])

        self.num = num

    def __len__(self):
        return self.num
    

    def get_data_and_mask(self, pkl_name):
        pkl = pkl_name +  '.pkl'
        with open(pkl, "rb") as f:
            image_label_mask = pickle.load(f)

        # query_img = Image.fromarray(image_label_mask["image"])
        # query_label = Image.fromarray(image_label_mask["label"])
        query_masks = np.asarray([MyCommon.rle_to_mask_pytorch(one)
                                for one in image_label_mask["masks"]], dtype=np.int8)
        return query_masks

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        _, _, support_imgs, support_masks ,\
            query_sam, support_sams = self.load_frame(query_name, support_names)

        if self.shot == 1:
            idx = 0
        else:
            idx = random.randint(0, self.shot-1)
        q_img = support_imgs[idx]
        q_img = np.array(q_img)
        q_mask = support_masks[idx]
        q_mask = np.array(Image.open(q_mask).convert('L'))

        qry_sam_mask = support_sams[idx] # 得到预
        
        #prepare data
        num = qry_sam_mask.shape[0]
        sam_masks = {f'mask{i}': qry_sam_mask[i] for i in range(num)}
        sam_masks.update({f'mask{num}' : q_mask})
        additional_targets = {f'mask{i}': 'mask' for i in range(0, num + 1)}

        transform = A.Compose(self.q_transform.transforms, additional_targets=additional_targets)

        # 创建包含图片和掩码的输入字典
        input_data = {'image': q_img}
        input_data.update(sam_masks)

        # 应用变换
        transformed = transform(**input_data)
        transformed_masks = [transformed[key] for key in sam_masks.keys()] # masks



        # pair_transform = self.q_transform(image=q_img, mask=q_mask)
        # query_img = pair_transform['image']
        # query_mask = pair_transform['mask']
        qry_sam_masks = [torch.tensor(transformed_masks[i]) for i in range(0, num)]
        qry_sam_masks = torch.stack(qry_sam_masks, dim = 0) # 40 x H X W
        query_mask = transformed_masks[-1]
        query_img = transformed['image']

        q_img_transform = self.q_transform2(image=query_img)
        query_img = q_img_transform['image']
        query_img = Image.fromarray(query_img)
        query_img = self.transform(query_img)
        query_mask = self.process_mask(query_mask)
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        query_mask = query_mask.long()
        qry_sam_masks = F.interpolate(qry_sam_masks.unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks_tmp = []
        for smask in support_masks:
            smask = self.read_mask(smask)
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        support_sam_masks = [torch.tensor(support_sam) for support_sam in support_sams]
        support_sam_masks = torch.stack(support_sam_masks, dim = 0) # shot x 40 x H X W
        support_sam_masks = F.interpolate(support_sam_masks.float(), query_img.size()[-2:], mode='nearest')

        return support_imgs, support_masks, query_img, query_mask, class_sample, support_names, query_name, qry_sam_masks, support_sam_masks

    def load_frame(self, query_name, support_names):
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        query_id = query_name.split('/')[-1].split('.')[0] # 1
        query_name = os.path.join(os.path.dirname(query_name), query_id) + '.png' # ./dataset/FSS-1000/ab_wheel/1.png
        support_ids = [name.split('/')[-1].split('.')[0] for name in support_names]
        support_names = [os.path.join(os.path.dirname(name), sid) + '.png' for name, sid in zip(support_names, support_ids)]

        category_name = os.path.dirname(query_name)
        category_name = category_name.split('/')[-1]

        query_sam = self.get_data_and_mask(os.path.join(self.base_path, 'FSS-SAM-Mask', category_name,'train', 'sam_mask_vit_b_t50_p32_s50', query_id))
        support_sams =[self.get_data_and_mask(os.path.join(self.base_path, 'FSS-SAM-Mask',category_name, 'train', 'sam_mask_vit_b_t50_p32_s50'
                                                            ,supp_id))
                            for supp_id in support_ids]

        query_mask = self.read_mask(query_name)

        return query_img, query_mask, support_imgs, support_names, query_sam, support_sams

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask
    
    def process_mask(self, img):
        mask = torch.tensor(img)
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        x = len(self.img_metadata)
        query_name = self.img_metadata[idx] #
        class_sample = self.categories.index(query_name.split('/')[-2])
        if self.split == 'val':
            class_sample += 520
        elif self.split == 'test':
            class_sample += 760

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(range(1, 11), 1, replace=False)[0]
            support_name = os.path.join(os.path.dirname(query_name), str(support_name)) + '.jpg'
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_sample

    def build_class_ids(self):
        if self.split == 'trn':
            class_ids = range(0, 520)
        elif self.split == 'val':
            class_ids = range(520, 760)
        elif self.split == 'test':
            class_ids = range(760, 1000)
        return class_ids

    def build_img_metadata(self):
        img_metadata = []
        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, cat))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'jpg':
                    img_metadata.append(img_path)
        return img_metadata
