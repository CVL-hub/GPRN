import csv
import os
from shutil import copy

ImgPath = '/mnt/sda/shifengpeng/data/ISIC2018_Task1-2_Training_Input'
AnnPath = '/mnt/sda/shifengpeng/data/ISIC2018_Task1_Training_GroundTruth'


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

with open('/home/shifengpeng/IFA-master/data/isic/class_id.csv', encoding='utf-8-sig') as f:
    for row in csv.reader(f, skipinitialspace=True):
        if row[0] != 'ID':
            imgpath = os.path.join(ImgPath, row[0]) + '.jpg'
            annpath = os.path.join(AnnPath, row[0]) + '_segmentation.png'
            
            if row[1] == 'nevus':
                img_target_dir = os.path.join(ImgPath, '1')
                ann_target_dir = os.path.join(AnnPath, '1')
            elif row[1] == 'seborrheic_keratosis':
                img_target_dir = os.path.join(ImgPath, '2')
                ann_target_dir = os.path.join(AnnPath, '2')
            elif row[1] == 'melanoma':
                img_target_dir = os.path.join(ImgPath, '3')
                ann_target_dir = os.path.join(AnnPath, '3')
            else:
                continue  # Skip rows with unexpected class labels

            # Ensure target directories exist
            ensure_dir(img_target_dir)
            ensure_dir(ann_target_dir)

            # Copy files to the target directories
            copy(imgpath, os.path.join(img_target_dir, os.path.basename(imgpath)))
            copy(annpath, os.path.join(ann_target_dir, os.path.basename(annpath)))