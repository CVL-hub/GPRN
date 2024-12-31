from model.GPRN import GPRN_Net
from util.utils import count_params, set_seed, mIOU

import argparse
import os
import torch
from torch.nn import DataParallel
from tqdm import tqdm
import glob
from SAM2pred import SAM_pred
from data.dataset import FSSDataset
import numpy as np
import cv2
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
from torchvision.utils import save_image
def parse_args():
    parser = argparse.ArgumentParser(description='IFA for CD-FSS')
    # basic arguments
    parser.add_argument('--data-root',
                        type=str,
                        default="./dataset",
                        required=True,
                        help='root path of training dataset')
    parser.add_argument('--dataset',
                        type=str,
                        default='deepglobe',
                        choices=['fss', 'deepglobe', 'isic', 'lung'],
                        help='training dataset')
    parser.add_argument('--batch-size',
                        type=int,
                        default=6,
                        help='batch size of training')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='learning rate')
    parser.add_argument('--crop-size',
                        type=int,
                        default=473,
                        help='cropping size of training samples')
    parser.add_argument('--backbone',
                        type=str,
                        choices=['resnet50', 'resnet101'],
                        default='resnet50',
                        help='backbone of semantic segmentation model')
    parser.add_argument('--refine', dest='refine', action='store_true', default=True)
    parser.add_argument('--shot',
                        type=int,
                        default=1,
                        help='number of support pairs')
    parser.add_argument('--episode',
                        type=int,
                        default=24000,
                        help='total episodes of training')
    parser.add_argument('--snapshot',
                        type=int,
                        default=1200,
                        help='save the model after each snapshot episodes')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='random seed to generate tesing samples')
    parser.add_argument("--sam_type",
                        type=str,
                        default= "vit_b")
    
    parser.add_argument("--alpha",
                        type=float,
                        default= 1.0)
    

    parser.add_argument("--weight",
                    type=float,
                    default= 0.1)
    
    parser.add_argument("--ckpt",
                        type=str,
                        default= "../pretrained_model/SAM/sam_vit_b_01ec64.pth")
    
    parser.add_argument("--positive_point",
                        type=int,
                        default=20)
    
    parser.add_argument("--negative_point",
                        type=int,
                        default=20)
    

    parser.add_argument("--point",
                        type=int,
                        default=20)
    

    parser.add_argument("--keep_num",
                        type=int,
                        default=10)
    
    parser.add_argument("--fuse_method",
                        type=str,
                        default="entropy", choices=['entropy', 'coff', 'sum', 'union', 'xor'])
    

    parser.add_argument('--vis', dest='vis', action='store_true', default=False)
    parser.add_argument('--second_refine', dest='second_refine', action='store_true', default=False)
    parser.add_argument('--post_refine', dest='post_refine', action='store_true', default=False)

    args = parser.parse_args()
    return args


def evaluate(model, SAM, dataloader, args):
    tbar = tqdm(dataloader)

    if args.dataset == 'fss':
        num_classes = 1000
    elif args.dataset == 'deepglobe':
        num_classes = 6
    elif args.dataset == 'isic':
        num_classes = 3
    elif args.dataset == 'lung':
        num_classes = 1

    metric = mIOU(num_classes)

    for i, (img_s_list, mask_s_list, img_q, mask_q, cls, _, id_q, qry_sam_masks, support_sam_masks) in enumerate(tbar):

        img_s_list = img_s_list.permute(1,0,2,3,4)
        mask_s_list = mask_s_list.permute(1,0,2,3)
        support_sam_masks = support_sam_masks.permute(1,0,2,3,4)  
            
        img_s_list = img_s_list.numpy().tolist()
        mask_s_list = mask_s_list.numpy().tolist()
        support_sam_masks = support_sam_masks.numpy().tolist()

        img_q, mask_q, qry_sam_masks = img_q.cuda(), mask_q.cuda(), qry_sam_masks.cuda()

        for k in range(len(img_s_list)):
            img_s_list[k], mask_s_list[k], support_sam_masks[k] = torch.Tensor(img_s_list[k]), torch.Tensor(mask_s_list[k]), torch.Tensor(support_sam_masks[k])
            img_s_list[k], mask_s_list[k], support_sam_masks[k] = img_s_list[k].cuda(), mask_s_list[k].cuda(), support_sam_masks[k].cuda()

        cls = cls[0].item()
        cls = cls + 1

        with torch.no_grad():
            out_ls = model(img_s_list, mask_s_list, img_q, mask_q, qry_sam_masks, support_sam_masks)
            pred = torch.argmax(out_ls[0], dim = 1)
            if args.post_refine:
                pred, _ = SAM(query_img = img_q, prediction = out_ls[0], origin_pred = out_ls[2])
                pred = torch.argmax(pred, dim = 1)

        pred[pred == 1] = cls
        mask_q[mask_q == 1] = cls

        metric.add_batch(pred.cpu().numpy(), mask_q.cpu().numpy())

        tbar.set_description("Testing mIOU: %.2f" % (metric.evaluate() * 100.0))

    return metric.evaluate() * 100.0

def main():
    args = parse_args()
    print('\n' + str(args))

    FSSDataset.initialize(img_size=400, datapath=args.data_root)
    testdataset,testloader = FSSDataset.build_dataloader(args.dataset, args.batch_size, 4, '0', 'val', args.shot)

    model = GPRN_Net(args.backbone, args.refine, args.shot, args)

    ### Please modify the following paths with your model path if needed.
    if args.dataset == 'deepglobe':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = ''
            if args.shot == 5:
                checkpoint_path = ''
    if args.dataset == 'isic':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = ''
            if args.shot == 5:
                checkpoint_path = ''
    if args.dataset == 'lung':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = ''
            if args.shot == 5:
                checkpoint_path = ''
    if args.dataset == 'fss':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = ''
            if args.shot == 5:
                checkpoint_path = ''



    print('Evaluating model:', checkpoint_path)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    SAM = SAM_pred(args)

    print('\nParams: %.1fM' % count_params(model))
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    #best_model = DataParallel(model).cuda()
    best_model = model.cuda()
    SAM = SAM.cuda()
    SAM = SAM.eval()
    print('\nParams: %.1fM' % count_params(model))

    # for param in model.encoder.parameters():
    #     param.requires_grad = False

    print('\nEvaluating on 5 seeds.....')
    total_miou = 0.0
    model.eval()
    for seed in range(5):
        print('\nRun %i:' % (seed + 1))
        set_seed(args.seed + seed)

        miou = evaluate(best_model, SAM, testloader, args)
        total_miou += miou

    print('\n' + '*' * 32)
    print('Averaged mIOU on 5 seeds: %.2f' % (total_miou / 5))
    print('*' * 32 + '\n')



if __name__ == '__main__':
    main()



