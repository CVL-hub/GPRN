from model.GPRN import GPRN_Net
from util.utils import count_params, set_seed, mIOU

import argparse
from copy import deepcopy
import os
import sys
import time
import torch
from torch.nn import CrossEntropyLoss, DataParallel, NLLLoss
from SAM2pred import SAM_pred
import torch.nn.functional as F
from torch.optim import SGD
from tqdm import tqdm
from data.dataset import FSSDataset
import cv2
from torchvision.utils import save_image
import pandas as pd
import numpy as np
torch.autograd.set_detect_anomaly(True)
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description='IFA for CD-FSS')
    # basic arguments
    parser.add_argument('--data-root',
                        type=str,
                        default="./dataset",
                        # required=True,
                        help='root path of training dataset')
    parser.add_argument('--dataset',
                        type=str,
                        default='lung',
                        choices=['fss', 'deepglobe', 'isic', 'lung'],
                        help='training dataset')
    parser.add_argument('--batch-size',
                        type=int,
                        default=4,
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
    
    parser.add_argument('--train_seed',
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
                        default= 1.0)
    
    parser.add_argument("--ckpt",
                        type=str,
                        default= "../pretrained_model/SAM/sam_vit_b_01ec64.pth")
    
    parser.add_argument("--positive_point",
                        type=int,
                        default=10)
    
    parser.add_argument("--negative_point",
                        type=int,
                        default=30)
    

    parser.add_argument("--point",
                        type=int,
                        default=20)
    

    parser.add_argument("--keep_num",
                        type=int,
                        default=10)
    
    parser.add_argument("--fuse_method",
                        type=str,
                        default="entropy", choices=['entropy', 'coff', 'sum', 'union', 'xor'])
    

    parser.add_argument('--vis', dest='vis', action='store_true', default=True)
    parser.add_argument('--second_refine', dest='second_refine', action='store_true', default=False)
    parser.add_argument('--post_refine', dest='post_refine', action='store_true', default=True)
    parser.add_argument('--train', dest='train', action='store_true', default=True)
   
    
    
    

    args = parser.parse_args()
    return args

def evaluate(model,SAM,dataloader, args):
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

    for i, (img_s_list, mask_s_list, img_q, mask_q, cls, id_s, id_q, qry_sam_masks, support_sam_masks) in enumerate(tbar):

        img_s_list = img_s_list.permute(1,0,2,3,4)
        mask_s_list = mask_s_list.permute(1,0,2,3)
        support_sam_masks = support_sam_masks.permute(1,0,2,3,4)  
            
        img_s_list = img_s_list.numpy().tolist()
        mask_s_list = mask_s_list.numpy().tolist()
        support_sam_masks = support_sam_masks.numpy().tolist()

        img_q, mask_q, qry_sam_masks= img_q.cuda(), mask_q.cuda(), qry_sam_masks.cuda()
       

        for k in range(len(img_s_list)):
            img_s_list[k], mask_s_list[k],support_sam_masks[k] = torch.Tensor(img_s_list[k]), torch.Tensor(mask_s_list[k]), torch.Tensor(support_sam_masks[k])
            img_s_list[k], mask_s_list[k],support_sam_masks[k]= img_s_list[k].cuda(), mask_s_list[k].cuda(), support_sam_masks[k].cuda()
        cls = cls[0].item()
        cls = cls + 1


        with torch.no_grad():
            
            out_ls= model(img_s_list, mask_s_list, img_q, mask_q, qry_sam_masks, support_sam_masks)
            pred = torch.argmax(out_ls[0], dim = 1)
            if args.post_refine:
                pred, _= SAM(query_img = img_q, prediction = out_ls[0], origin_pred = out_ls[2])
                pred = torch.argmax(pred, dim = 1)

        pred[pred == 1] = cls
        mask_q[mask_q == 1] = cls

        metric.add_batch(pred.cpu().numpy(), mask_q.cpu().numpy())

        tbar.set_description("Testing mIOU: %.2f" % (metric.evaluate() * 100.0))

    return metric.evaluate() * 100.0

def main():
    path_dir = 'ifa'

    args = parse_args()
    print('\n' + str(args))

    ### Please modify the following paths with your trained model paths.
    if args.dataset == 'deepglobe':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = 'outdir/models/deepglobe/resnet50_1shot_47.29.pth'
            if args.shot == 5:
                checkpoint_path = 'outdir/models/deepglobe/resnet50_1shot_47.29.pth'
    if args.dataset == 'isic':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = 'outdir/models/isic/resnet50_1shot_53.42.pth'
            if args.shot == 5:
                checkpoint_path = 'outdir/models/isic/resnet50_1shot_53.42.pth'
    if args.dataset == 'lung':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = './outdir/models/lung/resnet50_1shot_79.09.pth'
            if args.shot == 5:
                checkpoint_path = './outdir/models/lung/resnet50_1shot_79.09.pth'
    if args.dataset == 'fss':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = ''
            if args.shot == 5:
                checkpoint_path = ''
    
    miou = 0
    save_path = 'outdir/models/%s/%s' % (args.dataset, path_dir)
    os.makedirs(save_path, exist_ok=True)
 
    FSSDataset.initialize(img_size=400, datapath=args.data_root)
    train_dataset = args.dataset+'ifa'
    traindataset,trainloader = FSSDataset.build_dataloader(train_dataset, args.batch_size, 4, '0', 'val', args.shot)
    FSSDataset.initialize(img_size=400, datapath=args.data_root)
    testdataset,testloader = FSSDataset.build_dataloader(args.dataset, args.batch_size, 4, '0', 'val', args.shot)

    print('Do we use SSP refinement?', args.refine)
    model = GPRN_Net(args.backbone, args.refine, args.shot, args)
    print('\nParams: %.1fM' % count_params(model))

    print('Loaded model:', checkpoint_path)

    if args.train:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
    

    SAM = SAM_pred(args)

    # checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint)

    for param in model.layer0.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False

   
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = False
    

    criterion = CrossEntropyLoss(ignore_index=255)
    criterion_nll = NLLLoss()
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
    optimizer = SGD([param for param in model.parameters() if param.requires_grad],
                    lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    SAM = SAM.cuda()
    model = DataParallel(model).cuda()
    best_model = None

    iters = 0
    total_iters = args.episode // args.batch_size
    lr_decay_iters = [total_iters // 3, total_iters * 2 // 3]

    previous_best = float(miou)

    exclude_file_list = set()

    # each snapshot is considered as an epoch
    t = args.episode // args.snapshot if args.dataset != 'fss' else 1
    for epoch in range(t):
        
        print("\n==> Epoch %i, learning rate = %.5f\t\t\t\t Previous best = %.2f"
              % (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()

        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

        total_loss = 0.0

        tbar = tqdm(trainloader)
        # int(time.time())
        set_seed(args.train_seed + epoch)

        for i, (img_s_list, mask_s_list, img_q, mask_q, _, id_s,id_q, qry_sam_masks, support_sam_masks) in enumerate(tbar):

            img_s_list = img_s_list.permute(1,0,2,3,4)
            mask_s_list = mask_s_list.permute(1,0,2,3)    
            support_sam_masks = support_sam_masks.permute(1,0,2,3,4)  
            img_s_list = img_s_list.numpy().tolist()
            mask_s_list = mask_s_list.numpy().tolist()
            support_sam_masks = support_sam_masks.numpy().tolist()
            img_q, mask_q, qry_sam_masks = img_q.cuda(), mask_q.cuda(), qry_sam_masks.cuda()


        
            for k in range(len(img_s_list)):
                img_s_list[k], mask_s_list[k], support_sam_masks[k]= torch.Tensor(img_s_list[k]), torch.Tensor(mask_s_list[k]), torch.Tensor(support_sam_masks[k])
                img_s_list[k], mask_s_list[k], support_sam_masks[k] = img_s_list[k].cuda(), mask_s_list[k].cuda(), support_sam_masks[k].cuda()

            out_ls= model(img_s_list, mask_s_list, img_q, mask_q, qry_sam_masks, support_sam_masks)
           
            mask_s = torch.cat(mask_s_list, dim=0)
            mask_s = mask_s.long()
        
            if args.refine:
                loss = criterion(out_ls[0], mask_q) + criterion(out_ls[1], mask_q) + criterion(out_ls[2], mask_q) + criterion(out_ls[3], mask_s) * 0.2
                    
            else:
                loss = criterion(out_ls[0], mask_q) + criterion(out_ls[1], mask_q) + criterion(out_ls[2], mask_s) * 0.4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.item()

            iters += 1
            if iters in lr_decay_iters:
                optimizer.param_groups[0]['lr'] /= 10.0

            tbar.set_description('Loss: %.3f' % (total_loss / (i + 1)))

       
        model.eval()
        set_seed(args.seed)
        testdataset.set_exclude_file(exclude_file_list)
        miou = evaluate(model,SAM,testloader, args)
        torch.cuda.empty_cache()
        #
        #best_model = deepcopy(model)

        if miou >= previous_best:
            best_model = deepcopy(model)
            previous_best = miou
            torch.save(best_model.module.state_dict(),
                os.path.join(save_path, '%s_%ishot_%.2f.pth' % (args.backbone, args.shot, miou)))
            
    print('\nEvaluating on 5 seeds.....')
    total_miou = 0.0
    for seed in range(5):
        print('\nRun %i:' % (seed + 1))
        set_seed(args.seed + seed)

        miou = evaluate(best_model,SAM,testloader, args)
        total_miou += miou

    print('\n' + '*' * 32)
    print('Averaged mIOU on 5 seeds: %.2f' % (total_miou / 5))
    print('*' * 32 + '\n')

    torch.save(best_model.module.state_dict(),
               os.path.join(save_path, '%s_%ishot_avg_%.2f.pth' % (args.backbone, args.shot, total_miou / 5)))


if __name__ == '__main__':
    main()
