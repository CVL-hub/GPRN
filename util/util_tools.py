import os
import torch
import datetime
import logging
import random
import numpy as np
from operator import add
import torch.optim as optim
from functools import reduce
from alisuretool.Tools import Tools
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter


class MyCommon(object):

    @staticmethod
    def fix_randseed(seed):
        r""" Set random seeds for reproducibility """
        if seed is None:
            seed = int(random.random() * 1e5)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def mean(x):
        return sum(x) / len(x) if len(x) > 0 else 0.0

    @staticmethod
    def to_cuda(batch, device):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        return batch

    @staticmethod
    def to_cpu(tensor):
        return tensor.detach().clone().cpu()

    @staticmethod
    def gpu_setup(use_gpu, gpu_id):
        if torch.cuda.is_available() and use_gpu:
            Tools.print()
            Tools.print('Cuda available with GPU: {}'.format(torch.cuda.get_device_name(0)))
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            device = torch.device("cuda:{}".format(gpu_id))
        else:
            Tools.print()
            Tools.print('Cuda not available')
            device = torch.device("cpu")
        return device

    @staticmethod
    def mask_to_rle_pytorch(tensor):
        # Put in fortran order and flatten h,w
        b, h, w = tensor.shape
        tensor = tensor.permute(0, 2, 1).flatten(1)

        # Compute change indices
        diff = tensor[:, 1:] ^ tensor[:, :-1]
        change_indices = diff.nonzero()

        # Encode run length
        out = []
        for i in range(b):
            cur_idxs = change_indices[change_indices[:, 0] == i, 1]
            cur_idxs = torch.cat(
                [
                    torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                    cur_idxs + 1,
                    torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
                ]
            )
            btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
            counts = [] if tensor[i, 0] == 0 else [0]
            counts.extend(btw_idxs.detach().cpu().tolist())
            out.append({"size": [h, w], "counts": counts})
        return out

    @staticmethod
    def rle_to_mask(rle):
        """Compute a binary mask from an uncompressed RLE."""
        h, w = rle["size"]
        mask = np.empty(h * w, dtype=bool)
        idx = 0
        parity = False
        for count in rle["counts"]:
            mask[idx: idx + count] = parity
            idx += count
            parity ^= True
        mask = mask.reshape(w, h)
        return mask.transpose()  # Put in C order
    
    @staticmethod
    def rle_to_mask_pytorch(rle):
         """Compute a binary mask from an uncompressed RLE on the GPU."""
         h, w = rle["size"]
        
         # Create a tensor for the mask
         mask = torch.empty(h * w, dtype=torch.bool, device='cuda')

         idx = 0
         parity = False
         counts = torch.tensor(rle["counts"], device='cuda')

         for count in counts:
             mask[idx: idx + count] = parity
             idx += count
             parity = not parity

         mask = mask.reshape(h, w)
    
         return mask.T.cpu().numpy()  # Transpose to get the correct shape


class MyOptim(object):

    @classmethod
    def get_finetune_optimizer(cls, args, model):
        if args.finetune_backbone:
            return cls.get_finetune_optimizer_backbone(args, model)
        else:
            return cls.get_finetune_optimizer_no_backbone(args, model)
        pass

    @classmethod
    def adjust_learning_rate_poly(cls, args, optimizer, iter):
        if args.finetune_backbone:
            return cls.adjust_learning_rate_poly_backbone(args, optimizer, iter)
        else:
            return cls.adjust_learning_rate_poly_no_backbone(args, optimizer, iter)
        pass

    @staticmethod
    def get_finetune_optimizer_no_backbone(args, model):
        lr = args.lr
        weight_list = []
        bias_list = []
        pretrain_weight_list = []
        pretrain_bias_list =[]
        for name, value in model.named_parameters():
            if 'model_res' in name or 'model_backbone' in name or 'backbone' in name:
                if 'weight' in name:
                    pretrain_weight_list.append(value)
                elif 'bias' in name:
                    pretrain_bias_list.append(value)
            else:
                if 'weight' in name:
                    weight_list.append(value)
                elif 'bias' in name:
                    bias_list.append(value)

        opt = optim.SGD([{'params': weight_list, 'lr': lr * 1},
                         {'params': bias_list, 'lr': lr * 2}],
                        momentum=0.90, weight_decay=0.0005)
        return opt

    @staticmethod
    def adjust_learning_rate_poly_no_backbone(args, optimizer, iter):
        base_lr = args.lr
        max_iter = args.max_steps
        reduce = ((1-float(iter)/max_iter)**(args.power))
        lr = base_lr * reduce
        optimizer.param_groups[0]['lr'] = lr * 1
        optimizer.param_groups[1]['lr'] = lr * 2
        return lr

    @staticmethod
    def get_finetune_optimizer_backbone(args, model):
        lr = args.lr
        weight_list = []
        bias_list = []
        pretrain_weight_list = []
        pretrain_bias_list =[]
        for name, value in model.named_parameters():
            if 'model_res' in name or 'model_backbone' in name or 'backbone' in name:
                if 'weight' in name:
                    pretrain_weight_list.append(value)
                elif 'bias' in name:
                    pretrain_bias_list.append(value)
            else:
                if 'weight' in name:
                    weight_list.append(value)
                elif 'bias' in name:
                    bias_list.append(value)

        opt = optim.SGD([{'params': pretrain_weight_list, 'lr':lr/1000},
                          {'params': pretrain_bias_list, 'lr':lr/500},
                          {'params': weight_list, 'lr':lr*1},
                          {'params': bias_list, 'lr':lr*2}],
                        momentum=0.90, weight_decay=0.0005)
        return opt

    @staticmethod
    def adjust_learning_rate_poly_backbone(args, optimizer, iter):
        base_lr = args.lr
        max_iter = args.max_steps
        reduce = ((1-float(iter)/max_iter)**(args.power))
        lr = base_lr * reduce
        optimizer.param_groups[0]['lr'] = lr / 1000
        optimizer.param_groups[1]['lr'] = lr / 500
        optimizer.param_groups[2]['lr'] = lr * 1
        optimizer.param_groups[3]['lr'] = lr * 2
        return lr

    pass


class AverageMeter:
    r""" Stores loss, evaluation results """
    def __init__(self, dataset, device=None):
        self.benchmark = dataset.benchmark
        self.class_ids_interest = dataset.class_ids
        self.class_ids_interest = torch.tensor(self.class_ids_interest).to(device)

        if self.benchmark == 'isaid':
            self.nclass = 15
        elif self.benchmark == 'dlrsd':
            self.nclass = 15
        elif self.benchmark == 'pascal':
            self.nclass = 20
        elif self.benchmark == 'coco':
            self.nclass = 80
        elif self.benchmark == 'fss':
            self.nclass = 1000

        self.intersection_buf = torch.zeros([2, self.nclass]).float().to(device)
        self.union_buf = torch.zeros([2, self.nclass]).float().to(device)
        self.ones = torch.ones_like(self.union_buf)
        self.loss_buf = []

    def update(self, inter_b, union_b, class_id, loss):
        self.intersection_buf.index_add_(1, class_id, inter_b.float())
        self.union_buf.index_add_(1, class_id, union_b.float())
        if loss is None:
            loss = torch.tensor(0.0)
        self.loss_buf.append(loss)

    def compute_iou(self):
        iou = self.intersection_buf.float() / \
              torch.max(torch.stack([self.union_buf, self.ones]), dim=0)[0]
        iou = iou.index_select(1, self.class_ids_interest)
        miou = iou[1].mean() * 100

        fb_iou = (self.intersection_buf.index_select(1, self.class_ids_interest).sum(dim=1) /
                  self.union_buf.index_select(1, self.class_ids_interest).sum(dim=1)).mean() * 100

        return miou, fb_iou

    def compute_iou_class(self):
        iou = self.intersection_buf.float() / \
              torch.max(torch.stack([self.union_buf, self.ones]), dim=0)[0]
        iou = iou.index_select(1, self.class_ids_interest)
        miou = iou[1].mean() * 100

        fb_iou = (self.intersection_buf.index_select(1, self.class_ids_interest).sum(dim=1) /
                  self.union_buf.index_select(1, self.class_ids_interest).sum(dim=1)).mean() * 100

        return miou, fb_iou, iou[1]

    def write_result(self, split, epoch):
        iou, fb_iou = self.compute_iou()

        loss_buf = torch.stack(self.loss_buf)
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch
        msg += 'Avg L: %6.5f  ' % loss_buf.mean()
        msg += 'mIoU: %5.2f   ' % iou
        msg += 'FB-IoU: %5.2f   ' % fb_iou

        msg += '***\n'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch, write_batch_idx=20):
        if batch_idx % write_batch_idx == 0:
            msg = '[Epoch: %02d] ' % epoch if epoch != -1 else ''
            msg += '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)
            iou, fb_iou = self.compute_iou()
            if epoch != -1:
                loss_buf = torch.stack(self.loss_buf)
                msg += 'L: %6.5f  ' % loss_buf[-1]
                msg += 'Avg L: %6.5f  ' % loss_buf.mean()
            msg += 'mIoU: %5.2f  |  ' % iou
            msg += 'FB-IoU: %5.2f' % fb_iou
            Logger.info(msg)
            pass
        pass
    pass


class Logger:
    r""" Writes evaluation results of training/testing """
    @classmethod
    def initialize(cls, args, training):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logpath = args.logpath if training else '_TEST_' + args.load.split('/')[-2].split('.')[0] + logtime
        if logpath == '': logpath = logtime

        cls.logpath = os.path.join('logs', logpath + '.log')
        cls.benchmark = args.benchmark
        if not os.path.exists(cls.logpath):
            os.makedirs(cls.logpath)

        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log arguments
        logging.info('\n:=========== Few-shot Seg. with HSNet ===========')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
        logging.info(':================================================\n')

    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        logging.info(msg)

    @classmethod
    def save_model_miou(cls, model, epoch, val_miou):
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'best_model.pt'))
        cls.info('Model saved @%d w/ val. mIoU: %5.2f.\n' % (epoch, val_miou))

    @classmethod
    def log_params(cls, model):
        backbone_param = 0
        learner_param = 0
        for k in model.state_dict().keys():
            n_param = model.state_dict()[k].view(-1).size(0)
            if k.split('.')[0] in 'backbone':
                if k.split('.')[1] in ['classifier', 'fc']:  # as fc layers are not used in HSNet
                    continue
                backbone_param += n_param
            else:
                learner_param += n_param
        Logger.info('Backbone # param.: %d' % backbone_param)
        Logger.info('Learnable # param.: %d' % learner_param)
        Logger.info('Total # param.: %d' % (backbone_param + learner_param))

    pass
