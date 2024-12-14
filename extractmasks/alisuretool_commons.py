import os
import sys
import torch
import pickle
import random
import platform
import numpy as np
from alisuretool.Tools import Tools


class Commons(object):

    @staticmethod
    def get_att(obj):
        att = {}
        for item in dir(obj):
            if "__" not in item:
                att[item] = getattr(obj, item)
            pass
        return att

    @staticmethod
    def setup_gpu(use_gpu, gpu_id):
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
    def setup_seed(seed):
        if seed > 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance
        pass

    @staticmethod
    def to_device(obj, one, non_blocking=False):
        device = obj.device if hasattr(obj, "device") else obj
        return one.to(device, non_blocking=non_blocking)

    @staticmethod
    def change_lr(optimizer, epoch, lr_list):
        for now_epoch, lr in lr_list:
            if now_epoch == epoch:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    pass
                return lr
            pass
        return optimizer.param_groups[0]['lr']

    @staticmethod
    def view_model_param(model):
        total_param = 0
        for param in model.parameters():
            total_param += np.prod(list(param.data.size()))
        return total_param

    @staticmethod
    def save_checkpoint(model, root_ckpt_dir, epoch):
        cp_file_name = os.path.join(root_ckpt_dir, 'epoch_{}.pkl'.format(epoch))
        torch.save(model.state_dict(), cp_file_name)
        Tools.print(f"Save checkpoint to {cp_file_name}")
        return cp_file_name

    @staticmethod
    def load_checkpoint(model, checkpoint):
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location=None)
                pass
            model.load_state_dict(state_dict, strict=True)
            Tools.print("load weights from {}".format(checkpoint))
        else:
            Tools.print("load weights error due to no checkpoint")
        pass

    pass
