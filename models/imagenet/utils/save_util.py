import torch
import os
import shutil
from utils.misc import get_root_logger as logger
from pape.distributed import get_rank


class Saver:
    def __init__(self, arch, save_dir):
        self.checkpoint = None
        self.arch = arch
        self.save_dir = str(save_dir)
        if save_dir and get_rank() == 0 and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    def check_keys(self, own_keys, ckpt_keys):
        own_keys = set(own_keys)
        ckpt_keys = set(ckpt_keys)
        missing_keys = own_keys - ckpt_keys
        for missing_k in missing_keys:
            logger().info('caution: missing keys:{}'.format(missing_k))
        unexpected_keys = ckpt_keys - own_keys
        if len(unexpected_keys) > 0:
            logger().info('caution: {} unexpected keys. '.format(
                len(unexpected_keys)))
        shared_keys = own_keys & ckpt_keys
        return shared_keys

    def adapt_prefix(self, model_keys, state_dict, prefix):
        ret_dict = {}
        if list(model_keys)[0].startswith(prefix):
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    ret_dict[key] = value
                else:
                    ret_dict[prefix + key] = value
        else:
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    ret_dict[key.split(prefix, 1)[-1]] = value
                else:
                    ret_dict[key] = value
        return ret_dict

    def load_state(self, model, cfg, strict=False):
        if cfg.checkpoint is None:
            logger().info('=> no checkpoint found!')
            return
        else:
            if not os.path.isfile(cfg.checkpoint):
                logger().info('=> no checkpoint found at {}'.format(cfg.checkpoint))
                return
            checkpoint = torch.load(cfg.checkpoint, map_location='cpu')
            logger().info('=> load checkpoint from lustre......')
        checkpoint['best_acc1'] = torch.as_tensor(checkpoint['best_acc1'])
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model_keys = model.state_dict().keys()
        state_dict = self.adapt_prefix(model_keys, state_dict, 'model.')
        share_keys = self.check_keys(model_keys, state_dict.keys())
        logger().info('=> loading {} keys......'.format(len(share_keys)))
        model.load_state_dict(state_dict, strict)
        logger().info('=> loading checkpoint done!')
        self.checkpoint = checkpoint

    def load_optimizer(self, optimizer):
        best_acc1 = self.checkpoint['best_acc1'].cpu().cuda()
        start_epoch = self.checkpoint['epoch']
        optimizer.load_state_dict(self.checkpoint['optimizer'])
        logger().info('=> also load optimizer from checkpoint')
        return best_acc1, start_epoch

    def save_ckpt(self, save_writer, epoch, cfg_saver, model, optimizer,
                  file_name, best_acc1, cur_iter):
        epoch = epoch + 1
        state_dict = {
            'epoch': epoch,
            'arch': self.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict()
        }

        # if cfg_saver.save_pavi:
        #     save_writer.add_snapshot(file_name, state_dict, cur_iter)
        #     if cfg_saver.save_latest:
        #         save_writer.add_snapshot(self.arch + '_ckpt_latest.pth', state_dict,
        #                                  cur_iter)
        if cfg_saver.save_dir:
            file_name = os.path.join(self.save_dir, file_name)
            torch.save(state_dict, file_name)
            if cfg_saver.save_latest:
                shutil.copyfile(file_name,
                                os.path.join(self.save_dir, self.arch + '_ckpt_latest.pth'))
