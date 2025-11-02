import argparse
import time

import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests
from mmengine.analysis import get_model_complexity_info
from mmengine.analysis.print_helper import _format_size
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from models.network_fusion2 import MaskedSwinFusion as net
from utils import utils_option as option
from torch.nn import functional as F

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default=r"model.pth")
    parser.add_argument('--folder_ir', type=str, default=r"LLVIP\ir")
    parser.add_argument('--folder_vi', type=str, default=r"LLVIP\vi")
    parser.add_argument('--save_dir', type=str, default=r"outputs\LLVIP",
                        help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None,
                        help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')

    parser.add_argument('--opt', type=str, default=r"options\fusion.json", help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=False)
    global opt_net
    opt_net = opt['netG']

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # set up model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    else:
        print(f'loading error')

    model = define_model(args)
    model.eval()
    model = model.to(device)
    folder, window_size = args.folder_ir, 8

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    paths=os.listdir(folder)
    for idx, name in enumerate(paths):
        # read image
        img_ir = cv2.imread(os.path.join(args.folder_ir,name), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        img_vi = cv2.imread(os.path.join(args.folder_vi,name), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        print(name)
        img_ir = np.expand_dims(img_ir, axis=-1)
        img_vi = np.expand_dims(img_vi, axis=-1)

        img_ir = np.transpose(img_ir if img_ir.shape[2] == 1 else img_ir[:, :, [2, 1, 0]], (2, 0, 1))
        img_ir = torch.from_numpy(img_ir).float().unsqueeze(0).to(device)

        img_vi = np.transpose(img_vi if img_vi.shape[2] == 1 else img_vi[:, :, [2, 1, 0]], (2, 0, 1))
        img_vi = torch.from_numpy(img_vi).float().unsqueeze(0).to(device)

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_ir.size()
            mod_pad_h, mod_pad_w = 0, 0
            _, _, h, w = img_ir.size()
            if h % window_size != 0:
                mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                mod_pad_w = window_size - w % window_size
            img_ir = F.pad(img_ir, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            img_vi = F.pad(img_vi, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            output = model(img_ir, img_vi)
            output = output[:, :, :h_old, :w_old]
        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        cv2.imwrite(f'{save_dir}/{name}', output)



def get_bare_model(net):
    """Get bare model, especially under wrapping with
    DistributedDataParallel or DataParallel.
    """
    if isinstance(net, (DataParallel, DistributedDataParallel)):
        net = net.module
    return net


def define_model(args):
    global opt_net
    model = net(upscale=opt_net['upscale'],
                in_chans=opt_net['in_chans'],
                img_size=opt_net['img_size'],
                window_size=opt_net['window_size'],
                img_range=opt_net['img_range'],
                depths=opt_net['depths'],
                embed_dim=opt_net['embed_dim'],
                num_heads=opt_net['num_heads'],
                mlp_ratio=opt_net['mlp_ratio'],
                upsampler=opt_net['upsampler'],
                talking_heads=opt_net['talking_heads'],
                use_attn_fn=opt_net['attn_fn'],
                head_scale=opt_net['head_scale'],
                on_attn=opt_net['on_attn'],
                use_mask=opt_net['use_mask'],
                mask_ratio1=opt_net['mask_ratio1'],
                mask_ratio2=opt_net['mask_ratio2'],
                mask_is_diff=opt_net['mask_is_diff'],
                type=opt_net['type'],
                opt=opt_net,
                nd_deconder=False
                )

    param_key_g = 'params'
    print(args.model_path)
    pretrained_model = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    # print(pretrained_model.keys())
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
                          strict=True)

    return model


def test(img_ir, img_vi, model, args, window_size):
    output = model(img_ir, img_vi)


    return output


if __name__ == '__main__':
    main()

