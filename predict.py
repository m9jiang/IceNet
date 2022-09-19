import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
import numpy as np  # noqa: E402
import cv2  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torchvision import models  # noqa: E402
from utils import (read_yaml, get_one_batch, read_img_as_patches,  # noqa: E402
                   color_label)  # noqa: E402
from models import SSResNet  # noqa: E402
from PIL import Image  # noqa: E402
from copy import deepcopy  # noqa: E402
import time  # noqa: E402
import math  # noqa: E402
import argparse  # noqa: E402
import gc  # noqa: E402


def release_cuda(var: torch.Tensor) -> None:
    var.cpu()
    del var
    gc.collect()
    torch.cuda.empty_cache()

    return


def test(model, data, target=None, batch_size=8192):
    """
    output is hard label
    """
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    try:
        with torch.no_grad():
            output = model(data).cpu().data
    except RuntimeError:
        output = None
        for idx, batch_data in enumerate(get_one_batch(data, batch_size=batch_size)):
            with torch.no_grad():
                batch_output = model(batch_data[0]).cpu().data
                # batch_output = model(batch_data[0].cuda()).cpu().data
            if idx == 0:
                output = batch_output
            else:
                output = torch.cat((output, batch_output), dim=0)
    pred = torch.max(output, dim=1)[1].numpy()
    release_cuda(output)
    # release_cuda(data)

    return pred


def main(root: str, config_path: str, model_path: str) -> None:
    config = read_yaml(config_path)
    patch_size = config['model']['input_shape'][2]
    pad_size = patch_size//2
    img_split_ratio = config['img_split_ratio']
    dirs = os.listdir(root)
    dirs.sort()
    if 'results' in dirs:
        dirs.remove('results')

    for idx, dir_name in enumerate(dirs):
        hh_path = os.path.join(root, dir_name, 'imagery_HH4_by_4average.tif')
        hv_path = os.path.join(root, dir_name, 'imagery_HV4_by_4average.tif')
        landmask_path = os.path.join(root, dir_name, 'landmask.bmp')
        hh = cv2.imread(hh_path)[:, :, 0]
        hv = cv2.imread(hv_path)[:, :, 0]
        landmask = cv2.imread(landmask_path)[:, :, 0]
        landmask[landmask == 255] = 1
        if hh.shape != hv.shape or hh.shape != landmask.shape:
            raise ValueError('Input images have different sizes!')

        hh = hh/255
        hv = hv/255

        feature_map = np.zeros((2, hh.shape[0], hh.shape[1]), dtype=float)
        feature_map[0, :, :] = hh
        feature_map[1, :, :] = hv
        model = SSResNet.ResNet(config['model'])
        # model = models.resnet50(num_classes=config['model']['n_classes'])
        # model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = nn.DataParallel(model, device_ids=[0, 1, 2])
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # process whole image by patches due to GUP RAM consumption
        split_size_0 = math.ceil(hh.shape[0]/img_split_ratio)
        split_size_1 = math.ceil(hh.shape[1]/img_split_ratio)
        split_size_0_pad_size = hh.shape[0] % img_split_ratio
        split_size_1_pad_size = hh.shape[1] % img_split_ratio

        feature_map_padded = np.pad(feature_map,
                                    ((0, 0),
                                     (pad_size,
                                      pad_size + split_size_0_pad_size),
                                     (pad_size,
                                      pad_size + split_size_1_pad_size)),
                                    'constant')
        img_patch = np.zeros((img_split_ratio * img_split_ratio,
                              2, split_size_0 + 2*pad_size,
                              split_size_1 + 2*pad_size),
                             dtype=float)
        for x in range(img_split_ratio):
            for y in range(img_split_ratio):
                img_patch[x*img_split_ratio + y, :, :, :] = feature_map_padded[:, x*split_size_0:(x+1)*split_size_0+2*pad_size, y*split_size_1:(y+1)*split_size_1+2*pad_size]

        map = np.zeros((hh.shape[0], hh.shape[1]))

        for x in range(img_split_ratio):
            for y in range(img_split_ratio):
                i = x*img_split_ratio + y
                start = time.time()
                patch_data = read_img_as_patches(img_patch[i], patch_size)
                patch_data.detach()
                end = time.time()
                m, s = divmod(end - start, 60)
                h, m = divmod(m, 60)
                print(f'Loading Scene {idx:2}, patch {i:2}. Runtime {h:.0f}:{m:.0f}:{s:.0f}.')
                start = time.time()
                pred_split = test(model, patch_data, batch_size=3000)
                patch_data = patch_data.cpu()
                del patch_data
                gc.collect()
                torch.cuda.empty_cache()
                end = time.time()
                m, s = divmod(end - start, 60)
                h, m = divmod(m, 60)
                print(f'Predicting Scene {idx:2}, patch {i:2}. Runtime {h:.0f}:{m:.0f}:{s:.0f}.')
                map_split = pred_split.reshape(split_size_0+2*pad_size, split_size_1+2*pad_size)
                # map_split = map_split + 1
                # map_split = map_split*landmask
                # map_split_color = color_label(map_split, landmask=None)
                # map_split_color_img = Image.fromarray(map_split_color)
                # map_split_color_img.save(os.path.join(root,dir_name, 'ResNet_Debug_patch_{}.tif'.format(i)))
                if x == img_split_ratio - 1 and y != img_split_ratio - 1:
                    map[x*split_size_0:(x+1)*split_size_0-split_size_0_pad_size, y*split_size_1:(y+1)*split_size_1] = \
                        map_split[pad_size:pad_size+split_size_0-split_size_0_pad_size, pad_size:pad_size+split_size_1]
                elif y == img_split_ratio - 1 and x != img_split_ratio - 1:
                    map[x*split_size_0:(x+1)*split_size_0, y*split_size_1:(y+1)*split_size_1-split_size_1_pad_size] = \
                        map_split[pad_size:pad_size+split_size_0, pad_size:pad_size+split_size_1-split_size_1_pad_size]
                elif x == img_split_ratio - 1 and y == img_split_ratio - 1:
                    map[x*split_size_0:(x+1)*split_size_0-split_size_0_pad_size, y*split_size_1:(y+1)*split_size_1-split_size_1_pad_size] = \
                        map_split[pad_size:pad_size+split_size_0-split_size_0_pad_size, pad_size:pad_size+split_size_1-split_size_1_pad_size]
                else:
                    map[x*split_size_0:(x+1)*split_size_0, y*split_size_1:(y+1)*split_size_1] = \
                        map_split[pad_size:pad_size+split_size_0, pad_size:pad_size+split_size_1]

        # map = pred.reshape(HH.shape[0], HH.shape[1])
        map = map + 1
        map = map*landmask
        map_color = color_label(map, landmask=None)

        map_img = Image.fromarray(map).convert('RGB')
        map_img.save(os.path.join(root, dir_name,
                                  f'ResNet_patch_{patch_size}.png'))
        map_color_img = Image.fromarray(map_color)
        map_color_img.save(os.path.join(root, dir_name,
                           f'ResNet_patch_{patch_size}_color.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Predict sea-ice maps.'))
    parser.add_argument('-c', '--config', type=str, dest='config',
                        default='config/predict.yaml',
                        help='Path to the config file (.yaml)')
    parser.add_argument('-d', '--dir', type=str, dest='dir',
                        default=('/home/major/data/21-scene/resnet/'
                                 'multi_folder-2022'),
                        help='Root directory of dataset')
    parser.add_argument('-m', '--model', type=str, dest='model',
                        default=('/home/major/data/21-scene/resnet/'
                                 'multi_folder-2022/results/'
                                 'debug_encoder_resnet_kernel_3_patch_25_2022-07-20-23-25/'
                                 'ResNet_batch_4000_epoch_300.pkl'),
                        help='directory of model')
    args = parser.parse_args()
    if torch.cuda.is_available():
        print(f'CUDA is available. Version: {torch.version.cuda}')
        print(f'GPU model is: {torch.cuda.get_device_name()}')
        print(f'GPU count is: {torch.cuda.device_count()}')
    else:
        print("CUDA is unavailable!")
    main(root=args.dir, config_path=args.config, model_path=args.model)
