from functools import singledispatch
import numpy as np
import torch
from utils import *
from models import SSResNet
from PIL import Image
from copy import deepcopy
import time
import math


def test(model, data, target=None, batch_size = 2000):
    """
    output is hard label
    """

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    try:
        with torch.no_grad():
            output = model(data).cpu().data  # copy cuda tensor to host memory then convert to ndarray
    except:
        output = None
        for idx, batch_data in enumerate(get_one_batch(data, batch_size = batch_size)):
            with torch.no_grad():
                batch_output = model(batch_data[0]).cpu().data
                # batch_output = model(batch_data[0].cuda()).cpu().data
            if idx == 0:
                output = batch_output
            else:
                output = torch.cat((output, batch_output), dim=0)
            # processing_rate = (idx + 1)*batch_size/len(data)
            # if processing_rate < 1:
            #     print('\r','Predicting [{:.2%}]   '.format(processing_rate), end='', flush=True)
            # else:
            #     print('\r','Predicting [{:.2%}]   '.format(1), end='', flush=True)
    pred = torch.max(output, dim=1)[1].numpy()

    return pred


img_split_ratio = 2
patch_size = 13
pad_size = patch_size//2
to_tensor = True
config = {
    'input_shape': [1, 2, patch_size, patch_size],
    'n_classes': 4,
    'channels': 128,
    'blocks': 3,
    'is_bn': True,
    'is_dropout': False,
    'p': 0.2
}

root = "D:/Data/Resnet/Multi_folder"
dirs = os.listdir(root)

for idx, dir_name in enumerate(dirs):

    # model_path = os.path.join(root,dir_name,'Debug_model', 'ResNet_5000_50.pkl')
    model_path = os.path.join(root,dir_name,'Debug_LOO_model', 'ResNet_2500_50.pkl')
    hh_path = os.path.join(root,dir_name,'imagery_HH4_by_4average.tif')
    hv_path = os.path.join(root,dir_name,'imagery_HV4_by_4average.tif')
    labeld_img_path = os.path.join(root,dir_name,'all_train_mask.png')
    landmask_path = os.path.join(root,dir_name,'landmask.bmp')
    HH = cv2.imread(hh_path)[:,:,0]
    HV = cv2.imread(hv_path)[:,:,0]
    labeld_img = cv2.imread(labeld_img_path)[:,:,0]
    landmask = cv2.imread(landmask_path)[:,:,0]
    landmask[landmask == 255] = 1

    if HH.shape != HV.shape or HH.shape != labeld_img.shape or HH.shape != landmask.shape:
        print("Input images have different sizes!")
        exit()

    HH = HH/255
    HV = HV/255

    feature_map = np.zeros((2, HH.shape[0], HH.shape[1]),dtype=float)
    feature_map[0,:,:] = HH
    feature_map[1,:,:] = HV

    model  = SSResNet.ResNet(config)
    model.load_state_dict(torch.load(model_path))

    # process whole image by patches due to GUP RAM consumption
    split_size_0 = math.ceil(HH.shape[0]/img_split_ratio)
    split_size_1 = math.ceil(HH.shape[1]/img_split_ratio)
    split_size_0_pad_size = HH.shape[0] % img_split_ratio
    split_size_1_pad_size = HH.shape[1] % img_split_ratio


    feature_map_padded = np.pad(feature_map, ((0, 0), (pad_size, pad_size + split_size_0_pad_size), (pad_size, pad_size + split_size_1_pad_size)), 'constant')
    img_patch = np.zeros((img_split_ratio * img_split_ratio, 2, split_size_0 + 2*pad_size, split_size_1 + 2*pad_size), dtype = float)
    for x in range(img_split_ratio):
        for y in range(img_split_ratio):
            img_patch[x*img_split_ratio + y,:,:,:] = feature_map_padded[:, x*split_size_0:(x+1)*split_size_0+2*pad_size, y*split_size_1:(y+1)*split_size_1+2*pad_size]

    map = np.zeros((HH.shape[0],HH.shape[1]))

    # for i in range(len(img_patch)):
    for x in range(img_split_ratio):
        for y in range(img_split_ratio):
            i = x*img_split_ratio+y
            start = time.time()
            patch_data = read_img_as_patches(img_patch[i], patch_size, to_tensor=True)
            end = time.time()
            m, s = divmod(end - start, 60)
            h, m = divmod(m, 60)
            print('Loading Scene {:2}, patch {:2}. Runtime {:.0f}:{:.0f}:{:.0f}'.format(idx+1,i,h,m,s))
            start = time.time()
            pred_split = test(model, patch_data, batch_size=5000)
            end = time.time()
            m, s = divmod(end - start, 60)
            h, m = divmod(m, 60)
            print('Predicting Scene {:2}, patch {:2}. Runtime {:.0f}:{:.0f}:{:.0f}'.format(idx+1,i,h,m,s))
            map_split = pred_split.reshape(split_size_0+2*pad_size, split_size_1+2*pad_size)
            # map_split = map_split + 1
            # map_split = map_split*landmask
            # map_split_color = color_label(map_split, landmask=None)
            # map_split_color_img = Image.fromarray(map_split_color)
            # map_split_color_img.save(os.path.join(root,dir_name, 'ResNet_Debug_patch_{}.tif'.format(i)))
            if x == img_split_ratio - 1 and y != img_split_ratio - 1:
                map[x*split_size_0:(x+1)*split_size_0-split_size_0_pad_size, y*split_size_1:(y+1)*split_size_1] = \
                    map_split[pad_size:pad_size+split_size_0-split_size_0_pad_size,pad_size:pad_size+split_size_1]
            elif y == img_split_ratio - 1 and x != img_split_ratio - 1:
                map[x*split_size_0:(x+1)*split_size_0, y*split_size_1:(y+1)*split_size_1-split_size_1_pad_size] = \
                    map_split[pad_size:pad_size+split_size_0,pad_size:pad_size+split_size_1-split_size_1_pad_size]
            elif x == img_split_ratio - 1 and y == img_split_ratio - 1:
                map[x*split_size_0:(x+1)*split_size_0-split_size_0_pad_size, y*split_size_1:(y+1)*split_size_1-split_size_1_pad_size] = \
                    map_split[pad_size:pad_size+split_size_0-split_size_0_pad_size,pad_size:pad_size+split_size_1-split_size_1_pad_size]
            else:
                map[x*split_size_0:(x+1)*split_size_0, y*split_size_1:(y+1)*split_size_1] = \
                    map_split[pad_size:pad_size+split_size_0,pad_size:pad_size+split_size_1]

    # map = pred.reshape(HH.shape[0], HH.shape[1])
    map = map + 1
    map = map*landmask
    map_color = color_label(map, landmask=None)

    map_img = Image.fromarray(map)
    map_img.save(os.path.join(root,dir_name, 'ResNet_Debug_patch_{}.png'.format(patch_size)))
    map_color_img = Image.fromarray(map_color)
    map_color_img.save(os.path.join(root,dir_name, 'ResNet_Debug_patch_{}_color.png'.format(patch_size)))

print('Done!')