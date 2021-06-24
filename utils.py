import os
import sys
import numpy as np
import torch
import csv
import cv2
import scipy.io as sio
from PIL import Image

def color_label(grey_label, landmask = None):
    """
    Parameters
    ----------
    grey_label : grey-level labeled ice map
        0: background 
        1: yong ice
        2: first year ice
        3: multi-year ice
        4: open water
    landmask : 0: land

    Returns
    -------
    result : array of float, shape (M, N, 3)
        WMO standard RGS ice map
        Background (0, 0, 0)
        Yong ice (170, 40, 240)
        First year ice (255, 255, 0)
        Multi-year ice (255, 0, 0)
        Open water (150, 200, 255)
    """
    image = np.zeros(grey_label.shape + (3,), dtype=np.uint8)
    i, j = np.where(grey_label == 1)
    image[i, j, :] = (170, 40, 240)
    i, j = np.where(grey_label == 2)
    image[i, j, :] = (255, 255, 0)
    i, j = np.where(grey_label == 3)
    image[i, j, :] = (255, 0, 0)
    i, j = np.where(grey_label == 4)
    image[i, j, :] = (150, 200, 255)
    if landmask is not None:
        landmask = landmask[:, :, 0]
        i, j = np.where(landmask == 0)
        image[i, j, :] = (0, 0, 0)
    
    return image

def get_data_from_labeled_img(labeled_img):
    """
    get data and target from mask image
    """
    # target = np.array([])
    target = []
    x_list = []
    y_list = []
    class_list = np.unique(labeled_img)
    if class_list[0] == 0: #unknown: 0
        class_list = class_list[1:]
    n_class = class_list.size
    for i in range(n_class):
        x, y = np.where(labeled_img == class_list[i])
        target = np.append(target,np.ones(len(x))*class_list[i])
        x_list = np.append(x_list,x)
        y_list = np.append(y_list,y)
    # shuffle
    state = np.random.get_state()
    np.random.shuffle(x_list)
    np.random.set_state(state)
    np.random.shuffle(y_list)
    np.random.set_state(state)
    np.random.shuffle(target)
    
    # csv_headers = ['Label','Row (Starts from 0)','Col (Starts from 0)']
    # f = open('test.csv','w',encoding='utf-8',newline='')
    # csv_writer = csv.writer(f)
    # csv_writer.writerow(csv_headers)
    # csv_data = np.transpose(np.vstack((target,x_list,y_list)))
    # csv_writer.writerows(csv_data)
    # f.close()

    return target,x_list,y_list

def get_masks_from_labeled_img(labeled_img, train_prop = 1, val_prop = 0, save_dir=None):
    assert train_prop + val_prop <= 1, "train_prop + val_prop > 1"
    test = True
    if train_prop + val_prop == 1:
        test = False
    train_mask = np.zeros((labeled_img.shape[0], labeled_img.shape[1]))
    val_mask = train_mask.copy()
    test_mask = train_mask.copy()
    class_list = np.unique(labeled_img)
    if class_list[0] == 0: #unknown: 0
        class_list = class_list[1:]
    n_class = class_list.size
    for i in range(n_class):
        idx = np.argwhere(labeled_img == class_list[i])
        train_num = int(round(len(idx) * train_prop))
        val_num = int(round(len(idx) * val_prop))

        np.random.shuffle(idx)
        train_idx = idx[:train_num]
        val_idx = idx[train_num:train_num + val_num]
        # test_idx = idx[train_num + val_num:]

        train_mask[train_idx[:, 0], train_idx[:, 1]] = 1
        val_mask[val_idx[:, 0], val_idx[:, 1]] = 1
        # test_mask[test_idx[:, 0], test_idx[:, 1]] = 1

    if save_dir:
        # Or save masks as images
        folder_name = 'train_' + str(train_prop) + '_val_' + str(val_prop)
        save_dir = os.path.join(save_dir, folder_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        sio.savemat(os.path.join(save_dir, 'train_mask.mat'), {'train_mask': train_mask})
        sio.savemat(os.path.join(save_dir, 'val_mask.mat'), {'val_mask': val_mask})
        # sio.savemat(os.path.join(save_dir, 'test_mask.mat'), {'test_mask': test_mask})
        
        train_mask_img = color_label(train_mask*labeled_img)
        train_mask_img = Image.fromarray(train_mask_img)
        train_mask_img.save(os.path.join(save_dir, 'train_mask_img.png'))
        val_mask_img = color_label(val_mask*labeled_img)
        val_mask_img = Image.fromarray(val_mask_img)
        val_mask_img.save(os.path.join(save_dir, 'val_mask_img.png'))
        # test_mask_img = color_label(test_mask)
        # test_mask_img = Image.fromarray(test_mask_img)
        # test_mask_img.save(os.path.join(save_dir, 'test_mask_img.png'))

    return train_mask, val_mask

def load_masks(labeled_img, mask_dir, train_prop = 1, val_prop = 0):
    mask_fname = os.path.join(mask_dir, 'train_' + str(train_prop) + '_val_' + str(val_prop))
    if not (os.path.exists(mask_fname) and os.listdir(mask_fname)):
        # train_mask, val_mask, test_mask = get_masks(target, train_prop, val_prop, save_dir=mask_dir)
        train_mask, val_mask = get_masks_from_labeled_img(labeled_img, train_prop, val_prop, save_dir=mask_dir)

    else:
        train_mask = sio.loadmat(os.path.join(mask_fname, 'train_mask.mat'))['train_mask']
        val_mask = sio.loadmat(os.path.join(mask_fname, 'val_mask.mat'))['val_mask']
        # test_mask = sio.loadmat(os.path.join(mask_fname, 'test_mask.mat'))['test_mask']
    if train_mask == None or val_mask ==None:
        print('No mask images!')
        exit()

    return train_mask, val_mask

def get_patch_samples(data, target, mask, patch_size=13, to_tensor=True):
    if data.ndim != 3:
        print("Unsupported feature image size!")
        sys.exit()
    if data.shape[1]!=target.shape[0] or data.shape[2]!=target.shape[1]:
        print("data and target sizes do not match!")
        sys.exit()
    # padding data
    pad_size = patch_size // 2
    data = np.pad(data, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'constant')
    target = np.pad(target, ((pad_size, pad_size), (pad_size, pad_size)), 'constant')
    mask = np.pad(mask, ((pad_size, pad_size), (pad_size, pad_size)), 'constant')

    # # get patches
    # patch_target = target * mask
    # patch_target = patch_target[patch_target != 0] - 1
    # patch_data = np.zeros((patch_target.shape[0], data.shape[0], patch_size, patch_size))
    # index = np.argwhere(mask == 1)
    # for i, loc in enumerate(index):
    #     patch = data[:, loc[0] - pad_size:loc[0] + pad_size + 1, loc[1] - pad_size:loc[1] + pad_size + 1]
    #     patch_data[i, :, :, :] = patch

    # get patches
    index = np.argwhere(mask == 1)
    patch_target = np.zeros((index.shape[0]))
    patch_data = np.zeros((index.shape[0], data.shape[0], patch_size, patch_size))
    print("Number of samples is {}".format(index.shape[0]))
    
    for i, loc in enumerate(index):
        patch = data[:, loc[0] - pad_size:loc[0] + pad_size + 1, loc[1] - pad_size:loc[1] + pad_size + 1]
        patch_data[i, :, :, :] = patch
        patch_target[i] = target[loc[0],loc[1]]

    # shuffle
    state = np.random.get_state()
    np.random.shuffle(patch_data)
    np.random.set_state(state)
    np.random.shuffle(patch_target)

    # convert data format
    if to_tensor:
        patch_data = torch.from_numpy(patch_data).float()
        patch_target = torch.from_numpy(patch_target).long()
        # if torch.cuda.is_available():
        #     patch_data = patch_data.cuda()
        #     patch_target = patch_target.cuda()

    return patch_data, patch_target

def get_one_batch(train_data, train_target=None, batch_size=500):

    if train_target is None:
        train_target = torch.zeros(train_data.shape[0])
        train_target = torch.split(train_target, batch_size, dim=0)
    else:
        train_target = torch.split(train_target, batch_size, dim=0)

    train_data = torch.split(train_data, batch_size, dim=0)

    for i in range(len(train_data)):
        if torch.cuda.is_available():
            yield train_data[i].cuda(), train_target[i].cuda()
        else:
            yield train_data[i], train_target[i]

def compute_accuracy(pred, target):
    accuracy = float((pred == target.data.cpu().numpy()).astype(int).sum()) / \
               float(target.size(0))  # compute accuracy
    return accuracy

def read_img_as_patches(feature_img, patch_size, to_tensor=True):
    """
    feature_img is a 3D array -- first dimension contains different image bands 

    Read whole image as patches
    """
    # if feature_img.ndim == 2:
    #     patch_data = np.zeros((feature_img.shape[0] * feature_img.shape[1], patch_size, patch_size))
    #     feature_img = np.pad(feature_img, ((pad_size,pad_size), (pad_size,pad_size)), 'constant')
    # elif feature_img.ndim == 3:
    #     patch_data = np.zeros((feature_img.shape[1] * feature_img.shape[2], feature_img.shape[0], patch_size, patch_size))
    #     feature_img = np.pad(feature_img, ((0,0), (pad_size,pad_size), (pad_size,pad_size)), 'constant')
    # else:

    if feature_img.ndim != 3:
        print("Unsupported feature image size!")
        sys.exit()
    pad_size = patch_size//2
    mask = np.ones((feature_img.shape[1], feature_img.shape[2]))
    patch_data = np.zeros((feature_img.shape[1] * feature_img.shape[2], feature_img.shape[0], patch_size, patch_size))
    feature_img = np.pad(feature_img, ((0,0), (pad_size,pad_size), (pad_size,pad_size)), 'constant')
    mask = np.pad(mask, ((pad_size, pad_size), (pad_size, pad_size)), 'constant')
    index = np.argwhere(mask)
    for i, loc in enumerate(index):
        patch_data[i, :, :, :] = feature_img[:, loc[0] - pad_size:loc[0] + pad_size + 1, loc[1] - pad_size:loc[1] + pad_size + 1]
    
    if to_tensor:
        patch_data = torch.from_numpy(patch_data).float()
        if torch.cuda.is_available():
            patch_data = patch_data.cuda()

    return patch_data
    

    

