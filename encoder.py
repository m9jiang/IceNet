import os
import cv2
import numpy as np
import torch
from models import SSResNet
from main import train
from utils import load_masks, get_patch_samples

patch_size = 13
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

all_train_data = None
all_train_target = None
all_val_data = None
all_val_target = None

for idx, dir_name in enumerate(dirs):

    hh_path = os.path.join(root, dir_name, 'imagery_HH4_by_4average.tif')
    hv_path = os.path.join(root, dir_name, 'imagery_HV4_by_4average.tif')
    labeld_img_path = os.path.join(root, dir_name, 'all_train_mask.png')
    HH = cv2.imread(hh_path)[:, :, 0]
    HV = cv2.imread(hv_path)[:, :, 0]
    labeld_img = cv2.imread(labeld_img_path)[:, :, 0]
    if HH.shape != HV.shape or HH.shape != labeld_img.shape:
        print("Input images have different sizes!")
        exit()

    HH = HH/255
    HV = HV/255

    feature_map = np.zeros((2, HH.shape[0], HH.shape[1]),dtype=float)
    feature_map[0, :, :] = HH
    feature_map[1, :, :] = HV

    train_mask, val_mask = load_masks(labeld_img,
                                      train_prop=TRAIN_PROP,
                                      val_prop=VAL_PROP,
                                      mask_dir=os.path.join(root, dir_name))
    train_data, train_target = get_patch_samples(feature_map,
                                                 labeld_img,
                                                 train_mask,
                                                 patch_size=patch_size,
                                                 to_tensor=False)
    val_data, val_target = get_patch_samples(feature_map,
                                             labeld_img,
                                             val_mask,
                                             patch_size=patch_size,
                                             to_tensor=False)
    # append or deep copy?
    if idx == 0:
        all_train_data = train_data
        all_train_target = train_target
        all_val_data = val_data
        all_val_target = val_target
    else:
        all_train_data = np.concatenate((all_train_data, train_data), axis=0)
        all_train_target = np.concatenate((all_train_target, train_target),
                                          axis=0)
        all_val_data = np.concatenate((all_val_data, val_data), axis=0)
        all_val_target = np.concatenate((all_val_target, val_target), axis=0)
        # Shuffle?

state = np.random.get_state()
np.random.shuffle(all_train_data)
np.random.set_state(state)
np.random.shuffle(all_train_target)

all_train_data = torch.from_numpy(all_train_data).float()
all_train_target = torch.from_numpy(all_train_target).long()

all_val_data = torch.from_numpy(all_val_data).float()
all_val_target = torch.from_numpy(all_val_target).long()

# test_data = all_train_data
# test_target = all_train_target

# delete model?
model = SSResNet.ResNet(config)
train(model, all_train_data, all_train_target, all_val_data,
      all_val_target, save_dir=os.path.join(root, dirs[0]))

print("Done")