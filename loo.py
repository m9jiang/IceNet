import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
from copy import deepcopy  # noqa: E402
import torch  # noqa: E402
from utils import (read_yaml, load_masks, get_patch_samples)  # noqa: E402
import cv2  # noqa: E402
from models import SSResNet  # noqa: E402
from main import train  # noqa: E402
import numpy as np  # noqa: E402
import argparse  # noqa: E402


def main(config_path: str,
         root: str
         ) -> None:
    '''Train on the whole dataset
    args:
        config (str): path to the config file (.yaml)
        root (str): root directory to the dataset
    '''
    dirs = os.listdir(root)
    config = read_yaml(config_path)
    train_prop = config['trainer']['train_prop']
    val_prop = config['trainer']['val_prop']
    patch_size = config['model']['input_shape'][2]
    LOO_train_data = None
    LOO_train_target = None

    for idx, dir_name in enumerate(dirs):
        one_scene = dir_name
        rest_scene = deepcopy(dirs)
        del rest_scene[idx]
        for i, scene_name in enumerate(rest_scene):
            hh_path = os.path.join(root, scene_name,
                                   'imagery_HH4_by_4average.tif')
            hv_path = os.path.join(root, scene_name,
                                   'imagery_HV4_by_4average.tif')
            labeld_img_path = os.path.join(root, scene_name,
                                           'all_train_mask.png')
            hh = cv2.imread(hh_path)[:, :, 0]
            hv = cv2.imread(hv_path)[:, :, 0]
            labeld_img = cv2.imread(labeld_img_path)[:, :, 0]
            if hh.shape != hv.shape or hh.shape != labeld_img.shape:
                raise ValueError('Input images have different sizes!')

            # Normalization
            hh = hh/255
            hv = hv/255

            feature_map = np.zeros((2, hh.shape[0], hh.shape[1]), dtype=float)
            feature_map[0, :, :] = hh
            feature_map[1, :, :] = hv

            train_mask, _ = load_masks(labeld_img,
                                       train_prop=train_prop,
                                       val_prop=val_prop,
                                       mask_dir=os.path.join(root,
                                                             scene_name))
            train_data, train_target = get_patch_samples(feature_map,
                                                         labeld_img,
                                                         train_mask,
                                                         patch_size=patch_size,
                                                         to_tensor=False)
            # append or deep copy
            if i == 0:
                LOO_train_data = train_data
                LOO_train_target = train_target
            else:
                LOO_train_data = np.concatenate((LOO_train_data, train_data),
                                                axis=0)
                LOO_train_target = np.concatenate((LOO_train_target,
                                                  train_target),
                                                  axis=0)

        state = np.random.get_state()
        np.random.shuffle(LOO_train_data)
        np.random.set_state(state)
        np.random.shuffle(LOO_train_target)

        HH_test = cv2.imread(os.path.join(root, one_scene,
                                          'imagery_HH4_by_4average.tif'))[:, :, 0]  # noqa: E501
        HV_test = cv2.imread(os.path.join(root, one_scene,
                                          'imagery_HV4_by_4average.tif'))[:, :, 0]  # noqa: E501
        feature_map_test = np.zeros((2, HH_test.shape[0], HH_test.shape[1]),
                                    dtype=float)
        feature_map_test[0, :, :] = HH_test/255
        feature_map_test[1, :, :] = HV_test/255
        labeld_img_test = cv2.imread(os.path.join(root, one_scene,
                                                  'all_train_mask.png'))[:, :, 0]  # noqa: E501
        test_mask, _ = load_masks(labeld_img_test,
                                  train_prop=train_prop,
                                  val_prop=val_prop,
                                  mask_dir=os.path.join(root, one_scene))
        test_data, test_target = get_patch_samples(feature_map_test,
                                                   labeld_img_test,
                                                   test_mask,
                                                   patch_size=patch_size,
                                                   to_tensor=True)
        LOO_train_data = torch.from_numpy(LOO_train_data).float()
        LOO_train_target = torch.from_numpy(LOO_train_target).long()
        model = SSResNet.ResNet(config['model'])
        train(model, LOO_train_data, LOO_train_target, test_data, test_target,
              save_dir=os.path.join(root, one_scene))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Train a featuer extractor '
                                                  '(encoder).'))
    parser.add_argument('-c', '--config', type=str, dest='config',
                        default='config/encoder.yaml',
                        help='Path to the config file (.yaml)')
    parser.add_argument('-d', '--dir', type=str, dest='dir',
                        default=('/home/major/data/21-scene/resnet/'
                                 'multi_folder-2022'),
                        help='Root directory of dataset')
    args = parser.parse_args()
    if torch.cuda.is_available():
        print(f'CUDA is available. Version: {torch.version.cuda}')
        print(f'GPU model is: {torch.cuda.get_device_name()}')
        print(f'GPU count is: {torch.cuda.device_count()}')
    else:
        print("CUDA is unavailable!")
    main(config_path=args.config, dir=args.dir)
