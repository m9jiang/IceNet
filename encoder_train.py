import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchvision import models  # noqa: E402
from models import SSResNet  # noqa: E402
from main import train  # noqa: E402
from utils import load_masks, get_patch_samples, read_yaml  # noqa: E402
import argparse  # noqa: E402


def main(config_path: str,
         root: str
         ) -> None:
    '''Train on the whole dataset
    args:
        config (str): path to the config file (.yaml)
        root (str): root directory to the dataset
    '''
    config = read_yaml(config_path)
    patch_size = config['model']['input_shape'][2]
    dirs = os.listdir(root)
    if 'results' in dirs:
        dirs.remove('results')
    all_train_data = None
    all_train_target = None
    all_val_data = None
    all_val_target = None

    for idx, dir_name in enumerate(dirs):

        hh_path = os.path.join(root, dir_name, 'imagery_HH4_by_4average.tif')
        hv_path = os.path.join(root, dir_name, 'imagery_HV4_by_4average.tif')
        labeld_img_path = os.path.join(root, dir_name, 'all_train_mask.png')
        hh = cv2.imread(hh_path)[:, :, 0]
        hv = cv2.imread(hv_path)[:, :, 0]
        labeld_img = cv2.imread(labeld_img_path)[:, :, 0]
        if hh.shape != hv.shape or hh.shape != labeld_img.shape:
            raise ValueError('Input images have different sizes!')

        hh = hh/255
        hv = hv/255

        feature_map = np.zeros((2, hh.shape[0], hh.shape[1]), dtype=float)
        feature_map[0, :, :] = hh
        feature_map[1, :, :] = hv

        (train_mask,
         val_mask) = load_masks(labeld_img,
                                train_prop=config['trainer']['train_prop'],
                                val_prop=config['trainer']['val_prop'],
                                mask_dir=os.path.join(root, dir_name))
        (train_data,
         train_target) = get_patch_samples(feature_map,
                                           labeld_img,
                                           train_mask,
                                           patch_size=patch_size,
                                           to_tensor=False)
        (val_data,
         val_target) = get_patch_samples(feature_map,
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
            all_train_data = np.concatenate((all_train_data, train_data),
                                            axis=0)
            all_train_target = np.concatenate((all_train_target, train_target),
                                              axis=0)
            all_val_data = np.concatenate((all_val_data, val_data),
                                          axis=0)
            all_val_target = np.concatenate((all_val_target, val_target),
                                            axis=0)
            # Shuffle?

    state = np.random.get_state()
    np.random.shuffle(all_train_data)
    np.random.set_state(state)
    np.random.shuffle(all_train_target)

    all_train_data = torch.from_numpy(all_train_data).float()
    all_train_target = torch.from_numpy(all_train_target).long()

    all_val_data = torch.from_numpy(all_val_data).float()
    all_val_target = torch.from_numpy(all_val_target).long()

    # TODO: add model to config
    model = SSResNet.ResNet(config['model'])
    # model = models.resnet50(num_classes=config['model']['n_classes'])
    # model.name = 'ResNet50'
    # model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    train(model=model,
          patch_size=patch_size,
          train_data=all_train_data,
          train_target=all_train_target,
          val_data=all_val_data,
          val_target=all_val_target,
          # batch_size=16384,
          batch_size=4000,
          gpu_idx=config['trainer']['gpu_idx'],
          n_epoch=300,
          save_dir=root)

    print("Done")


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
    main(config_path=args.config, root=args.dir)
