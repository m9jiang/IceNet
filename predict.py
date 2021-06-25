import numpy as np
import torch
from utils import *
from models import SSResNet
from PIL import Image

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
        for idx, batch_data in enumerate(get_one_batch(data, batch_size)):

            with torch.no_grad():
                batch_output = model(batch_data[0]).cpu().data
                # batch_output = model(batch_data[0].cuda()).cpu().data
            if idx == 0:
                output = batch_output
            else:
                output = torch.cat((output, batch_output), dim=0)
    pred = torch.max(output, dim=1)[1].numpy()

    return pred

model_path =""
hh_path = "D:/Github/IceNet/data/20100524_034756_hh.tif"
hv_path = "D:/Github/IceNet/data/20100524_034756_hv.tif"
labeld_img_path = "D:/Github/IceNet/data/20100524_034756_label.png"
landmask_path = ""
HH = cv2.imread(hh_path)[:,:,0]
HV = cv2.imread(hv_path)[:,:,0]
labeld_img = cv2.imread(labeld_img_path)[:,:,0]
landmask = cv2.imread(landmask_path)[:,:,0]
landmask[landmask == 255] = 1

if HH.shape != HV.shape or HH.shape != labeld_img.shape or HH.shape != landmask.shape:
    print("Input images have different sizes!")
    exit()

# Normalization
HH = HH/255
HV = HV/255

feature_map = np.zeros((2, HH.shape[0], HH.shape[1]),dtype=float)
feature_map[0,:,:] = HH
feature_map[1,:,:] = HV

patch_size = 13
config = {
    'input_shape': [1, 2, patch_size, patch_size],
    'n_classes': 3,
    'channels': 128,
    'blocks': 3,
    'is_bn': True,
    'is_dropout': False,
    'p': 0.2
}
model  = SSResNet.ResNet(config)
model_dict = model.load_state_dict(torch.load(model_path))

patch_data = read_img_as_patches(feature_map, patch_size, to_tensor=True)

pred = test(model_dict, patch_data, batch_size=5000)

map = pred.reshape(HH.shape[0], HH.shape[1])
map = map + 1
map = map*landmask
map_color = color_label(map, landmask=None)


map_img = Image.fromarray(map)
map_img.save(os.path.join(save_dir, 'val_mask_img.png'))
map_color_img = Image.fromarray(map_color)
map_color_img.save(os.path.join(save_dir, 'val_mask_img.png'))
