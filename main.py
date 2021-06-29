import numpy as np
from copy import deepcopy
from utils import *
from relabel import *
import cv2
import torch
from copy import deepcopy
from torch import nn, optim
from models import SSResNet
# from models import basic_cnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

############################################# set super-parameters ############################################

TRAIN_PROP = 1 #0.8
VAL_PROP = 0   #0.2
BATCH_SIZE = 5000   #5000
PATCH_SIZE = 13
EPOCH = 50
LR = 0.0001
TEST_INTERVAL = 1
NET_TYPE = 'ResNet'  # 'bpnet', 'basic_cnn', 'resnet', 'dip_resnet'
DATA_TYPE = 'patch'  # 'patch'(resnet, cnn), 'vector'(bp), 'full_image'(dip_resnet)


CONV_LAYERS = 3
FEATURE_NUMS = [32, 64, 64]
IS_BN = True  # set 'True' means using batch normalization
CONV_MODE = 'same' 

config = dict(conv_layers=CONV_LAYERS, feature_nums=FEATURE_NUMS, is_bn=IS_BN, conv_mode=CONV_MODE)#, act_fun=ACT_FUN, pad=PAD)


def train(model, train_data, train_target, save_dir = None):

    global LR, EPOCH, BATCH_SIZE, NET_TYPE, TEST_INTERVAL, \
        val_data, val_target, test_data, test_target
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()  # loss function: cross entropy
    optimizer = optim.Adam(model.parameters(), lr=LR)  # optimizer: adam
    loss_list = []
    # test_acc_list = []
    # train_acc_list =[]
    best_train = 0
    best_val = 0
    best_test = 0
    if save_dir == None:
        save_dir = './Debug_model'
    else:
        save_dir = os.path.join(save_dir,'Debug_model')
    state_dict = None
    best_state = None
    test_accuracy = None
    for epoch in range(EPOCH):
        model.train()
        # would it be better to shuffle data each epoch?
        for idx, samples in enumerate(get_one_batch(train_data, train_target, BATCH_SIZE)):
            data = samples[0]
            target = samples[1]
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_prop = (idx + 1)*BATCH_SIZE/len(train_data)
            # print('Training: Epoch: {0:5}, Batch: {1:3}, Batch rate: {2:.2%}'.format(epoch + 1, idx + 1, batch_prop))
            if batch_prop < 1:
                print('\r','Training {:.2%}   '.format(batch_prop), end='', flush=True)
            else:
                print('\r','Training {:.2%}   '.format(1), end='', flush=True)
    
            # if idx % TEST_INTERVAL == 0:
            #     if torch.cuda.is_available():
            #         train_accuracy = test(model, train_data.cuda(), train_target.cuda())[1]
            #         val_accuracy = test(model, val_data.cuda(), val_target.cuda())[1]
            #         test_accuracy = test(model, test_data.cuda(), test_target.cuda())[1]
            #     else:
            #         train_accuracy = test(model, train_data, train_target)[1]
            #         val_accuracy = test(model, val_data, val_target)[1]
            #         test_accuracy = test(model, test_data, test_target)[1]
            #     torch.cuda.empty_cache()
            #     print('Epoch: {0:5}, Batch: {1:3} | Loss: {2:13.8f} | Train: {3:.6f}　| Val: {4:.6f} | Test: {5:.6f}'.
            #           format(epoch + 1, idx + 1, loss.item(), train_accuracy, val_accuracy, test_accuracy),
            #           '\r', end='')
            #     if test_accuracy > best_test:
            #         best_train = train_accuracy
            #         best_val = val_accuracy
            #         best_test = test_accuracy
            #         best_state = [epoch + 1, idx + 1, loss, best_train, best_val, best_test]
            #         state_dict = deepcopy(model.state_dict())
            loss_list.append(loss.item())
            # train_acc_list.append(train_accuracy)
            # test_acc_list.append(test_accuracy)
        torch.cuda.empty_cache()
        model.eval()
        train_accuracy = test(model, train_data.cuda(), train_target.cuda())[1]
        # val_accuracy = test(model, val_data.cuda(), val_target.cuda())[1]
        torch.cuda.empty_cache()
        # test_accuracy = test(model, test_data.cuda(), test_target.cuda())[1]
        # torch.cuda.empty_cache()
        test_accuracy = train_accuracy
        val_accuracy = test_accuracy
        if test_accuracy > best_test:
            best_train = train_accuracy
            best_val = val_accuracy
            best_test = test_accuracy
            best_state = [epoch + 1, idx + 1, loss, best_train, best_val, best_test]
            state_dict = deepcopy(model.state_dict())
        print('Epoch: {0:4}, Batch: {1:3} | Loss: {2:13.8f} | Train: {3:.3%}　| Val: {4:.3%} | Test: {5:.3%}'.
            format(epoch + 1, idx + 1, loss.item(), train_accuracy, val_accuracy, test_accuracy))
        torch.cuda.empty_cache()

    # plot_curves(loss_list, train_acc_list, test_acc_list)
    model_name = NET_TYPE + '_' + str(BATCH_SIZE) + '_' + str(EPOCH) + '.pkl'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    model_dir = os.path.join(save_dir, model_name)
    torch.save(state_dict, model_dir)
    print('Best Results: ')
    print('Epoch: {:4}  Batch: {:3}  Loss: {:13.8f}  Train accuracy: {:.3%}  Val accuracy: {:.3%} Test accuracy: {:.3%}'.format(*best_state))
    with open(os.path.join(save_dir, 'train_log.txt'),'wt') as f:
        f.write('Best Results: ')
        f.write('Epoch: {:4}  Batch: {:3} | Loss: {:13.8f} | Train accuracy: {:.3%} | Val accuracy: {:.3%} | Test accuracy: {:.3%}'.format(*best_state))


def test(model, data, target=None, batch_size = 5000):
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
        for idx, batch_data in enumerate(get_one_batch(data, batch_size=batch_size)):

            with torch.no_grad():
                batch_output = model(batch_data[0]).cpu().data
                # batch_output = model(batch_data[0].cuda()).cpu().data
            if idx == 0:
                output = batch_output
            else:
                output = torch.cat((output, batch_output), dim=0)
    pred = torch.max(output, dim=1)[1].numpy()

    accuracy = None
    if target is not None:
        target = target.cpu()
        accuracy = compute_accuracy(pred, target)

    return pred, accuracy

def predict(model, feature_img, patch_size=13, target=None, batch_size = 2000):
    """
    output is hard label
    """
    # model.load_state_dict(torch.load(model_weight_path))
    data = read_img_as_patches(feature_img, patch_size, to_tensor=True)
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

    # accuracy = None
    # if target is not None:
    #     target = target.cpu()
    #     accuracy = compute_accuracy(pred, target)

    return pred


############################################# One Scene ############################################

# patch_size = 13
# to_tensor = True
# hh_path = "D:/Github/IceNet/data/20100524_034756_hh.tif"
# hv_path = "D:/Github/IceNet/data/20100524_034756_hv.tif"
# labeld_img_path = "D:/Github/IceNet/data/20100524_034756_label.png"
# HH = cv2.imread(hh_path)[:,:,0]
# HV = cv2.imread(hv_path)[:,:,0]
# labeld_img = cv2.imread(labeld_img_path)[:,:,0]
# if HH.shape != HV.shape or HH.shape != labeld_img.shape:
#     print("Input images have different sizes!")
#     exit()

# # Normalization
# HH = HH/255
# HV = HV/255

# feature_map = np.zeros((2, HH.shape[0], HH.shape[1]),dtype=float)
# feature_map[0,:,:] = HH
# feature_map[1,:,:] = HV

# train_mask, val_mask = load_masks(labeld_img,train_prop=TRAIN_PROP,val_prop=VAL_PROP, mask_dir ='D:\Github\IceNet')
# # train_mask, val_mask = get_masks_from_labeled_img(labeld_img,train_prop=TRAIN_PROP,val_prop=VAL_PROP, save_dir ='D:\Github\IceNet')



# train_data, train_target = get_patch_samples(feature_map, labeld_img, train_mask, patch_size=patch_size, to_tensor=to_tensor)
# val_data, val_target = get_patch_samples(feature_map, labeld_img, val_mask, patch_size=patch_size, to_tensor=to_tensor)
# # test_data, test_target = get_patch_samples(feature_map, labeld_img, test_mask, patch_size=patch_size, to_tensor=to_tensor)

# config = {
#     'input_shape': [1, 2, patch_size, patch_size],
#     'n_classes': 3,
#     'channels': 128,
#     'blocks': 3,
#     'is_bn': True,
#     'is_dropout': False,
#     'p': 0.2
# }
# model  = SSResNet.ResNet(config)

# train(model, train_data, train_target)

# print("Done!")

############################################# Leave-one-out ############################################
# patch_size = 13
# to_tensor = True
# config = {
#     'input_shape': [1, 2, patch_size, patch_size],
#     'n_classes': 4,
#     'channels': 128,
#     'blocks': 3,
#     'is_bn': True,
#     'is_dropout': False,
#     'p': 0.2
# }

# root = "D:/Data/Resnet/Multi_folder"
# # dirs = [x[0] for x in os.walk(root_path)][1:]
# dirs = os.listdir(root)

# LOO_train_data = None
# LOO_train_target = None

# for idx, dir_name in enumerate(dirs):
#     # relabel_train_val_from_sip(os.path.join(root,dir_name))
#     one_scene = dir_name
#     # deep copy?
#     # one_out = dirs
#     one_out = deepcopy(dirs)
#     del one_out[idx]
#     for i, one_out_name in enumerate(one_out):
#         hh_path = os.path.join(root,one_out_name,'imagery_HH4_by_4average.tif')
#         hv_path = os.path.join(root,one_out_name,'imagery_HV4_by_4average.tif')
#         labeld_img_path = os.path.join(root,one_out_name,'all_train_mask.png')
#         HH = cv2.imread(hh_path)[:,:,0]
#         HV = cv2.imread(hv_path)[:,:,0]
#         labeld_img = cv2.imread(labeld_img_path)[:,:,0]
#         if HH.shape != HV.shape or HH.shape != labeld_img.shape:
#             print("Input images have different sizes!")
#             exit()

#         # Normalization
#         HH = HH/255
#         HV = HV/255

#         feature_map = np.zeros((2, HH.shape[0], HH.shape[1]),dtype=float)
#         feature_map[0,:,:] = HH
#         feature_map[1,:,:] = HV

#         train_mask, _ = load_masks(labeld_img,train_prop=TRAIN_PROP,val_prop=VAL_PROP, mask_dir =os.path.join(root,one_out_name))
#         train_data, train_target = get_patch_samples(feature_map, labeld_img, train_mask, patch_size=patch_size, to_tensor=False)
#         # append or deep copy?
#         if i == 0:
#             LOO_train_data = train_data
#             LOO_train_target = train_target
#         else:
#             # LOO_train_data = torch.cat((LOO_train_data, train_data), dim=0)
#             # LOO_train_target = torch.cat((LOO_train_target, train_target), dim=0)
#             LOO_train_data = np.concatenate((LOO_train_data, train_data), axis=0)
#             LOO_train_target = np.concatenate((LOO_train_target, train_target), axis=0)
#             # Shuffle?
        
#     state = np.random.get_state()
#     np.random.shuffle(LOO_train_data)
#     np.random.set_state(state)
#     np.random.shuffle(LOO_train_target)

#     HH_test = cv2.imread(os.path.join(root,one_scene,'imagery_HH4_by_4average.tif'))[:,:,0]
#     HV_test = cv2.imread(os.path.join(root,one_scene,'imagery_HH4_by_4average.tif'))[:,:,0]
#     feature_map_test = np.zeros((2, HH_test.shape[0], HH_test.shape[1]), dtype=float)
#     feature_map_test[0,:,:] = HH_test/255
#     feature_map_test[1,:,:] = HV_test/255
#     labeld_img_test = cv2.imread(os.path.join(root,one_scene,'all_train_mask.png'))[:,:,0]
#     test_mask, _ = load_masks(labeld_img_test,train_prop=TRAIN_PROP,val_prop=VAL_PROP, mask_dir =os.path.join(root,one_scene))
#     test_data, test_target = get_patch_samples(feature_map_test, labeld_img_test, test_mask, patch_size=patch_size, to_tensor=to_tensor)
#     LOO_train_data = torch.from_numpy(LOO_train_data).float()
#     LOO_train_target = torch.from_numpy(LOO_train_target).long()
#     # delete model?
#     model  = SSResNet.ResNet(config)
#     train(model, LOO_train_data, LOO_train_target, save_dir = one_scene)



# print("Done")


############################################# Debug ############################################
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

LOO_train_data = None
LOO_train_target = None

for idx, dir_name in enumerate(dirs):

    hh_path = os.path.join(root,dir_name,'imagery_HH4_by_4average.tif')
    hv_path = os.path.join(root,dir_name,'imagery_HV4_by_4average.tif')
    labeld_img_path = os.path.join(root,dir_name,'all_train_mask.png')
    HH = cv2.imread(hh_path)[:,:,0]
    HV = cv2.imread(hv_path)[:,:,0]
    labeld_img = cv2.imread(labeld_img_path)[:,:,0]
    if HH.shape != HV.shape or HH.shape != labeld_img.shape:
        print("Input images have different sizes!")
        exit()

    HH = HH/255
    HV = HV/255

    feature_map = np.zeros((2, HH.shape[0], HH.shape[1]),dtype=float)
    feature_map[0,:,:] = HH
    feature_map[1,:,:] = HV

    train_mask, _ = load_masks(labeld_img,train_prop=TRAIN_PROP,val_prop=VAL_PROP, mask_dir =os.path.join(root,dir_name))
    train_data, train_target = get_patch_samples(feature_map, labeld_img, train_mask, patch_size=patch_size, to_tensor=False)
    # append or deep copy?
    if idx == 0:
        LOO_train_data = train_data
        LOO_train_target = train_target
    else:
        LOO_train_data = np.concatenate((LOO_train_data, train_data), axis=0)
        LOO_train_target = np.concatenate((LOO_train_target, train_target), axis=0)
        # Shuffle?

state = np.random.get_state()
np.random.shuffle(LOO_train_data)
np.random.set_state(state)
np.random.shuffle(LOO_train_target)

LOO_train_data = torch.from_numpy(LOO_train_data).float()
LOO_train_target = torch.from_numpy(LOO_train_target).long()

test_data = LOO_train_data
test_target = LOO_train_target

# delete model?
model  = SSResNet.ResNet(config)
train(model, LOO_train_data, LOO_train_target, save_dir = os.path.join(root,dir_name))

print("Done")