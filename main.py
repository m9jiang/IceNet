import numpy as np
from copy import deepcopy
from utils import *
import cv2
import torch
from torch import nn, optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

############################################# set super-parameters ############################################

TRAIN_PROP = 0.8
VAL_PROP = 0.2
BATCH_SIZE = 5000   #5000
PATCH_SIZE = 13
EPOCH = 50
LR = 0.0001
TEST_INTERVAL = 1
NET_TYPE = 'basic_cnn'  # 'bpnet', 'basic_cnn', 'resnet', 'dip_resnet'
DATA_TYPE = 'patch'  # 'patch'(resnet, cnn), 'vector'(bp), 'full_image'(dip_resnet)


CONV_LAYERS = 3
FEATURE_NUMS = [32, 64, 64]
IS_BN = True  # set 'True' means using batch normalization
CONV_MODE = 'same' 

config = dict(conv_layers=CONV_LAYERS, feature_nums=FEATURE_NUMS, is_bn=IS_BN, conv_mode=CONV_MODE)#, act_fun=ACT_FUN, pad=PAD)


def train(model, train_data, train_target):

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
    save_dir = './model_save'
    state_dict = None
    best_state = None
    test_accuracy = None
    for epoch in range(EPOCH):
        model.train()
        for idx, samples in enumerate(get_one_batch(train_data, train_target, BATCH_SIZE)):
            data = samples[0]
            target = samples[1]
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_prop = (idx + 1)/len(train_data)
            print('Training: Epoch: {0:5}, Batch: {1:3}, Batch rate: {2:.2%}'.format(epoch + 1, idx + 1, batch_prop))

    
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
        model.eval()
        train_accuracy = test(model, train_data.cuda(), train_target.cuda())[1]
        val_accuracy = test(model, val_data.cuda(), val_target.cuda())[1]
        # test_accuracy = test(model, test_data.cuda(), test_target.cuda())[1]
        test_accuracy = val_accuracy
        if test_accuracy > best_test:
            best_train = train_accuracy
            best_val = val_accuracy
            best_test = test_accuracy
            best_state = [epoch + 1, idx + 1, loss, best_train, best_val, best_test]
            state_dict = deepcopy(model.state_dict())
        print('Epoch: {0:5}, Batch: {1:3} | Loss: {2:13.8f} | Train: {3:.3%}　| Val: {4:.3%} | Test: {5:.3%}'.
            format(epoch + 1, idx + 1, loss.item(), train_accuracy, val_accuracy, test_accuracy))

    # plot_curves(loss_list, train_acc_list, test_acc_list)
    model_name = NET_TYPE + '_' + str(BATCH_SIZE) + '_' + str(EPOCH) + '.pkl'
    model_dir = os.path.join(save_dir, model_name)
    torch.save(state_dict, model_dir)
    print('Best Results: ')
    print('Epoch: {}  Batch: {}  Loss: {}  Train accuracy: {}  Val accuracy: {} Test accuracy: {}'.format(*best_state))

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


patch_size = 13
to_tensor = True
hh_path = "D:/Github/IceNet/data/20100524_034756_hh.tif"
hv_path = "D:/Github/IceNet/data/20100524_034756_hv.tif"
labeld_img_path = "D:/Github/IceNet/data/20100524_034756_label.png"
HH = cv2.imread(hh_path)[:,:,0]
HV = cv2.imread(hv_path)[:,:,0]
labeld_img = cv2.imread(labeld_img_path)[:,:,0]
if HH.shape != HV.shape or HH.shape != labeld_img.shape:
    print("Input images have different sizes!")
    exit()

# Normalization
HH = HH/255
HV = HV/255

feature_map = np.zeros((2, HH.shape[0], HH.shape[1]),dtype=float)
feature_map[0,:,:] = HH
feature_map[1,:,:] = HV

train_mask, val_mask = load_masks(labeld_img,train_prop=TRAIN_PROP,val_prop=VAL_PROP, save_dir ='D:\Github\IceNet')
# train_mask, val_mask = get_masks_from_labeled_img(labeld_img,train_prop=TRAIN_PROP,val_prop=VAL_PROP, save_dir ='D:\Github\IceNet')



train_data, train_target = get_patch_samples(feature_map, labeld_img, train_mask, patch_size=patch_size, to_tensor=to_tensor)
val_data, val_target = get_patch_samples(feature_map, labeld_img, val_mask, patch_size=patch_size, to_tensor=to_tensor)
# test_data, test_target = get_patch_samples(feature_map, labeld_img, test_mask, patch_size=patch_size, to_tensor=to_tensor)

print("111")