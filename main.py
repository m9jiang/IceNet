import numpy as np
from utils import *
import cv2
import torch
from torch import nn, optim


# def train(model, train_data, train_target):

#     global LR, EPOCH, BATCH_SIZE, NET_TYPE, TEST_INTERVAL, \
#         val_data, val_target, test_data, test_target
#     model.train()
#     if torch.cuda.is_available():
#         model = model.cuda()
#     criterion = nn.CrossEntropyLoss()  # loss function: cross entropy
#     optimizer = optim.Adam(model.parameters(), lr=LR)  # optimizer: adam
#     loss_list = []
#     test_acc_list = []
#     train_acc_list =[]
#     best_test = 0
#     save_dir = './model_save'
#     state_dict = None
#     best_state = None
#     test_accuracy = None
#     for epoch in range(EPOCH):

#         for idx, samples in enumerate(get_one_batch(train_data, train_target, BATCH_SIZE)):
#             data = samples[0]
#             target = samples[1]
#             output = model(data)
#             loss = criterion(output, target)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             print('Epoch: {0:5}, Batch: {1:3}'.
#                   format(epoch + 1, idx + 1),
#                   '\r', end='')

#             # if idx % TEST_INTERVAL == 0:
#             #     if torch.cuda.is_available():
#             #         train_accuracy = test(model, train_data.cuda(), train_target.cuda())[1]
#             #         val_accuracy = test(model, val_data.cuda(), val_target.cuda())[1]
#             #         test_accuracy = test(model, test_data.cuda(), test_target.cuda())[1]
#             #     else:
#             #         train_accuracy = test(model, train_data, train_target)[1]
#             #         val_accuracy = test(model, val_data, val_target)[1]
#             #         test_accuracy = test(model, test_data, test_target)[1]
#             #     torch.cuda.empty_cache()
#             #     print('Epoch: {0:5}, Batch: {1:3} | Loss: {2:13.8f} | Train: {3:.6f}　| Val: {4:.6f} | Test: {5:.6f}'.
#             #           format(epoch + 1, idx + 1, loss.item(), train_accuracy, val_accuracy, test_accuracy),
#             #           '\r', end='')
#             #     if test_accuracy > best_test:
#             #         best_train = train_accuracy
#             #         best_val = val_accuracy
#             #         best_test = test_accuracy
#             #         best_state = [epoch + 1, idx + 1, loss, best_train, best_val, best_test]
#             #         state_dict = deepcopy(model.state_dict())
#             loss_list.append(loss.item())
#             # train_acc_list.append(train_accuracy)
#             # test_acc_list.append(test_accuracy)

#     train_accuracy = test(model, train_data.cuda(), train_target.cuda())[1]
#     val_accuracy = test(model, val_data.cuda(), val_target.cuda())[1]
#     test_accuracy = test(model, test_data.cuda(), test_target.cuda())[1]
#     print('Epoch: {0:5}, Batch: {1:3} | Loss: {2:13.8f} | Train: {3:.6f}　| Val: {4:.6f} | Test: {5:.6f}'.
#           format(epoch + 1, idx + 1, loss.item(), train_accuracy, val_accuracy, test_accuracy),
#           '\r', end='')

#     # plot_curves(loss_list, train_acc_list, test_acc_list)
#     model_name = NET_TYPE + '_' + str(BATCH_SIZE) + '_' + str(EPOCH) + '.pkl'
#     model_dir = os.path.join(save_dir, model_name)
#     torch.save(state_dict, model_dir)
#     # print('Best Results: ')
#     # print('Epoch: {}  Batch: {}  Loss: {}  Train accuracy: {}  Val accuracy: {} Test accuracy: {}'.format(*best_state))

def test(model, data, target=None, batch_size = 2000):

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

hh_path = "D:/Github/IceNet/data/20100524_034756_hh.tif"
hv_path = "D:/Github/IceNet/data/20100524_034756_hv.tif"
labeld_img_path = ""
HH = cv2.imread(hh_path)
HV = cv2.imread(hv_path)
labeld_img = cv2.imread(labeld_img_path)
if HH.shape != HV.shape:
    print("Input images have different sizes!")
    exit()

# Normalization
HH = HH/255
HV = HV/255

feature_map = np.zeros((2, HH.shape[0], HH.shape[1]),dtype=float)
feature_map[0,:,:] = HH[:,:,0]
feature_map[1,:,:] = HV[:,:,0]

get_masks_from_labeled_img(labeld_img,train_prop=0.8,val_prop=0.2, save_dir ='D:\Github\IceNet')
print("111")