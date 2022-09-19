from copy import deepcopy
import time
import datetime
import torch
from torch import nn, optim
import pandas as pd
import os
from utils import (get_one_batch, plot_curves, compute_accuracy,
                   read_img_as_patches)

# TODO: Add Tensorboard and Logger
# TODO: Add type hints
# 'bpnet', 'basic_cnn', 'resnet', 'dip_resnet'
# 'patch'(resnet, cnn), 'vector'(bp), 'full_image'(dip_resnet)
# DATA_TYPE = 'patch'


def train(model, patch_size,
          train_data, train_target,
          val_data=None, val_target=None,
          test_data=None, test_target=None,
          batch_size=6000, n_epoch=500, lr=1e-4,
          gpu_idx=None, save_dir=None):

    start = time.time()
    # TODO: designate gpu for training
    model = nn.DataParallel(model, device_ids=[0, 1, 2])
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()  # loss function: cross entropy
    optimizer = optim.Adam(model.parameters(), lr=lr)  # optimizer: adam
    loss_list = []
    test_loss_list = []
    test_acc_list = []
    train_acc_list = []
    best_train = 0
    best_val = 0
    best_test = 0
    dt = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    if save_dir is None:
        save_dir = './Debug_model'
    else:
        save_dir = os.path.join(
            save_dir, 'results',
            f'debug_encoder_resnet_kernel_3_patch_{patch_size}_{dt}')
    model_name = (f'{model.module.name}_batch_{batch_size}_epoch_{n_epoch}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    state_dict = None
    best_state = None
    test_accuracy = 0
    for epoch in range(n_epoch):
        model.train()
        # would it be better to shuffle data each epoch?
        total_loss = 0
        for idx, samples in enumerate(get_one_batch(train_data, train_target,
                                      batch_size)):
            data = samples[0]
            target = samples[1]
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_prop = (idx + 1)*batch_size/len(train_data)
            if batch_prop < 1:
                print('\r', f'Training {batch_prop:.2%}  ', end='',
                      flush=True)
            else:
                print('\r', f'Training {1:.2%}  ', end='', flush=True)
            # TODO: loss
            total_loss += loss.item()*len(output)
        running_loss = total_loss/len(train_target)
        _, train_accuracy, train_loss = test(model, train_data.cuda(),
                                             train_target.cuda())
        _, val_accuracy, val_loss = test(model, val_data.cuda(),
                                         val_target.cuda())
        if test_data is None or test_target is None:
            test_accuracy = val_accuracy
            test_loss = val_loss
        else:
            _, test_accuracy, test_loss = test(model, test_data.cuda(),
                                               test_target.cuda())
        torch.cuda.empty_cache()
        if test_accuracy > best_test:
            best_train = train_accuracy
            best_val = val_accuracy
            best_test = test_accuracy
            best_state = [epoch, idx, running_loss, test_loss,
                          best_train, best_val, best_test]
            state_dict = deepcopy(model.state_dict())

        loss_list.append(running_loss)
        test_loss_list.append(test_loss)
        train_acc_list.append(train_accuracy)
        test_acc_list.append(test_accuracy)

        if epoch == 0:
            with open(os.path.join(save_dir, (model_name + '_train_log.txt')),
                      'wt', encoding="utf-8") as f:
                f.write(f'Epoch: {epoch:4}, Batch: {idx:3} '
                        f'| Train Loss: {running_loss:10.8f} '
                        f'| Val Loss: {test_loss:10.8f} '
                        f'| Train Accuracy: {train_accuracy:.3%} '
                        f'| Val Accuracy: {val_accuracy:.3%} '
                        f'| Test Accuracy: {test_accuracy:.3%}')
                f.write('\n')
        else:
            with open(os.path.join(save_dir, (model_name + '_train_log.txt')),
                      'a', encoding="utf-8") as f:
                f.write(f'Epoch: {epoch:4}, Batch: {idx:3} '
                        f'| Train Loss: {running_loss:10.8f} '
                        f'| Val Loss: {test_loss:10.8f} '
                        f'| Train Accuracy: {train_accuracy:.3%} '
                        f'| Val Accuracy: {val_accuracy:.3%} '
                        f'| Test Accuracy: {test_accuracy:.3%}')
                f.write('\n')

        print(f'Epoch: {epoch:4}, Batch: {idx:3} '
              f'| Train Loss: {running_loss:10.8f} '
              f'| Val Loss: {test_loss:10.8f} '
              f'| Train Accuracy: {train_accuracy:.3%} '
              f'| Val Accuracy: {val_accuracy:.3%} '
              f'| Test Accuracy: {test_accuracy:.3%}')
        torch.cuda.empty_cache()

    metric_list = pd.DataFrame({'Train Loss': loss_list,
                                'Val Loss': test_loss_list,
                                'Train accuracy': train_acc_list,
                                'Test accuracy': test_acc_list})
    metric_list.to_csv(os.path.join(save_dir, (model_name + '_train_log.csv')),
                       index=False, sep=',')
    # Or antoher way
    # import csv
    # rows = zip(loss_list,train_acc_list,test_acc_list)
    # with open(os.path.join(save_dir, (model_name + '_train_log.csv')),
    #           'w') as csvf:
    #     csvr = csv.writer(csvf)
    #     csvr.writerow(['Loss','Train accuracy','Test accuracy'])
    #     for row in rows:
    #         csvr.writerow(row)

    plot_curves(train_acc_list, test_acc_list, loss_list, test_loss_list,
                save_dir=save_dir)

    model_dir = os.path.join(save_dir, (model_name + '.pkl'))
    torch.save(state_dict, model_dir)
    print('Best Results: ')
    print(f'Epoch: {best_state[0]:4}, Batch: {best_state[1]:3} '
          f'| Train Loss: {best_state[2]:10.8f} '
          f'| Val Loss: {best_state[3]:10.8f} '
          f'| Train Accuracy: {best_state[4]:.3%} '
          f'| Val Accuracy: {best_state[5]:.3%} '
          f'| Test Accuracy: {best_state[6]:.3%}')
    with open(os.path.join(save_dir, (model_name + '_train_log.txt')),
              'r+', encoding="utf-8") as f:
        content = f.read()
        f.seek(0, 0)
        f.write('{:=^150s}'.format('Best Result')+'\n')
        f.write(f'Epoch: {best_state[0]:4}, Batch: {best_state[1]:3} '
                f'| Train Loss: {best_state[2]:10.8f} '
                f'| Val Loss: {best_state[3]:10.8f} '
                f'| Train Accuracy: {best_state[4]:.3%} '
                f'| Val Accuracy: {best_state[5]:.3%} '
                f'| Test Accuracy: {best_state[6]:.3%}')
        f.write('\n')
        f.write('{:=^150s}'.format('Training Log')+'\n')
        f.write(content)
    end = time.time()
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    print(f'Train time: {h:.0f}:{m:.0f}:{s:.0f}')


def test(model, data, target=None, batch_size=8192):
    """
    output is hard label
    """

    if torch.cuda.is_available():
        model = model.cuda()

    torch.cuda.empty_cache()
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss = None
    # try:
    #     with torch.no_grad():
    #         # copy cuda tensor to host memory then convert to ndarray
    #         output = model(data).cpu().data
    # except:
    output = None
    for idx, batch_data in enumerate(get_one_batch(data,
                                     batch_size=batch_size)):

        with torch.no_grad():
            batch_output = model(batch_data[0]).cpu().data
            # batch_output = model(batch_data[0].cuda()).cpu().data
        if idx == 0:
            output = batch_output
        else:
            output = torch.cat((output, batch_output), dim=0)
    if target is not None:
        loss = criterion(output, target.cpu()).item()
    pred = torch.max(output, dim=1)[1].numpy()

    accuracy = None
    if target is not None:
        target = target.cpu()
        accuracy = compute_accuracy(pred, target)

    return pred, accuracy, loss


def predict(model, feature_img, patch_size=13, target=None, batch_size=8192):
    """
    output is hard label
    """
    # model.load_state_dict(torch.load(model_weight_path))
    data = read_img_as_patches(feature_img, patch_size, to_tensor=True)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    with torch.no_grad():
        # copy cuda tensor to host memory then convert to ndarray
        output = model(data).cpu().data
    torch.cuda.empty_cache()
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

# train_mask, val_mask = load_masks(labeld_img,train_prop=TRAIN_PROP,
#                                   val_prop=VAL_PROP,
#                                   mask_dir='D:\\Github\\IceNet')
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
#     # one_scene = dirs
#     rest_scene = deepcopy(dirs)
#     del rest_scene[idx]
#     for i, one_out_name in enumerate(rest_scene):
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
#     HV_test = cv2.imread(os.path.join(root,one_scene,'imagery_HV4_by_4average.tif'))[:,:,0]
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
#     train(model, LOO_train_data, LOO_train_target, test_data, test_target, save_dir = os.path.join(root,one_scene))



# print("Done")


############################################# Debug ############################################

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
# dirs = os.listdir(root)

# all_train_data = None
# all_train_target = None
# all_val_data = None
# all_val_target = None

# for idx, dir_name in enumerate(dirs):

#     hh_path = os.path.join(root, dir_name, 'imagery_HH4_by_4average.tif')
#     hv_path = os.path.join(root, dir_name, 'imagery_HV4_by_4average.tif')
#     labeld_img_path = os.path.join(root, dir_name, 'all_train_mask.png')
#     HH = cv2.imread(hh_path)[:, :, 0]
#     HV = cv2.imread(hv_path)[:, :, 0]
#     labeld_img = cv2.imread(labeld_img_path)[:, :, 0]
#     if HH.shape != HV.shape or HH.shape != labeld_img.shape:
#         print("Input images have different sizes!")
#         exit()

#     HH = HH/255
#     HV = HV/255

#     feature_map = np.zeros((2, HH.shape[0], HH.shape[1]),dtype=float)
#     feature_map[0, :, :] = HH
#     feature_map[1, :, :] = HV

#     train_mask, val_mask = load_masks(labeld_img,
#                                       train_prop=TRAIN_PROP,
#                                       val_prop=VAL_PROP,
#                                       mask_dir=os.path.join(root,dir_name))
#     train_data, train_target = get_patch_samples(feature_map,
#                                                  labeld_img,
#                                                  train_mask,
#                                                  patch_size=patch_size,
#                                                  to_tensor=False)
#     val_data, val_target = get_patch_samples(feature_map,
#                                              labeld_img,
#                                              val_mask,
#                                              patch_size=patch_size,
#                                              to_tensor=False)
#     # append or deep copy?
#     if idx == 0:
#         all_train_data = train_data
#         all_train_target = train_target
#         all_val_data = val_data
#         all_val_target = val_target
#     else:
#         all_train_data = np.concatenate((all_train_data, train_data), axis=0)
#         all_train_target = np.concatenate((all_train_target, train_target),
#                                           axis=0)
#         all_val_data = np.concatenate((all_val_data, val_data), axis=0)
#         all_val_target = np.concatenate((all_val_target, val_target), axis=0)
#         # Shuffle?

# state = np.random.get_state()
# np.random.shuffle(all_train_data)
# np.random.set_state(state)
# np.random.shuffle(all_train_target)

# all_train_data = torch.from_numpy(all_train_data).float()
# all_train_target = torch.from_numpy(all_train_target).long()

# all_val_data = torch.from_numpy(all_val_data).float()
# all_val_target = torch.from_numpy(all_val_target).long()

# # test_data = all_train_data
# # test_target = all_train_target

# # delete model?
# model = SSResNet.ResNet(config)
# train(model, all_train_data, all_train_target, all_val_data,
#       all_val_target, save_dir=os.path.join(root, dirs[0]))

# print("Done")



############################################# ice-water ############################################
# patch_size = 13
# to_tensor = True
# config = {
#     'input_shape': [1, 2, patch_size, patch_size],
#     'n_classes': 2,
#     'channels': 128,
#     'blocks': 3,
#     'is_bn': True,
#     'is_dropout': False,
#     'p': 0.2
# }

# root = "D:/Data/Resnet/ice-water"
# dirs = os.listdir(root)

# all_train_data = None
# all_train_target = None
# all_val_data = None
# all_val_target = None

# for idx, dir_name in enumerate(dirs):

#     hh_path = os.path.join(root,dir_name,'imagery_HH4_by_4average.tif')
#     hv_path = os.path.join(root,dir_name,'imagery_HV4_by_4average.tif')
#     labeld_img_path = os.path.join(root,dir_name,'all_train_mask.png')
#     HH = cv2.imread(hh_path)[:,:,0]
#     HV = cv2.imread(hv_path)[:,:,0]
#     labeld_img = cv2.imread(labeld_img_path)[:,:,0]
#     if HH.shape != HV.shape or HH.shape != labeld_img.shape:
#         print("Input images have different sizes!")
#         exit()

#     HH = HH/255
#     HV = HV/255

#     feature_map = np.zeros((2, HH.shape[0], HH.shape[1]),dtype=float)
#     feature_map[0,:,:] = HH
#     feature_map[1,:,:] = HV
  
#     train_mask, val_mask = load_masks(labeld_img,train_prop=TRAIN_PROP,val_prop=VAL_PROP, mask_dir = os.path.join(root,dir_name))
#     train_data, train_target = get_patch_samples(feature_map, labeld_img, train_mask, patch_size=patch_size, to_tensor=False)
#     val_data, val_target = get_patch_samples(feature_map, labeld_img, val_mask, patch_size=patch_size, to_tensor=False)
    
#     # append or deep copy?
#     if idx == 0:
#         all_train_data = train_data
#         all_train_target = train_target
#         all_val_data = val_data
#         all_val_target = val_target
#     else:
#         all_train_data = np.concatenate((all_train_data, train_data), axis=0)
#         all_train_target = np.concatenate((all_train_target, train_target), axis=0)
#         all_val_data = np.concatenate((all_val_data, val_data), axis=0)
#         all_val_target = np.concatenate((all_val_target, val_target), axis=0)
#         # Shuffle?

# state = np.random.get_state()
# np.random.shuffle(all_train_data)
# np.random.set_state(state)
# np.random.shuffle(all_train_target)

# all_train_data = torch.from_numpy(all_train_data).float()
# all_train_target = torch.from_numpy(all_train_target).long()
# all_val_data = torch.from_numpy(all_val_data).float()
# all_val_target = torch.from_numpy(all_val_target).long()


# # test_data = all_train_data
# # test_target = all_train_target

# # delete model?
# model  = SSResNet.ResNet(config)
# train(model, all_train_data, all_train_target, all_val_data, all_val_target, save_dir = os.path.join(root,dirs[0]))

# print("Done")