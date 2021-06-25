import cv2
import numpy as np
import os

def relabel_train_val_from_sip(dir):

    if os.path.exists(os.path.join(dir, 'all_train_mask.png')):
        return

    yong_RGB = np.array([128,0,128])
    first_year_RGB = np.array([255,255,0])
    multi_year_RGB = np.array([255,0,0])
    water_RGB = np.array([0,0,255])   
    
    labeled_img1 = cv2.imread(os.path.join(dir, 'train_mask.png'))
    labeled_img1 = screen = cv2.cvtColor(labeled_img1, cv2.COLOR_BGR2RGB)

    labeled_img2 = cv2.imread(os.path.join(dir, 'val_mask.png'))
    labeled_img2 = screen = cv2.cvtColor(labeled_img2, cv2.COLOR_BGR2RGB)



    # another way to do it https://stackoverflow.com/questions/33196130/replacing-rgb-values-in-numpy-array-by-integer-is-extremely-slow
    idx_y1 = np.where(np.all(labeled_img1 == yong_RGB, axis=-1))
    idx_f1 = np.where(np.all(labeled_img1 == first_year_RGB, axis=-1))
    idx_m1 = np.where(np.all(labeled_img1 == multi_year_RGB, axis=-1))
    idx_w1 = np.where(np.all(labeled_img1 == water_RGB, axis=-1))

    idx_y2 = np.where(np.all(labeled_img2 == yong_RGB, axis=-1))
    idx_f2 = np.where(np.all(labeled_img2 == first_year_RGB, axis=-1))
    idx_m2 = np.where(np.all(labeled_img2 == multi_year_RGB, axis=-1))
    idx_w2 = np.where(np.all(labeled_img2 == water_RGB, axis=-1))


    relabel = np.zeros((labeled_img1.shape[0],labeled_img1.shape[1]),dtype=int)
    relabel[idx_y1] = 1
    relabel[idx_f1] = 2
    relabel[idx_m1] = 3
    relabel[idx_w1] = 4

    relabel[idx_y2] = 1
    relabel[idx_f2] = 2
    relabel[idx_m2] = 3
    relabel[idx_w2] = 4

    cv2.imwrite(os.path.join(dir, 'all_train_mask.png'),relabel)