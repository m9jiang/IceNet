import cv2
import numpy as np
labeled_img = cv2.imread('D:/Github/IceNet/data/20100524_034756_hh_train_mask.png')
labeled_img = screen = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB)

yong_RGB = np.array([128,0,128])
first_year_RGB = np.array([255,255,0])
multi_year_RGB = np.array([255,0,0])
water_RGB = np.array([0,0,255])


idx_y = np.where(np.all(labeled_img == yong_RGB, axis=-1))
idx_f = np.where(np.all(labeled_img == first_year_RGB, axis=-1))
idx_m = np.where(np.all(labeled_img == multi_year_RGB, axis=-1))
idx_w = np.where(np.all(labeled_img == water_RGB, axis=-1))

relabel = np.zeros((labeled_img.shape[0],labeled_img.shape[1]),dtype=int)
relabel[idx_y] = 1
relabel[idx_f] = 2
relabel[idx_m] = 3
relabel[idx_w] = 4

cv2.imwrite('D:/Github/IceNet/data/20100524_034756_label.png',relabel)