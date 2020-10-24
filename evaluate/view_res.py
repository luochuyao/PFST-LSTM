

import random
import numpy as np
from scipy.misc import imread,imsave
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from util.color_map import *

evaluate_root = '/mnt/A/meteorological/2500_ref_seq/'
test_root = '/mnt/A/CIKM2017/CIKM_datasets/test/'

def pixel_to_dBZ(img):
    img = img.astype(np.float)/255.0
    img = img * 95.0
    img[img<15] = 0
    return img.astype(np.int)

def color_radar(img,flag=True):
    if flag:
        img = pixel_to_dBZ(img)
        img = mapping(img)
    else:
        pass
    return img

def sample_radar_sequence(sample_index):
    g_root = test_root
    p_root = evaluate_root
    evaluate_folds = [
        "CIKM_dec_ConvLSTM_test",
        "CIKM_dec_TrajLSTM_test",
        "CIKM_dec_ST_TrajLSTM_test",
        "CIKM_dec_ST_ConvLSTM_test",
        "CIKM_dec_PF_ConvLSTM_test",
        "CIKM_dec_PFST_ConvLSTM_test",
    ]

    pre_res = []
    for evaluate_fold in evaluate_folds:
        sample_pred_path = p_root+evaluate_fold+'/sample_'+str(sample_index)+'/'
        # print(sample_pred_path)
        model_res = []
        for i in range(6,16,1):
            img_path = sample_pred_path+'img_'+str(i)+'.png'
            pred_imgs = imread(img_path)
            pred_imgs = pred_imgs.astype(np.uint8)
            model_res.append(pred_imgs)
        pre_res.append(model_res)
    sample_ground_truth_path = g_root+'sample_'+str(sample_index)+'/'
    ground_truth_res = []
    for i in range(1,16,1):
        img_path = sample_ground_truth_path+'img_'+str(i)+'.png'
        real_img = imread(img_path)
        real_img = real_img.astype(np.uint8)
        ground_truth_res.append(real_img)
    return pre_res, ground_truth_res

def plot_radar(preds,ground_truths,img_name,flag = True):

    fig = plt.figure(figsize=(15,7))
    gs = GridSpec(7,15)
    # fig,ax = plt.subplots(nrows = 6,ncols = 15)
    for i in range(15):
        ax = plt.subplot2grid((7,15),(0,i))
        ax.set_xticks([])
        ax.set_yticks([])
        if flag:
            ax.imshow(color_radar(ground_truths[i]))
        else:
            ax.imshow(ground_truths[i],cmap='Greys_r')

    for index in range(len(preds)):
        current_pred_imgs = preds[index]

        for i in range(10):

            ax = plt.subplot2grid((7, 15), (index+1, i+5))
            if flag:
                ax.imshow(color_radar(current_pred_imgs[i]))
            else:
                ax.imshow(current_pred_imgs[i],cmap='Greys_r')
            ax.set_xticks([])
            ax.set_yticks([])
    print('save path is:'+str(img_name)+'view_radar.png')
    plt.savefig(str(img_name)+'view_radar.png')

def view_result(sample_index):
    pred, real = sample_radar_sequence(sample_index)
    plot_radar(pred, real, sample_index, True)


if __name__ == '__main__':
    view_result('2361')
