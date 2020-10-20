
import os
import random
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
true_test_root = '/mnt/A/CIKM2017/CIKM_datasets/test/'
true_validation_root = '/mnt/A/CIKM2017/CIKM_datasets/validation/'
pred_root = '/mnt/A/meteorological/2500_ref_seq/'

import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.ndimage import gaussian_filter
from numpy.lib.stride_tricks import as_strided as ast
import math
import numpy as np


from math import sqrt
# def _tf_fspecial_gauss(size, sigma):
#     """Function to mimic the 'fspecial' gaussian MATLAB function
#     """
#     x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
#
#     x_data = np.expand_dims(x_data, axis=-1)
#     x_data = np.expand_dims(x_data, axis=-1)
#
#     y_data = np.expand_dims(y_data, axis=-1)
#     y_data = np.expand_dims(y_data, axis=-1)
#
#     x = tf.constant(x_data, dtype=tf.float32)
#     y = tf.constant(y_data, dtype=tf.float32)
#
#     g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
#     return g / tf.reduce_sum(g)
#
#
# def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
#     window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
#     K1 = 0.01
#     K2 = 0.03
#     K3 = 0.15
#     L = 1  # depth of image (255 in case the image has a differnt scale)
#     C1 = (K1*L)**2
#     C2 = (K2*L)**2
#     C3 = (K3*L)**2
#     mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='SAME')
#     mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='SAME')
#
#     mu1_sq = mu1*mu1
#     mu2_sq = mu2*mu2
#     mu1_mu2 = mu1*mu2
#     sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='SAME') - mu1_sq
#     sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='SAME') - mu2_sq
#     sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='SAME') - mu1_mu2
#     if cs_map:
#         value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
#                     (sigma1_sq + sigma2_sq + C2)),
#                 (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
#     else:
#         value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
#                     (sigma1_sq + sigma2_sq + C2))
#
#     if mean_metric:
#         value = tf.reduce_mean(value)
#     return value
#
#
# def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
#     weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
#     mssim = []
#     mcs = []
#     for l in range(level):
#         ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
#         mssim.append(tf.reduce_mean(ssim_map))
#         mcs.append(tf.reduce_mean(cs_map))
#         filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
#         filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
#         img1 = filtered_im1
#         img2 = filtered_im2
#
#     # list to tensor of dim D+1
#     mssim = tf.stack(mssim, axis=0)
#     mcs = tf.stack(mcs, axis=0)
#
#     value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
#                             (mssim[level-1]**weight[level-1]))
#
#     if mean_metric:
#         value = tf.reduce_mean(value)
#     return value
# def image_to_4d(image):
#     image = tf.expand_dims(image, 0)
#     image = tf.expand_dims(image, -1)
#     return image
#
# BATCH_SIZE = 1
# CHANNELS = 1
# image1 = tf.placeholder(tf.float32, shape=[101, 101])
# image2 = tf.placeholder(tf.float32, shape=[101, 101])
# image4d_1 = image_to_4d(image1)
# image4d_2 = image_to_4d(image2)
# ssim_index = tf_ssim(image4d_1, image4d_2)
# msssim_index = tf_ms_ssim(image4d_1, image4d_2)
#
# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
#
# def TF_SSIM(img1,img2):
#     tf_ssim_ = sess.run(ssim_index,
#                             feed_dict={image1: img1, image2: img2})
#
#     tf_msssim_ = sess.run(msssim_index,
#                             feed_dict={image1: img1, image2: img2})
#
#     return tf_ssim_,tf_msssim_
#
#     # print('tf_ssim_none', tf_ssim)
#     # print('tf_msssim_none', tf_msssim)
#







def PCC(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    p = np.corrcoef(y_true, y_pred)[1,0]
    if math.isnan(p):
        return -1
    return p

    #
    # ux = np.mean(y_true)
    # uy = np.mean(y_pred)
    # var_x = np.var(y_true)
    # var_y = np.var(y_pred)
    # if var_x ==0 or var_y == 0:
    #     print('error',str(var_y),str(var_x))
    #     return -1
    # std_x = np.sqrt(var_x)
    # std_y = np.sqrt(var_y)
    # var_xy = np.sum(np.dot((y_true - ux), (y_pred - uy))) / (101 * 101 - 1)
    # return var_xy / (std_x*std_y)

def block_view(A, block=(3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape = (int(A.shape[0]/ block[0]), int(A.shape[1]/ block[1]))+ block
    strides = (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return ast(A, shape= shape, strides= strides)

def SSIM(y_true, y_pred, C1=0.01**2, C2=0.03**2, C3=0.02**2):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    ux = np.mean(y_true)
    uy = np.mean(y_pred)

    var_x = np.var(y_true)
    var_y = np.var(y_pred)
    std_x = np.sqrt(var_x)
    std_y = np.sqrt(var_y)
    # var_xy = np.mean(np.cov(y_true,y_pred))[0,1]
    var_xy = np.sum(np.dot((y_true-ux),(y_pred-uy)))/(101*101-1)

    l = ((2*ux*uy+C1)/(ux*ux+uy*uy+C1))
    c = ((2*std_x*std_y+C1)/(var_x+var_y+C2))
    s = ((var_xy+C3)/(std_x*std_y+C3))

    return l*c*s


def MSE(pre,real):
    return np.mean(np.square(pre-real))

def MAE(pre,real):
    return np.sum(np.absolute(pre - real))/len(pre.reshape(-1))

def normalization(data):
    return data.astype(np.float32)/255.0

def eval_validation(true_fold,pred_fold,eval_type):
    res = 0
    for i in range(1,2000,1):
        # print('complete ',str(i*100.0/2000),'%')
        true_current_fold = true_fold + 'sample_' + str(i) + '/'
        pre_current_fold = pred_fold + 'sample_' + str(i) + '/'
        sample_res = 0
        for t in range(6,16,1):
            pred_path = pre_current_fold+'img_'+str(t)+'.png'
            true_path = true_current_fold+'img_'+str(t)+'.png'
            pre_img = imread(pred_path)
            true_img = imread(true_path)

            if eval_type == 'mse':
                current_res = MSE(normalization(pre_img),normalization(true_img))
            elif eval_type == 'mae':
                current_res = MAE(normalization(pre_img), normalization(true_img))
            elif eval_type == 'pcc':
                current_res = PCC(normalization(pre_img), normalization(true_img))
            # elif eval_type == 'ssim':
            #     current_res = TF_SSIM(normalization(pre_img), normalization(true_img))
            sample_res = sample_res + current_res
        sample_res = sample_res/10
        res = res + sample_res
    res = res / 2000
    return res

def eval_test(true_fold,pred_fold,eval_type):
    res = 0
    for i in range(1,4000,1):
        # print('complete ',str(i*100.0/4000),' %')
        true_current_fold = true_fold+'sample_'+str(i)+'/'
        pre_current_fold = pred_fold+'sample_'+str(i)+'/'
        sample_res = 0
        skip = 0
        for t in range(6, 16, 1):
            pred_path = pre_current_fold+'img_'+str(t)+'.png'
            true_path = true_current_fold+'img_'+str(t)+'.png'
            pre_img = imread(pred_path)
            true_img = imread(true_path)
            if eval_type == 'mse':
                current_res = MSE(normalization(pre_img),normalization(true_img))
            elif eval_type == 'mae':
                current_res = MAE(normalization(pre_img), normalization(true_img))
            elif eval_type == 'pcc':
                current_res = PCC(pre_img, true_img)
                if current_res == -1:
                    skip=skip+1
            # elif eval_type == 'ssim':
            #     current_res = TF_SSIM(normalization(pre_img), normalization(true_img))[0]
            # elif eval_type == 'ms-ssim':
            #     current_res = TF_SSIM(normalization(pre_img), normalization(true_img))[1]
            #     if math.isnan(current_res):
            #         skip=skip+1
            sample_res = sample_res + current_res
        sample_res = sample_res/(10-skip)
        res = res+sample_res

    res = res/4000
    return res


def sequence_mse(true_fold,pred_fold,eval_type='mse'):
    res = [0 for _ in range(10)]

    for i in range(1, 4000, 1):
        # print('complete ',str(i*100.0/4000),' %')
        true_current_fold = true_fold + 'sample_' + str(i) + '/'
        pre_current_fold = pred_fold + 'sample_' + str(i) + '/'
        sample_res = []
        skip = 0
        for t in range(6, 16, 1):
            pred_path = pre_current_fold + 'img_' + str(t) + '.png'
            true_path = true_current_fold + 'img_' + str(t) + '.png'
            pre_img = imread(pred_path)
            true_img = imread(true_path)
            if eval_type == 'mse':
                current_res = MSE(normalization(pre_img), normalization(true_img))
            elif eval_type == 'mae':
                current_res = MAE(normalization(pre_img), normalization(true_img))
            elif eval_type == 'pcc':
                current_res = PCC(pre_img, true_img)
                if current_res == -1:
                    skip = skip + 1
            # elif eval_type == 'ssim':
            #     current_res = TF_SSIM(normalization(pre_img), normalization(true_img))[0]
            # elif eval_type == 'ms-ssim':
            #     current_res = TF_SSIM(normalization(pre_img), normalization(true_img))[1]
            #     if math.isnan(current_res):
            #         skip = skip + 1
            sample_res.append(current_res)

        for i in range(len(res)):
            res[i] = res[i]+sample_res[i]

    for i in range(len(res)):
        res[i] = res[i] / 4000
    return res


def plot(datas,names,model_name):

    x = []
    for i in range(1, 11, 1):
        x.append(i*6)
    plt.figure()
    for idx,name in enumerate(names):
        plt.plot(x,datas[name])

    plt.grid()
    plt.legend(model_name)
    plt.xticks(x)
    plt.savefig('evalute.png')
    plt.show()


def effect_show(test_model_list,model_name):
    thr = 100
    random_select_index = random.randint(1, 4000)
    pred_imgs = []
    for model_name in test_model_list:
        model_samples_path = pred_root+model_name+'/sample_'+str(random_select_index)+'/'
        model_imgs = []
        for i in range(6,16):
            img_path = model_samples_path+'img_'+str(i)+'.png'
            img = imread(img_path)
            new_img = np.zeros(img.shape).astype(np.int)
            for i in range(len(img)):
                for j in range(len(img[i])):
                    if img[i][j] > thr:
                        new_img[i][j] = 1
                    else:
                        pass
            model_imgs.append(new_img)
        pred_imgs.append(model_imgs)

    ground_truth_imgs = []
    for i in range(6,16):
        img_path = true_test_root+'/sample_'+str(random_select_index)+'/img_'+str(i)+'.png'
        img = imread(img_path)
        new_img = np.zeros(img.shape).astype(np.int)
        for i in range(len(img)):
            for j in range(len(img[i])):
                if img[i][j] > thr:
                    new_img[i][j] = 1
                else:
                    pass
        ground_truth_imgs.append(new_img)

    error_pred_model = []
    for idx,current_model_imgs in enumerate(pred_imgs):

        error_view_imgs = []
        for i in range(len(ground_truth_imgs)):
            ground_img = ground_truth_imgs[i]
            pre_img = current_model_imgs[i]
            view_img = (ground_img^pre_img)*255
            error_view_imgs.append(view_img)
        error_pred_model.append(error_view_imgs)

    aspect_ratio = float(4)
    plt.figure(figsize=(len(error_pred_model), aspect_ratio))
    plt.title(model_name[idx])
    gs = gridspec.GridSpec(len(error_pred_model), 10)
    gs.update(wspace=0., hspace=0.)
    plt.subplots_adjust(left=0.04, top=0.96, right=0.96, bottom=0.04, wspace=0.01, hspace=0.01)

    for idx,error_view_imgs in enumerate(error_pred_model):
        for j,img in enumerate(error_view_imgs):
            plt.subplot(gs[idx*10+j])
            plt.imshow(img,interpolation='none',aspect='auto',cmap=plt.cm.gray)
            plt.title(str((j+1)*6), fontsize=6)
            plt.tick_params(
                axis='both',
                which='both',
                bottom='off',
                top='off',
                left='off',
                right='off',
                labelbottom='off',
                labelleft='off',
            )
            plt.axis('off')

    plt.show()





if __name__ == '__main__':

    test_model_list = [
        'CIKM_dec_ConvGRU_test',
        'CIKM_dec_ConvLSTM_test',
        'CIKM_dec_TrajGRU_test',
        'CIKM_dec_PredRNN_test',
        'CIKM_dec_ST_ConvLSTM_test',
        'CIKM_dec_PFST_ConvLSTM_test',
    ]
    model_name = [
        'ConvGRU',
        'ConvLSTM',
        'TrajGRU',
        'PredRNN',
        'ST_ConvLSTM',
        'PFST_ConvLSTM',
    ]
    # effect_show(test_model_list,model_name)
    seq_mse_res = {}
    for model in test_model_list:
        seq_mse = sequence_mse(true_test_root, pred_root + model + '/', 'mse')
        seq_mse_res[model] = seq_mse
    print(seq_mse_res)
    plot(seq_mse_res,test_model_list,model_name)



    # validation_model_list = ['CIKM_dec_ST_ConvLSTM_validation','CIKM_dec_ConvGRU_validation', 'CIKM_dec_ConvLSTM_validation']

    # test_model_list = ['CIKM_ST_ConvLSTM_test', 'CIKM_ConvGRU_test', 'CIKM_ConvLSTM_test']
    # validation_model_list = ['CIKM_ST_ConvLSTM_validation', 'CIKM_TrajGRU_validation','CIKM_ConvGRU_validation',
    #                          'CIKM_ConvLSTM_validation']
    # validation_model_mse = {}
    # for model in validation_model_list:
    #     mse = eval_validation(true_validation_root, pred_root + model+'/', 'mse')
    #     validation_model_mse[model] = mse
    # print(validation_model_mse)

    # test_model_mse = {}
    # for model in test_model_list:
    #     mse = eval_test(true_test_root, pred_root + model+'/', 'mse')
    #     test_model_mse[model] = mse
    #     print('model is:',model)
    #     print(test_model_mse[model])
    # test_model_mae = {}

    # for model in test_model_list:
    #     mae = eval_test(true_test_root, pred_root + model + '/', 'mae')
    #     test_model_mae[model] = mae
    # print(test_model_mae)
    # print('ssim')
    # test_model_ssim = {}
    # for id,model in enumerate(test_model_list):
    #     ssim = eval_test(true_test_root, pred_root + model + '/', 'ssim')
    #     test_model_ssim[model] = ssim
    # print(test_model_ssim)
    # print('*'*80)
    #
    # print('ms-ssim')
    # test_model_ms_ssim = {}
    # for model in test_model_list:
    #     ms_ssim = eval_test(true_test_root, pred_root + model + '/', 'ms-ssim')
    #     test_model_ms_ssim[model] = ms_ssim
    # print(test_model_ms_ssim)
    # test_model_pcc = {}
    # for model in test_model_list:
    #     pcc = eval_test(sess,true_test_root, pred_root + model + '/', 'pcc')
    #     test_model_pcc[model] = pcc
    # print(test_model_pcc)

    # print(seq_mse)

    # ST_ConvLSTM_mse = [0.0024288102901641653, 0.003943319082303788, 0.005390102719733022, 0.006799462498499452, 0.008158839733144305, 0.00952125993827667, 0.010843842759111794, 0.012042934615375998, 0.013134818804459427, 0.014214933834487966]
    # ConvGRU_mse = [0.002305122214578603, 0.003916686069469506, 0.005505802148131806, 0.007089604998223876, 0.00868397690187794, 0.01032106976072646, 0.01188595761911165, 0.013255351553541914, 0.01445867685882331, 0.01553596562302846]
    # ConvLSTM_mse = [0.002295292251142598, 0.003928239543610289, 0.005540265110989367, 0.007083509935046095, 0.008604068437414753, 0.010130964369691355, 0.01155991896984051, 0.012890271273096004, 0.01410613953669963, 0.015222110682059793]
    # dec_ST_ConvLSTM_mse = [0.002364268766264843, 0.003924998539975604, 0.005368746128718385, 0.006756065742258215, 0.008097049896226963, 0.009436994450309611, 0.010671709660275155, 0.011779709733296841, 0.012778628737465623, 0.013759537742738758]
    # dec_ConvGRU_mse = [0.0024157174499055147, 0.00393797474564235, 0.005376405584767781, 0.006824714914899232, 0.008270984332273657, 0.009713952445783434, 0.011055264989359784, 0.01226929799222671, 0.013360691534066063, 0.014434498674891074]
    # dec_ConvLSTM_mse = [0.002379041242172434, 0.003937165613700017, 0.0054181525517010415, 0.006915119287572452, 0.008500685820710714, 0.010079977763036367, 0.011591412258920172, 0.012939707933073806, 0.014157152997035155, 0.015297246081059711]

    # ST_ConvLSTM_mse = [0.002302415528975871, 0.0039257768223305905, 0.005485519943306372, 0.00702758381395779, 0.008578271590690746, 0.010112517563517031, 0.011512441961131117, 0.012731947152129578, 0.013820672234536688, 0.014873968772197259]
    # ConvGRU_mse = [0.002292289455213222, 0.003897478523168502, 0.005540548081439738, 0.007177185952796208, 0.008780076390316026, 0.010361579176989835, 0.011876730150050207, 0.013251351497496216, 0.014447874331055573, 0.015510209560919976]
    # ConvLSTM_mse = [0.002332684729696666, 0.0039732200092421404, 0.00555670658428221, 0.007159864549257691, 0.008815937865809247, 0.010477123910687624, 0.012064420173284816, 0.013430322390984656, 0.014589850207026757, 0.015688560969136234]
    # dec_ST_ConvLSTM_mse = [0.002344160946124248, 0.003938462443034041, 0.005453599778101306, 0.006914397735837156, 0.00835189992823507, 0.009795115892367904, 0.011185708106666425, 0.012469912361173556, 0.013657610348433082, 0.014840640853370132]
    # dec_ConvGRU_mse = [0.0026988081292238348, 0.004471600576382116, 0.006056587053779367, 0.007522618366359893, 0.008887108978367905, 0.01020160230613692, 0.011468532416663948, 0.012602082935030921, 0.01361446809602785, 0.014613186329894234]
    # dec_ConvLSTM_mse = [0.0023624405927522504, 0.003915410714364953, 0.005393563923598776, 0.006865441179938898, 0.008360895798168712, 0.009896392613007265, 0.01136580711356146, 0.01268309452279027, 0.013799309661371807, 0.014857410416574566]
    # print(
    # 'ConvGRU:'+format(np.mean(np.array(ConvGRU_mse))*100,'.4f')
    # , 'ConvLSTM:' + format(np.mean(np.array(ConvLSTM_mse)) * 100, '.4f')
    # , 'ST_ConvLSTM:'+format(np.mean(np.array(ST_ConvLSTM_mse))*100,'.4f')
    # , 'dec_ConvGRU:' + format(np.mean(np.array(dec_ConvGRU_mse)) * 100, '.4f')
    # , 'dec_ConvLSTM:' + format(np.mean(np.array(dec_ConvLSTM_mse)) * 100, '.4f')
    # , 'dec_ST_ConvLSTM:' + format(np.mean(np.array(dec_ST_ConvLSTM_mse)) * 100, '.4f')
    # )
    #
    #
    # plot(
    #     [ConvGRU_mse,ConvLSTM_mse,ST_ConvLSTM_mse,dec_ConvGRU_mse,dec_ConvLSTM_mse,dec_ST_ConvLSTM_mse],
    #     [
    #         'ConvGRU:'+format(np.mean(np.array(ConvGRU_mse))*100,'.4f')
    #         , 'ConvLSTM:' + format(np.mean(np.array(ConvLSTM_mse)) * 100, '.4f')
    #         , 'ST_ConvLSTM:'+format(np.mean(np.array(ST_ConvLSTM_mse))*100,'.4f')
    #         , 'dec_ConvGRU:' + format(np.mean(np.array(dec_ConvGRU_mse)) * 100, '.4f')
    #         , 'dec_ConvLSTM:' + format(np.mean(np.array(dec_ConvLSTM_mse)) * 100, '.4f')
    #         , 'dec_ST_ConvLSTM:' + format(np.mean(np.array(dec_ST_ConvLSTM_mse)) * 100, '.4f')
    #
    #     ]
    # )

    # mse = eval_test(true_test_root, pred_root + 'CIKM_cl_dec_ST_ConvLSTM_test/', 'mse')
    # print('CIKM_cl_dec_ST_ConvLSTM_test mse is:', mse)
    # mse = eval_test(true_test_root, pred_root + 'CIKM_dec_TrajGRU_test/', 'mse')
    # print('CIKM_dec_TrajGRU_test mse is:', mse)
    # mse = eval_test(true_test_root, pred_root + 'CIKM_Conv2DLSTM_test/', 'mse')
    # print('CIKM_Conv2DLSTM_test mse is:', mse)
    # mse = eval_test(true_test_root, pred_root + 'CIKM_cl_dec_ST_ConvLSTM_test/', 'mse')
    # print('CIKM_cl_dec_ST_ConvLSTM_test mse is:', mse)

    # mse = eval_validation(true_validation_root, pred_root + 'CIKM_ConvGRU_validation/', 'mse')
    # print('CIKM_ConvGRU_validation mse is:', mse)
    # mae = eval_test(true_test_root, pred_root + 'CIKM_ST_ConvLSTM_test/', 'mae')
    # print('mae is:', mae)
    # ssim = eval_test(true_test_root, pred_root + 'CIKM_ConvGRU_test/', 'ssim')
    # print('ssim is:', ssim)
    # pcc = eval_test(true_test_root, pred_root + 'CIKM_ConvGRU_test/', 'pcc')
    # print('pcc is:', pcc)