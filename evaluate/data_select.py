from scipy.misc import imread
import numpy as np
import os

def dBZ_to_pixel(reflective):
    pixel_vals = ((reflective + 10.0) * 255.0) / 95.0
    return pixel_vals



max_reflective = 40
max_pixel = dBZ_to_pixel(max_reflective)


real_root = '/mnt/A/CIKM2017/CIKM_datasets/test/'
validation_sample = []
for s_index in range(1,4001,1):
    # print('s_index is:',str(s_index))
    real_sample_path = os.path.join(real_root,'sample_'+str(s_index))
    flag = True
    for img_index in range(6,16,1):
        real_path = os.path.join(real_sample_path,'img_'+str(img_index)+'.png')
        real_img = imread(real_path)
        max_real_img_pixel = np.max(real_img)
        if max_real_img_pixel<max_pixel:
            flag = False
            break
        else:
            pass
    if flag:
        validation_sample.append(s_index)

validation_sample = np.array(validation_sample).astype(np.str)
np.savetxt('valid_test.txt',validation_sample,fmt='%s')
print('data selection finish and the shape of valid sample is:',validation_sample.shape)
