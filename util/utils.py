import os
import shutil
import torch.nn as nn
import numpy as np
from math import *
import jpype
import copy
import cv2
import torch
from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch.nn.functional as F



def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

def deconv(in_planes, out_planes,kernel_size = 3,stride = 2,padding = 1,output_padding = 1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=output_padding, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )

def nearest_neighbor_advection(im, flow):
    """

    Parameters
    ----------
    im : np.ndarray
        Shape: (batch_size, C, H, W)
    flow : np.ndarray
        Shape: (batch_size, 2, H, W)
    Returns
    -------
    new_im : nd.NDArray
    """
    predict_frame = np.empty(im.shape, dtype=im.dtype)
    batch_size, channel_num, height, width = im.shape
    assert channel_num == 1
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    interp_grid = np.hstack([grid_x.reshape((-1, 1)), grid_y.reshape((-1, 1))])
    for i in range(batch_size):
        flow_interpolator = NearestNDInterpolator(interp_grid, im[i].ravel())
        predict_grid = interp_grid + np.hstack([flow[i][0].reshape((-1, 1)),
                                                flow[i][1].reshape((-1, 1))])
        predict_frame[i, 0, ...] = flow_interpolator(predict_grid).reshape((height, width))
    return predict_frame
def pixel_to_dBZ(img):
    img = img.astype(np.float)/255.0
    img = img * 95.0
    img[img<15] = 0
    return img.astype(np.int)

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)


def get_cell_param(parameter):

    param = {}
    param['input_channels'] = parameter[0]
    param['output_channels'] = parameter[1]
    param['input_to_state_kernel_size'] = (parameter[2],parameter[2])
    param['state_to_state_kernel_size'] = (parameter[3],parameter[3])
    if len(parameter)==5:
        param['input_to_input_kernel_size'] = (parameter[4],parameter[4])
    return param

def get_pool_param(parameter,mode = 'max',padding = 'SAME'):

    param = {}
    param['padding'] = padding
    param['pool_mode'] = mode
    param['pool_size'] = (1,parameter[0],parameter[0],1)
    param['strides'] = (1,parameter[1],parameter[1],1)

    return param

def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")

def nor(frames):
    new_frames = frames.astype(np.float32)/255.0
    return new_frames

def de_nor(frames):
    new_frames = copy.deepcopy(frames)
    new_frames *= 255.0
    new_frames = new_frames.astype(np.uint8)
    return new_frames

def normalization(frames,up=80):
    new_frames = frames.astype(np.float32)
    new_frames /= (up/2)
    new_frames -= 1
    return new_frames

def denormalization(frames,up=80):
    new_frames = copy.deepcopy(frames)
    new_frames += 1
    new_frames *= (up/2)
    new_frames = new_frames.astype(np.uint8)
    return new_frames

def get_conv_param(parameter,padding,activate='relu',reset = False):
    param = {}
    if reset:
        param['in_channel'] = parameter[1]
        param['out_channel'] = parameter[0]
    else:
        param['in_channel'] = parameter[0]
        param['out_channel'] = parameter[1]
    param['kernel_size']=(parameter[2],parameter[2])
    if len(parameter)==4:
        param['stride'] = parameter[3]
    else:
        param['stride'] = 1
    if len(padding)==2:
        param['output_padding'] = padding[1]
    param['padding'] = padding[0]
    param['activate'] = activate
    return param

def clean_fold(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

# input: B, C, H, W
# flow: [B, 2, H, W]

def wrap(input, flow):
    B, C, H, W = input.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).cuda()
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).cuda()
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(input, vgrid)
    return output

def pre(second_img,flow):

    second_img = second_img[0]
    flow = flow[0]
    flow = flow.transpose((1, 2, 0))
    w = int(second_img.shape[1])
    h = int(second_img.shape[0])
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    coords = np.float32(np.dstack([x_coords, y_coords]))
    pixel_map = coords + flow
    new_frame = cv2.remap(second_img, pixel_map, None, 1)
    new_frame = new_frame[np.newaxis,:,:,:]
    return np.array(new_frame)

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    UNKNOWN_FLOW_THRESH = 1e7
    SMALLFLOW = 0.0
    LARGEFLOW = 1e8

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)




if __name__ == '__main__':
    from data.CIKM.data_iterator import *
    from model.varflow import *
    import cv2
    from torch.autograd import Variable
    from util.color_map import mapping
    index = 1
    batch_size = 4
    varflow_factory = VarFlowFactory(max_level=4, start_level=0, n1=2, n2=2, rho=2.8, alpha=1400, sigma=1.5)

    while True:

        dat, (index, b_cup) = sample(batch_size, data_type='test', index=index)
        if index < 2360:
            continue
        var_flows = []
        flow_imgs = []
        for t in range(14):
            I1 = dat.transpose(0, 1, 4, 2, 3)[:, t, 0]
            I2 = dat.transpose(0, 1, 4, 2, 3)[:, t + 1, 0]
            flow = varflow_factory.batch_calc_flow(I1=I1, I2=I2)[:, :, :, :]
            cur_flow = np.concatenate((flow[:,:1, :, :], -flow[:,1:, :, :]), axis=1)
            var_flows.append(cur_flow)

        nearest_preds = []
        wrap_preds = []
        for t in range(14):
            cur_img = dat.transpose(0, 1, 4, 2, 3)[:, t,]
            cur_flow = var_flows[t]
            next_img = nearest_neighbor_advection(cur_img,-cur_flow)
            nearest_preds.append(next_img)
            cur_img = Variable(torch.from_numpy(cur_img).float().cuda())
            cur_flow = Variable(torch.from_numpy(cur_flow).float().cuda())
            torch_next_img = wrap(cur_img,-cur_flow).data.cpu().numpy()
            wrap_preds.append(torch_next_img)

        fig = plt.figure(figsize=(15, 4))
        gs = GridSpec(4, 15)
        for t in range(15):
            ax = plt.subplot2grid((4, 15), (0, t))
            ax.set_xticks([])
            ax.set_yticks([])
            cur_dat = dat[0,t,:,:,0]
            cur_dat = cv2.pyrDown(cur_dat,0.5)
            print(cur_dat.shape)
            ax.imshow(mapping(pixel_to_dBZ(cur_dat)))
            if t==0:
                continue
            ax = plt.subplot2grid((4, 15), (1, t-1))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(mapping(pixel_to_dBZ(nearest_preds[t-1][0,0])))
            ax = plt.subplot2grid((4, 15), (2, t-1))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(mapping(pixel_to_dBZ(wrap_preds[t-1][0, 0])))
            flow_img = flow_to_image(var_flows[t-1][0].transpose((1,2,0)))

            ax = plt.subplot2grid((4, 15), (3, t-1))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(flow_img)
        plt.savefig('flow_example.png')
        break
