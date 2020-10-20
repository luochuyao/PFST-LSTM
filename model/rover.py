import numpy as np
from scipy.interpolate import NearestNDInterpolator
import os
from VarFlow.varflow import VarFlowFactory

flow_factory = VarFlowFactory(
        max_level=6,
        start_level=0,
        n1=2,
        n2=2,
        rho=1.5,
        alpha=2000,
        sigma=4.5)

from data.CIKM.data_iterator import *

# def nd_advection(im, flow):
#     """
#
#     Parameters
#     ----------
#     im : nd.NDArray
#         Shape: (batch_size, C, H, W)
#     flow : nd.NDArray
#         Shape: (batch_size, 2, H, W)
#     Returns
#     -------
#     new_im : nd.NDArray
#     """
#     grid = nd.GridGenerator(-flow, transform_type="warp")
#     new_im = nd.BilinearSampler(im, grid)
#     return new_im
#
#
# def nearest_neighbor_advection(im, flow):
#     """
#
#     Parameters
#     ----------
#     im : np.ndarray
#         Shape: (batch_size, C, H, W)
#     flow : np.ndarray
#         Shape: (batch_size, 2, H, W)
#     Returns
#     -------
#     new_im : nd.NDArray
#     """
#     predict_frame = np.empty(im.shape, dtype=im.dtype)
#     batch_size, channel_num, height, width = im.shape
#     assert channel_num == 1
#     grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
#     interp_grid = np.hstack([grid_x.reshape((-1, 1)), grid_y.reshape((-1, 1))])
#     for i in range(batch_size):
#         flow_interpolator = NearestNDInterpolator(interp_grid, im[i].ravel())
#         predict_grid = interp_grid + np.hstack([flow[i][0].reshape((-1, 1)),
#                                                 flow[i][1].reshape((-1, 1))])
#         predict_frame[i, 0, ...] = flow_interpolator(predict_grid).reshape((height, width))
#     return predict_frame

def get_flow_sequence(frame_dat):
    sequence = frame_dat.shape[0]
    flow_x = []
    flow_y = []
    for t in range(sequence-1):
        img1 = frame_dat[t,:,:,:,0]
        img2 = frame_dat[t+1,:,:,:,0]
        flow = flow_factory.batch_calc_flow(I1=img1, I2=img2)[:,:,:,:,np.newaxis]
        flow_x.append(flow[:,0,:,:,:,])
        flow_y.append(flow[:,1,:,:,:,])

    flow_x = np.stack(flow_x, 0)
    flow_y = np.stack(flow_y, 0)
    flow = [flow_x,flow_x]
    return flow

def run_test():
    counter = 0
    frame_dat = sample(
        batch_size=8
    )
    frame_dat = frame_dat.transpose(1,0,2,3,4)
    in_frame_dat = frame_dat[:5]
    output_frame_dat = frame_dat[5:]
    print(in_frame_dat.shape)
    flow = get_flow_sequence(in_frame_dat)
    # print(flow[0].shape,flow[1].shape)
    # print(in_frame_dat.shape)



if __name__ == '__main__':
    run_test()
