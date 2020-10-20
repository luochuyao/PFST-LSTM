import ctypes
import os
import numpy as np
from datetime import datetime
from scipy.misc import imread,imsave
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]

import sys
sys.path.append(rootPath)
sys.path.append(os.path.split(rootPath)[0])
import cv2
import shutil

from concurrent.futures import ThreadPoolExecutor, wait

_BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))

_VALID_DLL_PATH = [os.path.join(_BASE_PATH, '..', 'build', 'Release', 'varflow.dll'),
                   os.path.join(_BASE_PATH, '..', 'build', 'libvarflow.so')]

_VARFLOW_DLL_PATH = None
for p in _VALID_DLL_PATH:
    if os.path.exists(p):
        _VARFLOW_DLL_PATH = p
        break
if _VARFLOW_DLL_PATH is None:
    raise RuntimeError("DLL not found! Valid PATH=%s" %(_VALID_DLL_PATH))
_CDLL = ctypes.cdll.LoadLibrary(_VARFLOW_DLL_PATH)

def clean_fold(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)
class VarFlowFactory(object):
    def __init__(self, max_level, start_level, n1, n2, rho, alpha, sigma):
        self._max_level = max_level
        self._start_level = start_level
        self._n1 = n1
        self._n2 = n2
        self._rho = rho
        self._alpha = alpha
        self._sigma = sigma
        self._varflow_executor_pool = ThreadPoolExecutor(max_workers=16)

    def calc_flow(self, I1, I2):
        """

        Parameters
        ----------
        I1 : np.ndarray
            Shape: (H, W)
        I2 : np.ndarray
            Shape: (H, W)
        Returns
        -------
        velocity : np.ndarray
            Shape: (2, H, W)
            The channel dimension will be flow_x, flow_y
        """
        if I1.dtype == np.float32:
            I1 = (I1 * 255).astype(np.uint8)
        else:
            I1 = I1.astype(np.uint8)
        if I2.dtype == np.float32:
            I2 = (I2 * 255).astype(np.uint8)
        else:
            I2 = I2.astype(np.uint8)
        assert I1.ndim == 2 and I2.ndim == 2
        assert I1.shape == I2.shape
        np.ascontiguousarray(I1)
        np.ascontiguousarray(I2)
        height, width = I1.shape
        velocity = np.zeros((2,) + I1.shape, dtype=np.float32)
        self._base_varflow_call(velocity=velocity, I1=I1, I2=I2, width=width, height=height)
        return velocity

    def _base_varflow_call(self, velocity, I1, I2, width, height):
        _CDLL.varflow(ctypes.c_int32(width),
                      ctypes.c_int32(height),
                      ctypes.c_int32(self._max_level),
                      ctypes.c_int32(self._start_level),
                      ctypes.c_int32(self._n1),
                      ctypes.c_int32(self._n2),
                      ctypes.c_float(self._rho),
                      ctypes.c_float(self._alpha),
                      ctypes.c_float(self._sigma),
                      velocity[0].ctypes.data_as(ctypes.c_void_p),
                      velocity[1].ctypes.data_as(ctypes.c_void_p),
                      I1.ctypes.data_as(ctypes.c_void_p),
                      I2.ctypes.data_as(ctypes.c_void_p))

    def batch_calc_flow(self, I1, I2):
        """Calculate the optical flow from two

        Parameters
        ----------
        I1 : np.ndarray
            Shape: (batch_size, H, W)
        I2 : np.ndarray
            Shape: (batch_size, H, W)
        Returns
        -------
        velocity : np.ndarray
            Shape: (batch_size, 2, H, W)
            The channel dimension will be flow_x, flow_y
        """
        if I1.dtype == np.float32:
            I1 = (I1 * 255).astype(np.uint8)
        else:
            I1 = I1.astype(np.uint8)
        if I2.dtype == np.float32:
            I2 = (I2 * 255).astype(np.uint8)
        else:
            I2 = I2.astype(np.uint8)
        np.ascontiguousarray(I1)
        np.ascontiguousarray(I2)
        assert I1.ndim == 3 and I2.ndim == 3
        assert I1.shape == I2.shape
        batch_size, height, width = I1.shape
        velocity = np.zeros((batch_size, 2, height, width), dtype=np.float32)
        future_objs = []
        for i in range(batch_size):
            obj = self._varflow_executor_pool.submit(
                self._base_varflow_call, velocity[i], I1[i], I2[i], width, height)
            future_objs.append(obj)
        wait(future_objs)
        return velocity

def pre_figure(first_imgs,second_imgs,flows):
    batch_size = first_imgs.shape[0]
    new_frames = []

    for batch_idx in range(batch_size):
        first_img = first_imgs[batch_idx]
        second_img = second_imgs[batch_idx]
        flow = flows[batch_idx]
        flow = flow.transpose((1, 2, 0))
        w = int(first_img.shape[1])
        h = int(second_img.shape[0])
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        coords = np.float32(np.dstack([x_coords, y_coords]))
        pixel_map = coords + flow
        new_frame = cv2.remap(second_img, pixel_map, None, 1)
        new_frames.append(new_frame)
    return np.array(new_frames)

