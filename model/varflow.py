
import os
import sys
import ctypes
import numpy as np
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
