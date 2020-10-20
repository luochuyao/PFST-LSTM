import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.ConvRNN import *
from torch.nn.parameter import Parameter
from torch.autograd import Variable


class ConvGRU(ConvRNN):
    def __init__(self,
                 cell_param,
                 return_state,
                 return_sequence):
        super(ConvGRU, self).__init__(
            cell_param,
            return_state,
            return_sequence
        )

        self.build()

    def build(self):
        self.cell = ConvGRUCell(self.cell_param)

    def init_hidden(self, input):

        hidden_size = (input.size()[0], self.cell_param['output_channels'], input.size()[-2], input.size()[-1])
        h = Variable(self.init_paramter(hidden_size))
        state = h

        return state



class ConvGRUCell(ConvRNNCell):
    def __init__(self, cell_param):
        super(ConvGRUCell, self).__init__(cell_param)
        self.build_model()

    def get_parameter(self,shape,init_method = 'xavier'):
        param = Parameter(torch.Tensor(*shape).cuda())
        if init_method == 'xavier':
            nn.init.xavier_uniform_(param)
        elif init_method == 'zero':
            nn.init.constant_(param,0)
        else:
            raise ('init method error')
        return param

    def build_model(self):

        input_to_state_shape = [
            self.output_dim,
            self.input_dim,
            self.input_to_state_kernel_size[0],
            self.input_to_state_kernel_size[1]
        ]
        state_to_state_shape = [
            self.output_dim,
            self.output_dim,
            self.state_to_state_kernel_size[0],
            self.state_to_state_kernel_size[1]
        ]
        state_bias_shape = [
            1, self.output_dim, 1, 1
        ]

        self.w_xz = self.get_parameter(input_to_state_shape)
        self.w_hz = self.get_parameter(state_to_state_shape)
        self.w_xr = self.get_parameter(input_to_state_shape)
        self.w_hr = self.get_parameter(state_to_state_shape)
        self.w_xh = self.get_parameter(input_to_state_shape)
        self.w_hh = self.get_parameter(state_to_state_shape)


        self.b_z = self.get_parameter(state_bias_shape,'zero')
        self.b_r = self.get_parameter(state_bias_shape,'zero')
        self.b_h_ = self.get_parameter(state_bias_shape,'zero')

    def same_padding(self,kernel_size):
        if kernel_size[0]%2==0 or kernel_size[1]%2==0:
            raise('The kernel size must to be odd if you want padding!')
        else:
            padding = tuple((int((kernel_size[0]-1)/2),int((kernel_size[1]-1)/2)))
        return padding


    def cell(self, x_t, hidden):
        h_tm1 = hidden
        Z = torch.sigmoid(
            F.conv2d(x_t, self.w_xz, bias=None, padding=self.same_padding(self.input_to_state_kernel_size))
            + F.conv2d(h_tm1, self.w_hz, bias=None, padding=self.same_padding(self.state_to_state_kernel_size))
            + self.b_z
        )

        R = torch.sigmoid(
            F.conv2d(x_t, self.w_xr, bias=None, padding=self.same_padding(self.input_to_state_kernel_size))
            + F.conv2d(h_tm1, self.w_hr, bias=None, padding=self.same_padding(self.state_to_state_kernel_size))
            + self.b_r
        )

        H_ = F.leaky_relu(
            F.conv2d(x_t,self.w_xh,bias = None,padding = self.same_padding(self.input_to_state_kernel_size))
            + R*F.conv2d(h_tm1,self.w_hh,bias=None,padding = self.same_padding(self.state_to_state_kernel_size))
            + self.b_h_,negative_slope = 0.2
        )

        H = (1-Z)*H_ + Z*h_tm1

        return H

    def forward(self, input, hidden):
        h_t = self.cell(input,hidden)
        return h_t





if __name__ == '__main__':
    pass