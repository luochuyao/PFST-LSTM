import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.ConvRNN import *
from torch.nn.parameter import Parameter
from torch.autograd import Variable

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

class TrajGRU(ConvRNN):
    def __init__(self,
                 cell_param,
                 return_state,
                 return_sequence):
        super(TrajGRU, self).__init__(
            cell_param,
            return_state,
            return_sequence
        )

        self.build()

    def build(self):
        self.cell = TrajGRUCell(self.cell_param)

    def init_hidden(self, input):

        hidden_size = (input.size()[0], self.cell_param['output_channels'], input.size()[-2], input.size()[-1])
        h = Variable(self.init_paramter(hidden_size))
        state = h

        return state



class TrajGRUCell(ConvRNNCell):
    def __init__(self, cell_param):
        super(TrajGRUCell, self).__init__(cell_param)
        self.L = cell_param['L']
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
        state_to_flow_shape = [
            32,
            self.output_dim,
            5,
            5
        ]
        input_to_flow_shape = [
            32,
            self.input_dim,
            5,
            5
        ]
        flow_to_flow_shape = [
            self.L*2,
            32,
            5,
            5
        ]
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
        ret_shape = [
            self.output_dim*3,
            self.output_dim*self.L,
            1,
            1
        ]
        state_bias_shape = [
            1, self.output_dim, 1, 1
        ]
        flow_bias_shape = [
            1,32,1,1
        ]
        flow_flow_bias_shape=[
            1,self.L * 2,1,1
        ]
        ret_bias_shape = [
            1,3*self.output_dim,1,1
        ]

        self.w_if = self.get_parameter(input_to_flow_shape)
        self.w_hf = self.get_parameter(state_to_flow_shape)
        self.w_ff = self.get_parameter(flow_to_flow_shape)
        self.b_if = self.get_parameter(flow_bias_shape,'zero')
        self.b_hf = self.get_parameter(flow_bias_shape, 'zero')
        self.b_ff = self.get_parameter(flow_flow_bias_shape, 'zero')
        self.w_ret = self.get_parameter(ret_shape)
        self.b_ret = self.get_parameter(ret_bias_shape,'zero')

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

    def flow_generator(self,input,state):
        h1 = F.leaky_relu(
            F.conv2d(input,self.w_if,bias = None,padding = 2)+self.b_if+
            F.conv2d(state,self.w_hf,bias = None,padding = 2)+self.b_hf
        )
        flow = F.conv2d(h1,self.w_ff,bias = None,padding = 2)+self.b_ff
        flows = torch.split(flow, 2, dim=1)
        return flows

    def cell(self, x_t, hidden):
        flows = self.flow_generator(x_t,hidden)
        wrapped_data = []
        for j in range(len(flows)):
            flow = flows[j]
            wrapped_data.append(wrap(hidden,-flow))
        wrapped_data = torch.cat(wrapped_data,dim=1)
        h2h = F.conv2d(wrapped_data,self.w_ret,bias=None,padding = 0)+self.b_ret

        h2h_slice = torch.split(h2h,self.output_dim,dim=1)

        Z = torch.sigmoid(
            F.conv2d(x_t, self.w_xz, bias=None, padding=self.same_padding(self.input_to_state_kernel_size))
            + F.conv2d(h2h_slice[0], self.w_hz, bias=None, padding=self.same_padding(self.state_to_state_kernel_size))
            + self.b_z
        )

        R = torch.sigmoid(
            F.conv2d(x_t, self.w_xr, bias=None, padding=self.same_padding(self.input_to_state_kernel_size))
            + F.conv2d(h2h_slice[1], self.w_hr, bias=None, padding=self.same_padding(self.state_to_state_kernel_size))
            + self.b_r
        )

        H_ = F.leaky_relu(
            F.conv2d(x_t,self.w_xh,bias = None,padding = self.same_padding(self.input_to_state_kernel_size))
            + R*F.conv2d(h2h_slice[2],self.w_hh,bias=None,padding = self.same_padding(self.state_to_state_kernel_size))
            + self.b_h_,negative_slope = 0.2
        )

        H = (1-Z)*H_ + Z*hidden

        return H

    def forward(self, input, hidden):
        h_t = self.cell(input,hidden)
        return h_t


if __name__ == '__main__':
    pass