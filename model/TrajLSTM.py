import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.ConvRNN import *
from torch.nn.parameter import Parameter
from torch.autograd import Variable


class TrajLSTM(ConvRNN):
    def __init__(self,
                 cell_param,
                 return_state,
                 return_sequence):
        super(TrajLSTM, self).__init__(
            cell_param,
            return_state,
            return_sequence
        )

        self.build()

    def build(self):
        self.cell = TrajLSTMCell(self.cell_param)

    def init_hidden(self, input):

        hidden_size = (input.size()[0], self.cell_param['output_channels'], input.size()[-2], input.size()[-1])
        h = Variable(self.init_paramter(hidden_size))
        c = Variable(self.init_paramter(hidden_size))
        state = (h, c)
        return state

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

class TrajLSTMCell(ConvRNNCell):
    def __init__(self, cell_param):
        super(TrajLSTMCell, self).__init__(cell_param)
        self.L = cell_param['L']
        self.build_model()

    def get_parameter(self, shape, init_method='xavier'):
        param = Parameter(torch.Tensor(*shape).cuda())
        if init_method == 'xavier':
            nn.init.xavier_uniform_(param)
        elif init_method == 'zero':
            nn.init.constant_(param, 0)
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
            self.L * 2,
            32,
            5,
            5
        ]
        flow_bias_shape = [
            1, 32, 1, 1
        ]
        flow_flow_bias_shape = [
            1, self.L * 2, 1, 1
        ]
        ret_bias_shape = [
            1, 4 * self.output_dim, 1, 1
        ]
        ret_shape = [
            self.output_dim * 4,
            self.output_dim * self.L,
            1,
            1
        ]
        self.w_if = self.get_parameter(input_to_flow_shape)
        self.w_hflow = self.get_parameter(state_to_flow_shape)
        self.w_ff = self.get_parameter(flow_to_flow_shape)
        self.b_if = self.get_parameter(flow_bias_shape, 'zero')
        self.b_hf = self.get_parameter(flow_bias_shape, 'zero')
        self.b_ff = self.get_parameter(flow_flow_bias_shape, 'zero')
        self.w_ret = self.get_parameter(ret_shape)
        self.b_ret = self.get_parameter(ret_bias_shape, 'zero')

        self.w_hi = Parameter(torch.Tensor(self.output_dim, self.output_dim, self.state_to_state_kernel_size[0],
                                           self.state_to_state_kernel_size[1]).cuda())
        self.w_hf = Parameter(torch.Tensor(self.output_dim, self.output_dim, self.state_to_state_kernel_size[0],
                                           self.state_to_state_kernel_size[1]).cuda())
        self.w_hc = Parameter(torch.Tensor(self.output_dim, self.output_dim, self.state_to_state_kernel_size[0],
                                           self.state_to_state_kernel_size[1]).cuda())
        self.w_ho = Parameter(torch.Tensor(self.output_dim, self.output_dim, self.state_to_state_kernel_size[0],
                                           self.state_to_state_kernel_size[1]).cuda())
        self.weight_hs = [self.w_hi, self.w_hf, self.w_hc, self.w_ho]

        self.w_ci = Parameter(torch.Tensor(self.output_dim).cuda())
        self.w_cf = Parameter(torch.Tensor(self.output_dim).cuda())
        self.w_co = Parameter(torch.Tensor(self.output_dim).cuda())
        self.weight_cs = [self.w_ci, self.w_cf, self.w_co]

        self.b_i = Parameter(torch.Tensor(self.output_dim).cuda())
        self.b_f = Parameter(torch.Tensor(self.output_dim).cuda())
        self.b_o = Parameter(torch.Tensor(self.output_dim).cuda())
        self.b_c = Parameter(torch.Tensor(self.output_dim).cuda())
        self.biases = [self.b_i, self.b_f, self.b_c, self.b_o]

        self.w_xi = Parameter(torch.Tensor(self.output_dim, self.input_dim, self.input_to_state_kernel_size[0],
                                           self.input_to_state_kernel_size[1]).cuda())
        self.w_xf = Parameter(torch.Tensor(self.output_dim, self.input_dim, self.input_to_state_kernel_size[0],
                                           self.input_to_state_kernel_size[1]).cuda())
        self.w_xo = Parameter(torch.Tensor(self.output_dim, self.input_dim, self.input_to_state_kernel_size[0],
                                           self.input_to_state_kernel_size[1]).cuda())
        self.w_xc = Parameter(torch.Tensor(self.output_dim, self.input_dim, self.input_to_state_kernel_size[0],
                                           self.input_to_state_kernel_size[1]).cuda())
        self.weights_is = [self.w_xi, self.w_xf, self.w_xc, self.w_xo]

        self.reset_parameters()

    def same_padding(self,kernel_size):
        if kernel_size[0]%2==0 or kernel_size[1]%2==0:
            raise('The kernel size must to be odd if you want padding!')
        else:
            padding = tuple((int((kernel_size[0]-1)/2),int((kernel_size[1]-1)/2)))
        return padding

    def reset_parameters(self):

        # n = self.output_dim
        # stdv1 = 1. / math.sqrt(n)
        for weight_h in self.weight_hs:
            # weight_h.data.uniform_(-stdv1, stdv1)
            torch.nn.init.xavier_uniform_(weight_h)

        for bias_i in self.biases:
            # bias_i.data.zero_()
            torch.nn.init.constant_(bias_i, 0)

        for weight_c in self.weight_cs:
            # weight_c.data.zero_()
            torch.nn.init.constant_(weight_c, 0)

        # n = self.input_dim
        # stdv1 = 1. / math.sqrt(n)
        for weight_i in self.weights_is:
            # weight_i.data.uniform_(-stdv1, stdv1)
            torch.nn.init.xavier_uniform_(weight_i)

    def flow_generator(self,input,state):

        h1 = F.leaky_relu(
            F.conv2d(input,self.w_if,bias = None,padding = 2)+self.b_if+
            F.conv2d(state,self.w_hflow,bias = None,padding = 2)+self.b_hf
        )
        flow = F.conv2d(h1,self.w_ff,bias = None,padding = 2)+self.b_ff
        flows = torch.split(flow, 2, dim=1)
        return flows

    def cell(self, x_t, hidden):

        h_tm1, c_tm1 = hidden

        flows = self.flow_generator(x_t, h_tm1)
        wrapped_data = []
        for j in range(len(flows)):
            flow = flows[j]
            wrapped_data.append(wrap(h_tm1, -flow))
        wrapped_data = torch.cat(wrapped_data, dim=1)

        h2h = F.conv2d(wrapped_data, self.w_ret, bias=None, padding=0) + self.b_ret

        h2h_slice = torch.split(h2h, self.output_dim, dim=1)



        input_gate = torch.sigmoid(
            F.conv2d(x_t, self.w_xi, bias=None, padding=self.same_padding(self.input_to_state_kernel_size))
            + F.conv2d(h2h_slice[0], self.w_hi, bias=None, padding=self.same_padding(self.state_to_state_kernel_size))
            # + c_tm1 * self.w_ci.view(1, self.w_ci.size()[0], 1, 1)
            + self.b_i.view(1, self.b_i.size()[0], 1, 1)
        )

        forget_gate = torch.sigmoid(
            F.conv2d(x_t, self.w_xf, bias=None, padding=self.same_padding(self.input_to_state_kernel_size))
            + F.conv2d(h2h_slice[1], self.w_hf, bias=None, padding=self.same_padding(self.state_to_state_kernel_size))
            # + c_tm1 * self.w_cf.view(1, self.w_cf.size()[0], 1, 1)
            + self.b_f.view(1, self.b_f.size()[0], 1, 1)
        )

        c_t = forget_gate * c_tm1 + \
              input_gate * torch.tanh(
            F.conv2d(x_t, self.w_xc, bias=None, padding=self.same_padding(self.input_to_state_kernel_size))
            + F.conv2d(h2h_slice[2], self.w_hc, bias=None, padding=self.same_padding(self.state_to_state_kernel_size))
            + self.b_c.view(1, self.b_c.size()[0], 1, 1)
        )

        output_gate = torch.sigmoid(
            F.conv2d(x_t, self.w_xo, bias=None, padding=self.same_padding(self.input_to_state_kernel_size))
            + F.conv2d(h2h_slice[3], self.w_ho, bias=None, padding=self.same_padding(self.state_to_state_kernel_size))
            # + c_t * self.w_co.view(1, self.w_co.size()[0], 1, 1)
            + self.b_o.view(1, self.b_o.size()[0], 1, 1)
        )

        h_t = output_gate * torch.tanh(c_t)

        return h_t, c_t

    def forward(self, input, hidden):
        h_t,c_t = self.cell(input,hidden)
        return h_t,c_t





if __name__ == '__main__':
    pass