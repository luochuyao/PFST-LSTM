import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from model.ConvRNN import *
from torch.nn.parameter import Parameter
from torch.autograd import Variable






class PFST_ConvLSTMCell(ConvRNNCell):
    def __init__(self, cell_param):
        super(PFST_ConvLSTMCell, self).__init__(cell_param)

        self.m_dim = cell_param['m_channels']
        self.input_to_input_kernel_size = cell_param['input_to_input_kernel_size']
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

    # input: B, C, H, W
    # flow: [B, 2, H, W]
    def wrap(self,input, flow):

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
        input_to_m_shape = [
            self.m_dim,
            self.input_dim,
            self.input_to_input_kernel_size[0],
            self.input_to_input_kernel_size[1]
        ]
        m_to_m_shape = [
            self.m_dim,
            self.m_dim,
            self.input_to_input_kernel_size[0],
            self.input_to_input_kernel_size[1]
        ]
        m_to_state_shape = [
            self.output_dim,
            self.m_dim,
            self.input_to_input_kernel_size[0],
            self.input_to_input_kernel_size[1]
        ]
        state_bias_shape = [
            1,self.output_dim,1,1
        ]
        input_bias_shape = [
            1,self.input_dim,1,1
        ]
        m_bias_shape = [
            1,self.m_dim,1,1
        ]

        self.w_xg = self.get_parameter(input_to_state_shape)
        self.w_hg = self.get_parameter(state_to_state_shape)
        self.w_xi = self.get_parameter(input_to_state_shape)
        self.w_hi = self.get_parameter(state_to_state_shape)
        self.w_xf = self.get_parameter(input_to_state_shape)
        self.w_hf = self.get_parameter(state_to_state_shape)

        # self.weight_hs = [self.w_hg,self.w_hi,self.w_hf]

        self.w_xg_ = self.get_parameter(input_to_m_shape)
        self.w_mg = self.get_parameter(m_to_m_shape)
        self.w_xi_ = self.get_parameter(input_to_m_shape)
        self.w_mi = self.get_parameter(m_to_m_shape)
        self.w_xf_ = self.get_parameter(input_to_m_shape)
        self.w_mf = self.get_parameter(m_to_m_shape)

        # self.weight_is = [self.w_xg,self.w_xg_,self.w_xi,self.w_xi_,self.w_xf,self.w_xf_]
        # self.weight_ms = [self.w_mg,self.w_mi,self.w_mf]

        self.w_xo = self.get_parameter(input_to_state_shape)
        self.w_ho = self.get_parameter(state_to_state_shape)
        self.w_co = self.get_parameter(state_to_state_shape)
        self.w_mo = self.get_parameter(m_to_state_shape)

        # self.weight_cs = [self.w_xo,self.w_ho,self.w_co,self.w_mo]



        self.w_1x1 = self.get_parameter([
            self.output_dim,self.output_dim+self.m_dim,1,1
        ])

        self.b_g = self.get_parameter(state_bias_shape,'zero')
        self.b_i = self.get_parameter(state_bias_shape,'zero')
        self.b_f = self.get_parameter(state_bias_shape,'zero')
        self.b_g_ = self.get_parameter(m_bias_shape,'zero')
        self.b_i_ = self.get_parameter(m_bias_shape,'zero')
        self.b_f_ = self.get_parameter(m_bias_shape,'zero')
        self.b_o = self.get_parameter(state_bias_shape,'zero')

        self.biases = [self.b_g,self.b_i,self.b_f,self.b_g_,self.b_i_,self.b_f_,self.b_o]

        self.w_xd = self.get_parameter([
            2,
            self.input_dim,
            3,
            3
        ])
        self.w_md = self.get_parameter([
            2,
            self.m_dim,
            3,
            3
        ])
        self.w_hd = self.get_parameter([
            2,
            self.output_dim,
            3,
            3
        ])
        self.w_x_t_d = self.get_parameter([
            2,
            self.input_dim,
            3,
            3
        ])
        self.b_d = self.get_parameter([1,2,1,1],'zero')

        self.w_i_h1 = self.get_parameter([
            (int)(self.output_dim / 2),
            2 * self.input_dim,
            5,
            5
        ])
        self.b_i_h1 = self.get_parameter(
            [
                1,(int)(self.output_dim / 2),1,1
            ],'zero'
        )
        self.w_i_h2 = self.get_parameter(
            [
                self.output_dim,
                (int)(self.output_dim / 2),
                5,
                5
            ]
        )
        self.b_i_h2 = self.get_parameter(
            [
                1,self.output_dim,1,1
            ],'zero'
        )

        self.w_h_h1 = self.get_parameter(
            [
                self.output_dim,
                2*self.output_dim,
                3,
                3
            ]
        )
        self.b_h_h1 = self.get_parameter([
            1,self.output_dim,1,1
        ],'zero')
        self.w_h_h2 = self.get_parameter(
            [
                self.output_dim,
                self.output_dim,
                3,
                3
            ]
        )
        self.b_h_h2 = self.get_parameter(
            [1,self.output_dim,1,1]
        )




        self.w_c_h1 = self.get_parameter(
            [
                self.output_dim,
                2 * self.output_dim,
                3,
                3
            ]
        )
        self.b_c_h1 = self.get_parameter([
            1, self.output_dim, 1, 1
        ], 'zero')
        self.w_c_h2 = self.get_parameter(
            [
                self.output_dim,
                self.output_dim,
                3,
                3
            ]
        )
        self.b_c_h2 = self.get_parameter(
            [1, self.output_dim, 1, 1]
        )

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
            torch.nn.init.xavier_uniform_(weight_h)
            # weight_h.data.uniform_(-stdv1, stdv1)

        for weight_c in self.weight_cs:
            torch.nn.init.xavier_uniform_(weight_c)
            # weight_c.data.uniform_(-stdv1, stdv1)

        for bias_i in self.biases:
            torch.nn.init.constant_(bias_i,0)
            # bias_i.data.zero_()

        # n = self.input_dim
        # stdv1 = 1. / math.sqrt(n)
        for weight_i in self.weight_is:
            torch.nn.init.xavier_uniform_(weight_i)
            # weight_i.data.uniform_(-stdv1, stdv1)

        # n = self.m_dim
        # stdv1 = 1. / math.sqrt(n)
        for weight_m in self.weight_ms:
            torch.nn.init.xavier_uniform_(weight_m)
            # weight_m.data.uniform_(-stdv1, stdv1)

    def pseudo_flow_generate(self,x_t_1,x_t):
        input = torch.cat([x_t_1,x_t],1)
        h1 = torch.relu(F.conv2d(input,self.w_i_h1,bias=None,padding = 3)+self.b_i_h1)
        flow = torch.tanh(F.conv2d(h1,self.w_i_h2,bias=None,padding = 1)+self.b_i_h2)
        return flow

    def update_h(self,hidden,flow):
        h1 = torch.tanh(F.conv2d(torch.cat([hidden,flow],1),self.w_h_h1,bias = None,padding=1)+self.b_h_h1)
        output = torch.tanh(F.conv2d(h1,self.w_h_h2,bias = None,padding=1)+self.b_h_h2)
        return output

    def update_c(self,hidden,flow):
        c1 = torch.relu(F.conv2d(torch.cat([hidden,flow],1),self.w_c_h1,bias = None,padding=1)+self.b_c_h1)
        output = torch.tanh(F.conv2d(c1,self.w_c_h2,bias = None,padding=1)+self.b_c_h2)
        return output

    # def d_cell(self,x_t,x_t_1, hidden,M):
    #
    #     h_tm1, c_tm1 = hidden
    #
    #     d_t_1 = F.conv2d(x_t, self.w_xd, bias=None, padding=1) + \
    #             F.conv2d(h_tm1, self.w_hd, bias=None, padding=1) + \
    #             F.conv2d(x_t_1, self.w_x_t_d, bias=None, padding=1) + \
    #             F.conv2d(M, self.w_md, bias=None, padding=1) + \
    #             self.b_d
    #
    #
    #     h_tm1 = self.wrap(h_tm1, d_t_1)
    #     c_tm1 = self.wrap(c_tm1, d_t_1)
    #
    #     g_x = F.conv2d(x_t, self.w_xg, bias=None, padding=self.same_padding(self.input_to_state_kernel_size))
    #     g_h = F.conv2d(h_tm1, self.w_hg, bias=None, padding=self.same_padding(self.state_to_state_kernel_size))
    #     g = torch.tanh(
    #         g_x+
    #         g_h+
    #         self.b_g
    #     )
    #     i = torch.sigmoid(
    #         F.conv2d(x_t, self.w_xi, bias=None, padding=self.same_padding(self.input_to_state_kernel_size)) +
    #         F.conv2d(h_tm1, self.w_hi, bias=None, padding=self.same_padding(self.state_to_state_kernel_size)) +
    #         self.b_i
    #     )
    #
    #     f = torch.sigmoid(
    #         F.conv2d(x_t, self.w_xf, bias=None, padding=self.same_padding(self.input_to_state_kernel_size)) +
    #         F.conv2d(h_tm1, self.w_hf, bias=None, padding=self.same_padding(self.state_to_state_kernel_size)) +
    #         self.b_f
    #     )
    #     c = f * c_tm1 + i * g
    #
    #     g_ = torch.tanh(
    #         F.conv2d(x_t, self.w_xg_, bias=None, padding=self.same_padding(self.input_to_input_kernel_size)) +
    #         F.conv2d(M, self.w_mg, bias=None, padding=self.same_padding(self.input_to_input_kernel_size)) +
    #         self.b_g_
    #     )
    #     i_ = torch.sigmoid(
    #         F.conv2d(x_t, self.w_xi_, bias=None, padding=self.same_padding(self.input_to_input_kernel_size)) +
    #         F.conv2d(M, self.w_mi, bias=None, padding=self.same_padding(self.input_to_input_kernel_size)) +
    #         self.b_i_
    #     )
    #     f_ = torch.sigmoid(
    #         F.conv2d(x_t, self.w_xf_, bias=None, padding=self.same_padding(self.input_to_input_kernel_size)) +
    #         F.conv2d(M, self.w_mf, bias=None, padding=self.same_padding(self.input_to_input_kernel_size)) +
    #         self.b_f_
    #     )
    #     M_ = f_ * M + i_ * g_
    #
    #     o = torch.sigmoid(
    #         F.conv2d(x_t, self.w_xo, bias=None, padding=self.same_padding(self.input_to_state_kernel_size)) +
    #         F.conv2d(M_, self.w_mo, bias=None, padding=self.same_padding(self.input_to_input_kernel_size)) +
    #         F.conv2d(c, self.w_co, bias=None, padding=self.same_padding(self.state_to_state_kernel_size)) +
    #         F.conv2d(h_tm1, self.w_ho, bias=None, padding=self.same_padding(self.state_to_state_kernel_size)) +
    #         self.b_o
    #     )
    #
    #     h = o * torch.tanh(
    #         F.conv2d(
    #             torch.cat([c, M_], 1), self.w_1x1, bias=None, padding=0
    #         )
    #     )
    #     return h, c, M_,-d_t_1
    #
    # def hx_cell(self,x_t,x_t_1, hidden,M):
    #
    #     h_tm1, c_tm1 = hidden
    #
    #     d_t_1 = F.conv2d(x_t, self.w_xd, bias=None, padding=1) + \
    #             F.conv2d(h_tm1, self.w_hd, bias=None, padding=1) + \
    #             F.conv2d(x_t_1, self.w_x_t_d, bias=None, padding=1) + \
    #             F.conv2d(M, self.w_md, bias=None, padding=1) + \
    #             self.b_d
    #
    #
    #     h_tm1 = self.wrap(h_tm1, d_t_1)
    #     c_tm1 = self.wrap(c_tm1, d_t_1)
    #
    #     g_x = F.conv2d(x_t, self.w_xg, bias=None, padding=self.same_padding(self.input_to_state_kernel_size))
    #     g_h = F.conv2d(h_tm1, self.w_hg, bias=None, padding=self.same_padding(self.state_to_state_kernel_size))
    #     g = torch.tanh(
    #         g_x+
    #         g_h+
    #         self.b_g
    #     )
    #     i = torch.sigmoid(
    #         F.conv2d(x_t, self.w_xi, bias=None, padding=self.same_padding(self.input_to_state_kernel_size)) +
    #         F.conv2d(h_tm1, self.w_hi, bias=None, padding=self.same_padding(self.state_to_state_kernel_size)) +
    #         self.b_i
    #     )
    #
    #     f = torch.sigmoid(
    #         F.conv2d(x_t, self.w_xf, bias=None, padding=self.same_padding(self.input_to_state_kernel_size)) +
    #         F.conv2d(h_tm1, self.w_hf, bias=None, padding=self.same_padding(self.state_to_state_kernel_size)) +
    #         self.b_f
    #     )
    #     c = f * c_tm1 + i * g
    #
    #     g_ = torch.tanh(
    #         F.conv2d(x_t, self.w_xg_, bias=None, padding=self.same_padding(self.input_to_input_kernel_size)) +
    #         F.conv2d(M, self.w_mg, bias=None, padding=self.same_padding(self.input_to_input_kernel_size)) +
    #         self.b_g_
    #     )
    #     i_ = torch.sigmoid(
    #         F.conv2d(x_t, self.w_xi_, bias=None, padding=self.same_padding(self.input_to_input_kernel_size)) +
    #         F.conv2d(M, self.w_mi, bias=None, padding=self.same_padding(self.input_to_input_kernel_size)) +
    #         self.b_i_
    #     )
    #     f_ = torch.sigmoid(
    #         F.conv2d(x_t, self.w_xf_, bias=None, padding=self.same_padding(self.input_to_input_kernel_size)) +
    #         F.conv2d(M, self.w_mf, bias=None, padding=self.same_padding(self.input_to_input_kernel_size)) +
    #         self.b_f_
    #     )
    #     M_ = f_ * M + i_ * g_
    #
    #     o = torch.sigmoid(
    #         F.conv2d(x_t, self.w_xo, bias=None, padding=self.same_padding(self.input_to_state_kernel_size)) +
    #         F.conv2d(M_, self.w_mo, bias=None, padding=self.same_padding(self.input_to_input_kernel_size)) +
    #         F.conv2d(c, self.w_co, bias=None, padding=self.same_padding(self.state_to_state_kernel_size)) +
    #         F.conv2d(h_tm1, self.w_ho, bias=None, padding=self.same_padding(self.state_to_state_kernel_size)) +
    #         self.b_o
    #     )
    #
    #     h = o * torch.tanh(
    #         F.conv2d(
    #             torch.cat([c, M_], 1), self.w_1x1, bias=None, padding=0
    #         )
    #     )
    #     return h, c, M_,g_x,g_h


    def cell(self, x_t_1,x_t, hidden,M):
        h_tm1, c_tm1 = hidden
        d_t_1 = F.conv2d(x_t,self.w_xd,bias = None,padding = 1)+\
                F.conv2d(h_tm1,self.w_hd,bias = None,padding = 1)+\
                F.conv2d(x_t_1,self.w_x_t_d,bias=None,padding = 1)+\
                F.conv2d(M,self.w_md,bias=None,padding=1)+\
                self.b_d
        
        # d_t_1 = F.conv2d(x_t, self.w_xd, bias=None, padding=1) + \
        #         F.conv2d(h_tm1, self.w_hd, bias=None, padding=1) + \
        #         self.b_d
        # d_t_1 = F.conv2d(x_t, self.w_xd, bias=None, padding=1) + \
        #         F.conv2d(x_t_1, self.w_x_t_d, bias=None, padding=1) + \
        #         self.b_d
        h_tm1 = self.wrap(h_tm1,d_t_1)
        c_tm1 = self.wrap(c_tm1,d_t_1)
        # flow = self.pseudo_flow_generate(x_t_1,x_t)
        # h_tm1 = self.update_h(h_tm1,flow)
        # c_tm1 = self.update_c(c_tm1,flow)

        g = torch.tanh(
            F.conv2d(x_t,self.w_xg,bias=None, padding=self.same_padding(self.input_to_state_kernel_size))+
            F.conv2d(h_tm1,self.w_hg,bias=None, padding=self.same_padding(self.state_to_state_kernel_size))+
            self.b_g
        )
        i = torch.sigmoid(
            F.conv2d(x_t,self.w_xi,bias=None, padding=self.same_padding(self.input_to_state_kernel_size))+
            F.conv2d(h_tm1,self.w_hi,bias=None, padding=self.same_padding(self.state_to_state_kernel_size))+
            self.b_i
        )

        f = torch.sigmoid(
            F.conv2d(x_t,self.w_xf,bias=None, padding=self.same_padding(self.input_to_state_kernel_size))+
            F.conv2d(h_tm1,self.w_hf,bias=None, padding=self.same_padding(self.state_to_state_kernel_size))+
            self.b_f
        )
        c = f*c_tm1 + i*g

        g_ = torch.tanh(
            F.conv2d(x_t,self.w_xg_,bias=None,padding=self.same_padding(self.input_to_input_kernel_size))+
            F.conv2d(M,self.w_mg,bias=None,padding=self.same_padding(self.input_to_input_kernel_size))+
            self.b_g_
        )
        i_ = torch.sigmoid(
           F.conv2d(x_t,self.w_xi_,bias=None,padding=self.same_padding(self.input_to_input_kernel_size))+
           F.conv2d(M,self.w_mi,bias=None,padding=self.same_padding(self.input_to_input_kernel_size))+
           self.b_i_
        )
        f_ = torch.sigmoid(
            F.conv2d(x_t,self.w_xf_,bias=None,padding=self.same_padding(self.input_to_input_kernel_size))+
            F.conv2d(M,self.w_mf,bias=None,padding=self.same_padding(self.input_to_input_kernel_size))+
            self.b_f_
        )
        M_= f_*M + i_*g_

        o = torch.sigmoid(
            F.conv2d(x_t, self.w_xo, bias=None, padding=self.same_padding(self.input_to_state_kernel_size))+
            F.conv2d(M_, self.w_mo, bias=None, padding=self.same_padding(self.input_to_input_kernel_size))+
            F.conv2d(c,self.w_co,bias=None, padding=self.same_padding(self.state_to_state_kernel_size))+
            F.conv2d(h_tm1, self.w_ho, bias=None, padding=self.same_padding(self.state_to_state_kernel_size))+
            self.b_o
        )

        h = o*torch.tanh(
            F.conv2d(
                torch.cat([c,M_],1),self.w_1x1,bias = None,padding = 0
            )
        )

        return h,c,M_

    def forward(self, input,last_input, hidden, M):
        h_t,c_t,M = self.cell(last_input,input,hidden,M)
        return h_t,c_t,M





if __name__ == '__main__':
    pass