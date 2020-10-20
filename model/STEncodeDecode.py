import numpy as np
import torch
import torch.nn as nn
from model.ConvLSTM import *
from model.ST_ConvLSTM import *
import model.Conv as conv
import model.ConvSeq as conv_seq

import numpy as np
import torch
import torch.nn as nn
from model.ConvLSTM import *
from model.ST_ConvLSTM import *
import model.Conv as conv
import model.ConvSeq as conv_seq

class Decoder_ST_ConvLSTM(nn.Module):
    def __init__(
            self,
            conv_rnn_cells,
            conv_cells,
            conv_m_cells,
            conv_reset_m_cells,
            output_cells,
            info,
    ):
        super(Decoder_ST_ConvLSTM, self).__init__()
        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.conv_m_cells = conv_m_cells
        self.conv_reset_m_cells = conv_reset_m_cells
        self.output_cells = output_cells
        self.info =info
        self.models = []
        self.layer_num = len(self.conv_rnn_cells)
        self.m_upsample = []
        self.reset_m = []
        self.H = {}
        self.C = {}
        self.M = {}
        self.HH = {}

        for idx in range(len(self.conv_m_cells)):
            self.m_upsample.append(
                conv.DeConv2D(
                    cell_param=self.conv_m_cells[idx]
                ).cuda()
            )
            self.reset_m.append(
                conv.Conv2D(
                    cell_param=self.conv_reset_m_cells[idx]
                ).cuda()
            )
        # self.m_upsample = nn.ModuleList(self.m_upsample)
        # self.reset_m = nn.ModuleList(self.reset_m)

        for idx in range(self.layer_num):
            self.models.append(
                ST_ConvLSTMCell(
                    cell_param=self.conv_rnn_cells[idx],
                )
            )
            self.models.append(
                conv.DeConv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )
        self.models = nn.ModuleList(self.models)
        self.out_models = []
        for output_cell in self.output_cells:
            self.out_models.append(
                conv_seq.Conv2D(
                    cell_param=output_cell
                ).cuda()
            )
        self.out_models = nn.ModuleList(self.out_models)

    def forward(self,input,states,M):
        self.n_step = input.size()[1]
        output = []
        for t in range(self.n_step):
            x_t = input[:, t, :, :, :, ]
            if t == 0:
                for layer_idx in range(self.layer_num):
                    self.H[self.layer_num-1-layer_idx] = states[0][layer_idx]
                    self.C[self.layer_num-1-layer_idx] = states[1][layer_idx]
                    self.M[self.layer_num-1-layer_idx] = M[layer_idx]
                self.cell(x_t, True)

            else:
                self.cell(x_t, False)
            out = self.current_HH
            output.append(out)
        output = torch.stack(output,1)
        for out_model in self.out_models:
            output = out_model(output)

        return output

    @property
    def current_HH(self):
        return self.HH

    @property
    def current_states(self):
        return (self.H, self.C)

    @property
    def current_M(self):
        return self.M

    def cell_reset_m(self,M):
        for idx in range(len(self.conv_m_cells)):
            M = self.reset_m[idx](M)
        return M

    def cell(self,input,first_timestep = False):
        if first_timestep:
            for layer_idx in range(self.layer_num):
                if layer_idx == 0:
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2](input,(self.H[layer_idx],self.C[layer_idx]),self.M[layer_idx])
                else:
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2](self.models[2*layer_idx-1](self.H[layer_idx-1]),(self.H[layer_idx],self.C[layer_idx]),self.m_upsample[layer_idx-1](self.M[layer_idx-1]))

        else:
            for layer_idx in range(self.layer_num):
                if layer_idx == 0:
                    # input = self.models[layer_idx * 2](input)
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2](input, (
                    self.H[layer_idx], self.C[layer_idx]), self.cell_reset_m(self.M[self.layer_num - 1]))
                    continue
                else:
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2](self.models[layer_idx*2-1](self.H[layer_idx-1]),(
                    self.H[layer_idx], self.C[layer_idx]),self.m_upsample[layer_idx-1](self.M[layer_idx-1]))
                    continue

        self.HH = self.models[self.layer_num * 2-1](self.H[self.layer_num-1])
        return self.H,self.C,self.M






class Encoder_ST_ConvLSTM(nn.Module):
    def __init__(
            self,
            conv_rnn_cells,
            conv_cells,
            conv_m_cells,
            conv_reset_m_cells,
            info,
                 ):
        super(Encoder_ST_ConvLSTM, self).__init__()
        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.conv_m_cells = conv_m_cells
        self.conv_reset_m_cells = conv_reset_m_cells
        self.info = info
        self.models = []
        self.layer_num = len(self.conv_rnn_cells)
        self.m_downsample = []
        self.reset_m = []
        self.H = {}
        self.C = {}
        self.M = {}
        self.hs = {}
        self.cs = {}
        self.ms = {}
        for idx in range(len(self.conv_m_cells)):
            self.m_downsample.append(
                conv.Conv2D(
                    cell_param=self.conv_m_cells[idx]
                ).cuda()
            )
            self.reset_m.append(
                conv.DeConv2D(
                    cell_param=self.conv_reset_m_cells[idx]
                ).cuda()
            )
        self.m_downsample = nn.ModuleList(self.m_downsample)
        self.reset_m = nn.ModuleList(self.reset_m)

        for idx in range(self.layer_num):
            self.models.append(
                conv.Conv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )

            self.models.append(
                ST_ConvLSTMCell(
                    cell_param=self.conv_rnn_cells[idx],
                )
            )

        self.models = nn.ModuleList(self.models)

    def init_paramter(self,shape):
        return Variable(torch.zeros(shape).cuda(),requires_grad=True)


    def forward(self, input):

        self.n_step = input.size()[1]

        for t in range(self.n_step):
            x_t = input[:, t, :, :, :, ]
            if t == 0:
                self.cell(x_t,first_timestep=True)
            else:
                self.cell(x_t)

        return self.current_states,self.current_M

    @property
    def current_states(self):
        return (self.H,self.C)

    @property
    def current_M(self):
        return self.M

    def cell_reset_m(self,M):
        for idx in range(len(self.conv_m_cells)):
            M = self.reset_m[idx](M)
        return M

    def cell(self,input,first_timestep = False):

        new_states = []
        if first_timestep:
            input_shape = input.size()
            states = []
            for i in range(self.layer_num):
                h_shape = [
                    input_shape[0],
                    self.info['MODEL_NETS']['ENCODE_CELLS'][i][1],
                    self.info['MODEL_NETS']['ENSHAPE'][i+1],
                    self.info['MODEL_NETS']['ENSHAPE'][i + 1]
                ]
                c_shape = [
                    input_shape[0],
                    self.info['MODEL_NETS']['ENCODE_CELLS'][i][1],
                    self.info['MODEL_NETS']['ENSHAPE'][i + 1],
                    self.info['MODEL_NETS']['ENSHAPE'][i + 1]
                ]
                m_shape = [
                    input_shape[0],
                    self.info['MODEL_NETS']['m_channels'][i],
                    self.info['MODEL_NETS']['ENSHAPE'][i + 1],
                    self.info['MODEL_NETS']['ENSHAPE'][i + 1]
                ]

                m = self.init_paramter(m_shape)
                h = self.init_paramter(h_shape)
                c = self.init_paramter(c_shape)
                self.hs[i] = h
                self.cs[i] = c
                self.ms[i] = m

        else:
            pass
        # assert M is not None and states is not None
        for layer_idx in range(self.layer_num):
            if first_timestep == True:
                if layer_idx == 0:
                    input = self.models[layer_idx*2](input)
                    self.H[layer_idx],self.C[layer_idx],self.M[layer_idx] = self.models[layer_idx * 2 + 1](input,(self.hs[layer_idx],self.cs[layer_idx]), self.ms[layer_idx])
                    continue
                else:
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2 + 1](self.models[layer_idx*2](self.H[layer_idx-1]),(self.hs[layer_idx],self.cs[layer_idx]),self.m_downsample[layer_idx-1](self.M[layer_idx-1]))
                    continue
            else:
                if layer_idx == 0:
                    input = self.models[layer_idx * 2](input)
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2 + 1](input, (
                    self.H[layer_idx], self.C[layer_idx]), self.cell_reset_m(self.M[self.layer_num - 1]))
                    continue
                else:
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2 + 1](self.models[layer_idx*2](self.H[layer_idx-1]),(
                    self.H[layer_idx], self.C[layer_idx]),self.m_downsample[layer_idx-1](self.M[layer_idx-1]))
                    continue

        return self.H,self.C,self.M




class Encode_Decode_ST_ConvLSTM(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            info,
    ):
        super(Encode_Decode_ST_ConvLSTM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.info = info
        self.models = nn.ModuleList([self.encoder,self.decoder])

    def forward(self, input):
        in_decode_frame_dat = Variable(torch.zeros(
            self.info['TRAIN']['BATCH_SIZE'],
            self.info['DATA']['OUTPUT_SEQ_LEN'],
            self.info['MODEL_NETS']['ENCODE_CELLS'][-1][1],
            self.info['MODEL_NETS']['DESHAPE'][0],
            self.info['MODEL_NETS']['DESHAPE'][0],
        ).cuda(),requires_grad=True)
        encode_states,M = self.models[0](input)
        output = self.models[1](in_decode_frame_dat, encode_states,M)
        return output