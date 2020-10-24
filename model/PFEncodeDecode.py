import numpy as np
import torch
import torch.nn as nn
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model.PF_ConvLSTM import *
from model.Conv import *
import model.ConvSeq as conv_seq

class Decoder_PF_ConvLSTM(nn.Module):
    def __init__(
            self,
            conv_rnn_cells,
            conv_cells,
            output_cells,
            input_cells,
            info,
    ):
        super(Decoder_PF_ConvLSTM, self).__init__()
        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.input_cells = input_cells
        self.output_cells = output_cells
        self.info =info
        self.models = []
        self.input_models = []
        self.layer_num = len(self.conv_rnn_cells)
        self.last_input = {}
        self.H = {}
        self.C = {}
        self.HH = {}
        self.flows = {}

        for idx in range(len(self.input_cells)):
            self.input_models.append(
                Conv2D(
                    cell_param = self.input_cells[idx]
                )
            )

        self.input_models = nn.ModuleList(self.input_models)

        self.input_conv = []

        for idx in range(self.layer_num):
            self.models.append(
                PF_ConvLSTMCell(
                    cell_param=self.conv_rnn_cells[idx],
                )
            )
            self.models.append(
                DeConv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )
        self.models = nn.ModuleList(self.models)
        self.out_models = []
        for output_cell in self.output_cells:
            self.out_models.append(
                Conv2D(
                    cell_param=output_cell
                ).cuda()
            )
        self.out_models = nn.ModuleList(self.out_models)

    def inverse(self,data):
        new_data = []
        for i in range(len(data)):
            new_data.append(data[len(data)-i-1])
        return new_data



    def forward(self,input,states):
        self.n_step = input.size()[1]
        output = []

        for t in range(self.n_step):
            x_t = input[:, t, :, :, :, ]
            if t == 0:
                for layer_idx in range(self.layer_num):
                    self.H[self.layer_num-1-layer_idx] = states[0][layer_idx]
                    self.C[self.layer_num-1-layer_idx] = states[1][layer_idx]
                self.cell(x_t, True)
            else:
                self.cell(x_t, False)

            out = self.current_HH
            for out_model in self.out_models:
                out = out_model(out)
            self.update_last_input(out)

            output.append(out)
        output = torch.stack(output,1)


        return output

    @property
    def current_HH(self):
        return self.HH

    @property
    def current_states(self):
        return (self.H, self.C)


    def update_last_input(self,out):

        for idx in range(len(self.input_models)):
            out = self.input_models[idx](out)
            if idx == 0:
                pass
            else:
                self.last_input[self.layer_num-idx] = out
        pass



    def cell(self,input,first_timestep = False):
        if first_timestep:
            for layer_idx in range(self.layer_num):
                if layer_idx == 0:
                    self.H[layer_idx], self.C[layer_idx] = self.models[layer_idx * 2](input,input,(self.H[layer_idx],self.C[layer_idx]))
                else:
                    self.H[layer_idx], self.C[layer_idx] = self.models[layer_idx * 2](self.models[2*layer_idx-1](self.H[layer_idx-1]),self.models[2*layer_idx-1](self.H[layer_idx-1]),(self.H[layer_idx],self.C[layer_idx]))

        else:
            for layer_idx in range(self.layer_num):
                if layer_idx == 0:
                    # input = self.models[layer_idx * 2](input)
                    self.H[layer_idx], self.C[layer_idx] = self.models[layer_idx * 2](input, self.last_input[layer_idx],(
                    self.H[layer_idx], self.C[layer_idx]))

                    continue
                else:
                    self.H[layer_idx], self.C[layer_idx] = self.models[layer_idx * 2](self.models[layer_idx*2-1](self.H[layer_idx-1]),self.last_input[layer_idx],(
                    self.H[layer_idx], self.C[layer_idx]))

                    continue

        self.HH = self.models[self.layer_num * 2-1](self.H[self.layer_num-1])

        return self.H,self.C


class Encoder_PF_ConvLSTM(nn.Module):
    def __init__(
            self,
            conv_rnn_cells,
            conv_cells,
            info,
                 ):
        super(Encoder_PF_ConvLSTM, self).__init__()
        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.info = info
        self.models = []
        self.layer_num = len(self.conv_rnn_cells)
        self.m_downsample = []
        self.reset_m = []
        self.last_input = {}
        self.H = {}
        self.C = {}
        self.hs = {}
        self.cs = {}
        self.flows = {}


        for idx in range(self.layer_num):
            self.models.append(
                Conv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )

            self.models.append(
                PF_ConvLSTMCell(
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

        return self.current_states

    @property
    def current_states(self):
        return (self.H,self.C)


    def cell(self,input,first_timestep = False):

        if first_timestep:
            input_shape = input.size()

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



                h = self.init_paramter(h_shape)
                c = self.init_paramter(c_shape)
                self.hs[i] = h
                self.cs[i] = c

        else:
            pass
        # assert M is not None and states is not None
        for layer_idx in range(self.layer_num):
            if first_timestep == True:
                if layer_idx == 0:
                    input = self.models[layer_idx*2](input)
                    self.last_input[layer_idx] = input
                    self.H[layer_idx],self.C[layer_idx] = self.models[layer_idx * 2 + 1](input,input,(self.hs[layer_idx],self.cs[layer_idx]))

                    continue
                else:
                    self.last_input[layer_idx] = self.models[layer_idx*2](self.H[layer_idx-1])
                    self.H[layer_idx], self.C[layer_idx] = self.models[layer_idx * 2 + 1](self.models[layer_idx*2](self.H[layer_idx-1]),self.models[layer_idx*2](self.H[layer_idx-1]),(self.hs[layer_idx],self.cs[layer_idx]))
                    continue
            else:
                if layer_idx == 0:
                    input = self.models[layer_idx * 2](input)
                    self.H[layer_idx], self.C[layer_idx] = self.models[layer_idx * 2 + 1](input,self.last_input[layer_idx],(
                    self.H[layer_idx], self.C[layer_idx]))
                    self.last_input[layer_idx] = input
                    continue
                else:
                    self.H[layer_idx], self.C[layer_idx]= self.models[layer_idx * 2 + 1](self.models[layer_idx*2](self.H[layer_idx-1]),self.last_input[layer_idx],(
                    self.H[layer_idx], self.C[layer_idx]))
                    self.last_input[layer_idx] = self.models[layer_idx*2](self.H[layer_idx-1])
                    continue

        return self.H,self.C




class Encode_Decode_PF_ConvLSTM(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            info,
    ):
        super(Encode_Decode_PF_ConvLSTM, self).__init__()
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
        encode_states = self.models[0](input)
        output = self.models[1](in_decode_frame_dat, encode_states)
        return output


if __name__ == '__main__':
    import yaml
    from util.utils import *
    from data.CIKM.data_iterator import *
    from model.rover import *
    path = '/home/ices/PycharmProject/FST_ConvRNNs/experiment/CIKM/config/dec_PF_ConvLSTM_CIKM.yml'
    f = open(path)
    configuration = yaml.safe_load(f)
    print(configuration)

    encode_conv_rnn_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['ENCODE_CELLS']):
        param = get_cell_param(cell)
        encode_conv_rnn_cells.append(param)

    downsample_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['DOWNSAMPLE_CONVS']):
        if idx == len(configuration['MODEL_NETS']['DOWNSAMPLE_CONVS']) - 1:
            downsample_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['ENCODE_PADDING'][idx], activate='tanh'))
        else:
            downsample_cells.append(get_conv_param(cell, padding=configuration['MODEL_NETS']['ENCODE_PADDING'][idx]))


    decode_conv_rnn_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['DECODE_CELLS']):
        param = get_cell_param(cell)
        decode_conv_rnn_cells.append(param)

    output_conv_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['OUTPUT_CONV']):
        if idx == len(configuration['MODEL_NETS']['OUTPUT_CONV']) - 1:
            output_conv_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['OUTPUT_PADDING'][idx], activate='tanh'))
        else:
            output_conv_cells.append(get_conv_param(cell, padding=configuration['MODEL_NETS']['OUTPUT_PADDING'][idx]))




    upsample_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['UPSAMPLE_CONVS']):
        if idx == len(configuration['MODEL_NETS']['UPSAMPLE_CONVS']) - 1:
            upsample_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['DECODE_PADDING'][idx], activate='tanh'))
        else:
            upsample_cells.append(get_conv_param(cell, padding=configuration['MODEL_NETS']['DECODE_PADDING'][idx]))



    input_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['INPUT_CELL']):
        if idx == len(configuration['MODEL_NETS']['INPUT_CELL']) - 1:
            input_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['INPUT_PADDING'][idx], activate='tanh'))
        else:
            input_cells.append(get_conv_param(cell, padding=configuration['MODEL_NETS']['INPUT_PADDING'][idx]))

    encoder = Encoder_PF_ConvLSTM(
        conv_rnn_cells=encode_conv_rnn_cells,
        conv_cells=downsample_cells,
        info=configuration
    ).cuda()

    decoder = Decoder_PF_ConvLSTM(
        conv_rnn_cells=decode_conv_rnn_cells,
        conv_cells=upsample_cells,
        output_cells=output_conv_cells,
        input_cells=input_cells,
        info=configuration
    ).cuda()

    encoder_decoder_model = Encode_Decode_PF_ConvLSTM(
        encoder=encoder,
        decoder=decoder,
        info=configuration
    ).cuda()
    optimizer = optim.Adam(encoder_decoder_model.parameters(), lr=configuration['TRAIN']['LEARNING_RATE'])

    for i in range(100):
        frame_dat = sample(
            batch_size=configuration['TRAIN']['BATCH_SIZE']
        )
        frame_dat = frame_dat.transpose(1,0,2,3,4)
        in_frame_dat = frame_dat[:5]
        target_frame_dat = frame_dat[5:]


        in_frame_dat = normalization(in_frame_dat,255.0)
        target_frame_dat = normalization(target_frame_dat, 255.0)

        in_frame_dat = in_frame_dat.transpose(1,0,4,2,3)
        target_frame_dat = target_frame_dat.transpose(1, 0, 4, 2, 3)




        if torch.cuda.is_available():
            in_frame_dat = Variable(torch.from_numpy(in_frame_dat).float().cuda(),requires_grad=True)
            target_frame_dat = Variable(torch.from_numpy(target_frame_dat).float().cuda())




        optimizer.zero_grad()
        output = encoder_decoder_model(in_frame_dat)
        criterion = torch.nn.MSELoss()
        loss = criterion(output, target_frame_dat)
        loss.backward()
        optimizer.step()
        print(loss)
