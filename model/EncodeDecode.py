import numpy as np
import torch
import torch.nn as nn
from model.ConvLSTM import *
from model.ConvGRU import *
from model.TrajGRU import *
from model.ST_ConvLSTM import *
from model.ST_TrajLSTM import *
from model.TrajLSTM import *
import model.Conv as conv
import model.ConvSeq as conv_seq





class Decoder_TrajLSTM(nn.Module):
    def __init__(self,
                 conv_rnn_cells,
                 conv_cells,
                 output_cells,
                 ):
        super(Decoder_TrajLSTM, self).__init__()
        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.models = []
        self.output_cells = output_cells
        self.layer_num = len(self.conv_rnn_cells)
        for idx in range(self.layer_num):
            self.models.append(
                TrajLSTM(
                    cell_param=self.conv_rnn_cells[idx],
                    return_sequence=True,
                    return_state=False,
                )
            )
            self.models.append(
                conv_seq.DeConv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )
        for output_cell in output_cells:
            self.models.append(
                conv_seq.Conv2D(
                    cell_param=output_cell
                ).cuda()
            )
        self.models = nn.ModuleList(self.models)

    def forward(self, input,state = None):
        assert state is not None
        for layer_idx in range(self.layer_num):
            current_conv_rnn_output = self.models[2*layer_idx](input,state[self.layer_num-1-layer_idx])
            current_conv_output = self.models[2*layer_idx+1](current_conv_rnn_output)

            if layer_idx == self.layer_num-1:
                pass
            else:
                input = current_conv_output

        output = current_conv_output
        for out_layer_idx in range(len(self.output_cells)):
            output = self.models[2*len(self.conv_rnn_cells)+out_layer_idx](output)
        # print('the size of output is:', output.size(), layer_idx)
        return output

class Encoder_TrajLSTM(nn.Module):
    def __init__(self,
                 conv_rnn_cells,
                 conv_cells,
    ):
        super(Encoder_TrajLSTM, self).__init__()

        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.models = []
        self.layer_num = len(self.conv_rnn_cells)
        for idx in range(self.layer_num):
            self.models.append(
                conv_seq.Conv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )
            self.models.append(
                TrajLSTM(
                    cell_param=self.conv_rnn_cells[idx],
                    return_sequence=True,
                    return_state=True,
                )
            )

        self.models = nn.ModuleList(self.models)

    def forward(self, input,state = None):
        encode_state = []
        for layer_idx in range(self.layer_num):
            current_conv_output = self.models[2*layer_idx](input)
            current_conv_rnn_output,state = self.models[2*layer_idx+1](current_conv_output)
            encode_state.append(state)
            if layer_idx == self.layer_num-1:
                pass
            else:
                input = current_conv_rnn_output
        return encode_state

class Encode_Decode_TrajLSTM(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 info,
                 ):
        super(Encode_Decode_TrajLSTM, self).__init__()
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
        ).cuda())
        encode_states = self.models[0](input)
        output = self.models[1](in_decode_frame_dat, encode_states)
        return output

class Decoder_ST_TrajLSTM(nn.Module):
    def __init__(
            self,
            conv_rnn_cells,
            conv_cells,
            conv_m_cells,
            conv_reset_m_cells,
            output_cells,
            info,
    ):
        super(Decoder_ST_TrajLSTM, self).__init__()
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
                ST_TrajLSTMCell(
                    cell_param=self.conv_rnn_cells[idx],
                )
            )
            self.models.append(
                conv.DeConv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )

        for output_cell in self.output_cells:

            self.models.append(
                conv.Conv2D(
                    cell_param=output_cell
                ).cuda()
            )
        self.models = nn.ModuleList(self.models)

    def forward(self, input,states,M):
        states.reverse()
        self.n_step = input.size()[1]
        output = []
        for t in range(self.n_step):
            x_t = input[:,t,:,:,:,]
            out,states,M = self.cell(x_t,M,states)
            output.append(out)
            if t == self.n_step-1:
                pass
            else:
                M = self.cell_reset_m(M)
        output = torch.stack(output,1)
        return output

    def cell_reset_m(self,M):
        for idx in range(len(self.conv_m_cells)):
            M = self.reset_m[idx](M)
        return M

    def cell(self,input,M=None,states=None):
        new_states = []
        assert M is not None and states is not None
        for layer_idx in range(self.layer_num):
            h,c,M = self.models[layer_idx*2](input,states[layer_idx],M)
            new_states.append((h,c))
            input = h
            input = self.models[layer_idx*2+1](input)
            if layer_idx<self.layer_num-1:
                M = self.m_upsample[layer_idx](M)

        output = input

        for out_layer_idx in range(len(self.output_cells)):
            output = self.models[2 * self.layer_num  + out_layer_idx](output)
        temp_model = self.models[2 * self.layer_num  + 0]

        return output,new_states,M
class Encoder_ST_TrajLSTM(nn.Module):
    def __init__(
            self,
            conv_rnn_cells,
            conv_cells,
            conv_m_cells,
            conv_reset_m_cells,
            info,
            test_hidden_layer=0
                 ):
        super(Encoder_ST_TrajLSTM, self).__init__()
        self.test_hidden_layer = test_hidden_layer
        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.conv_m_cells = conv_m_cells
        self.conv_reset_m_cells = conv_reset_m_cells
        self.info = info
        self.models = []
        self.layer_num = len(self.conv_rnn_cells)
        self.m_downsample = []
        self.reset_m = []
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
                ST_TrajLSTMCell(
                    cell_param=self.conv_rnn_cells[idx],
                )
            )

        self.models = nn.ModuleList(self.models)

    def init_paramter(self,shape):
        return Variable(torch.zeros(shape).cuda(),requires_grad=True)


    def forward(self, input):

        self.n_step = input.size()[1]
        states = None
        M = None
        for t in range(self.n_step):

            x_t = input[:,t,:,:,:,]
            states,M = self.cell(x_t,M,states)
            if t == self.n_step-1:
                pass
            else:
                M = self.cell_reset_m(M)

        return states,M

    def cell_reset_m(self,M):
        for idx in range(len(self.conv_m_cells)):
            M = self.reset_m[idx](M)
        return M

    def cell(self,input,M=None,states=None):

        new_states = []
        if states is None and M is None:
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

                h = self.init_paramter(h_shape)
                c = self.init_paramter(c_shape)

                states.append((h,c))
                if i == 0:
                    m_shape = [
                        input_shape[0],
                        self.info['MODEL_NETS']['m_channels'][i],
                        self.info['MODEL_NETS']['ENSHAPE'][i + 1],
                        self.info['MODEL_NETS']['ENSHAPE'][i + 1]
                    ]
                    M = self.init_paramter(m_shape)
        else:
            pass
        assert M is not None and states is not None
        for layer_idx in range(self.layer_num):

            if layer_idx == 0:
                input = self.models[layer_idx*2](input)
                h,c,M = self.models[layer_idx*2+1](input,states[layer_idx],M)
                input = h
            else:
                input = self.models[layer_idx*2](input)
                M = self.m_downsample[layer_idx-1](M)
                h,c,M = self.models[layer_idx*2+1](input,states[layer_idx],M)
                input = h
            # if layer_idx == self.test_hidden_layer:
            #     print('At encode step, the shape of C is ',c.size(),' at ',str(layer_idx),' layer')
            new_states.append((h, c))

        return new_states,M

class Encode_Decode_ST_TrajLSTM(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            info,
    ):
        super(Encode_Decode_ST_TrajLSTM, self).__init__()
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

        for output_cell in self.output_cells:

            self.models.append(
                conv.Conv2D(
                    cell_param=output_cell
                ).cuda()
            )
        self.models = nn.ModuleList(self.models)

    def forward(self, input,states,M):
        states.reverse()
        self.n_step = input.size()[1]
        output = []
        for t in range(self.n_step):
            x_t = input[:,t,:,:,:,]
            out,states,M = self.cell(x_t,M,states)
            output.append(out)
            if t == self.n_step-1:
                pass
            else:
                M = self.cell_reset_m(M)
        output = torch.stack(output,1)
        return output

    def inverse(self,data):
        new_data = []
        for i in range(len(data)):
            new_data.append(data[len(data)-i-1])
        return new_data



    def cell_reset_m(self,M):
        for idx in range(len(self.conv_m_cells)):
            M = self.reset_m[idx](M)
        return M

    def cell(self,input,M=None,states=None):
        new_states = []
        assert M is not None and states is not None
        for layer_idx in range(self.layer_num):
            h,c,M = self.models[layer_idx*2](input,states[layer_idx],M)
            new_states.append((h,c))
            input = h
            input = self.models[layer_idx*2+1](input)
            if layer_idx<self.layer_num-1:
                M = self.m_upsample[layer_idx](M)

        output = input

        for out_layer_idx in range(len(self.output_cells)):
            output = self.models[2 * self.layer_num  + out_layer_idx](output)
        temp_model = self.models[2 * self.layer_num  + 0]

        return output,new_states,M






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
        states = None
        M = None
        for t in range(self.n_step):

            x_t = input[:,t,:,:,:,]
            states,M = self.cell(x_t,M,states)
            if t == self.n_step-1:
                pass
            else:
                M = self.cell_reset_m(M)

        return states,M

    def cell_reset_m(self,M):
        for idx in range(len(self.conv_m_cells)):
            M = self.reset_m[idx](M)
        return M


    def cell(self,input,M=None,states=None):

        new_states = []
        if states is None and M is None:
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

                h = self.init_paramter(h_shape)
                c = self.init_paramter(c_shape)

                states.append((h,c))
                if i == 0:
                    m_shape = [
                        input_shape[0],
                        self.info['MODEL_NETS']['m_channels'][i],
                        self.info['MODEL_NETS']['ENSHAPE'][i + 1],
                        self.info['MODEL_NETS']['ENSHAPE'][i + 1]
                    ]
                    M = self.init_paramter(m_shape)
        else:
            pass
        assert M is not None and states is not None
        for layer_idx in range(self.layer_num):
            if layer_idx == 0:
                input = self.models[layer_idx*2](input)
                h,c,M = self.models[layer_idx*2+1](input,states[layer_idx],M)
                input = h
            else:
                input = self.models[layer_idx*2](input)
                M = self.m_downsample[layer_idx-1](M)
                h,c,M = self.models[layer_idx*2+1](input,states[layer_idx],M)
                input = h
            new_states.append((h, c))

        return new_states,M




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

class Encode_Decode_ConvLSTM(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 info,
                 ):
        super(Encode_Decode_ConvLSTM, self).__init__()
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
        ).cuda())
        encode_states = self.models[0](input)
        output = self.models[1](in_decode_frame_dat, encode_states)
        return output

class Encode_Decode_ConvGRU(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 info,
                 ):
        super(Encode_Decode_ConvGRU, self).__init__()
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
        ).cuda())
        encode_states = self.models[0](input)
        output = self.models[1](in_decode_frame_dat, encode_states)
        return output

class Encode_Decode_TrajGRU(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 info,
                 ):
        super(Encode_Decode_TrajGRU, self).__init__()
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
        ).cuda())
        encode_states = self.models[0](input)
        output = self.models[1](in_decode_frame_dat, encode_states)
        return output


class Decoder_TrajGRU(nn.Module):
    def __init__(self,
                 conv_rnn_cells,
                 conv_cells,
                 output_cells,
                 ):
        super(Decoder_TrajGRU, self).__init__()
        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.models = []
        self.output_cells = output_cells
        self.layer_num = len(self.conv_rnn_cells)
        for idx in range(self.layer_num):
            self.models.append(
                TrajGRU(
                    cell_param=self.conv_rnn_cells[idx],
                    return_sequence=True,
                    return_state=False,
                )
            )
            self.models.append(
                conv_seq.DeConv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )
        for output_cell in output_cells:
            self.models.append(
                conv_seq.Conv2D(
                    cell_param=output_cell
                ).cuda()
            )
        self.models = nn.ModuleList(self.models)

    def forward(self, input,state = None):
        assert state is not None
        for layer_idx in range(self.layer_num):
            current_conv_rnn_output = self.models[2*layer_idx](input,state[self.layer_num-1-layer_idx])
            current_conv_output = self.models[2*layer_idx+1](current_conv_rnn_output)

            if layer_idx == self.layer_num-1:
                pass
            else:
                input = current_conv_output

        output = current_conv_output
        for out_layer_idx in range(len(self.output_cells)):
            output = self.models[2*len(self.conv_rnn_cells)+out_layer_idx](output)
        # print('the size of output is:', output.size(), layer_idx)
        return output

class Decoder_ConvGRU(nn.Module):
    def __init__(self,
                 conv_rnn_cells,
                 conv_cells,
                 output_cells,
                 ):
        super(Decoder_ConvGRU, self).__init__()
        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.models = []
        self.output_cells = output_cells
        self.layer_num = len(self.conv_rnn_cells)
        for idx in range(self.layer_num):
            self.models.append(
                ConvGRU(
                    cell_param=self.conv_rnn_cells[idx],
                    return_sequence=True,
                    return_state=False,
                )
            )
            self.models.append(
                conv_seq.DeConv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )
        for output_cell in output_cells:
            self.models.append(
                conv_seq.Conv2D(
                    cell_param=output_cell
                ).cuda()
            )
        self.models = nn.ModuleList(self.models)

    def forward(self, input,state = None):
        assert state is not None
        for layer_idx in range(self.layer_num):
            current_conv_rnn_output = self.models[2*layer_idx](input,state[self.layer_num-1-layer_idx])
            current_conv_output = self.models[2*layer_idx+1](current_conv_rnn_output)

            if layer_idx == self.layer_num-1:
                pass
            else:
                input = current_conv_output

        output = current_conv_output
        for out_layer_idx in range(len(self.output_cells)):
            output = self.models[2*len(self.conv_rnn_cells)+out_layer_idx](output)
        # print('the size of output is:', output.size(), layer_idx)
        return output


class Decoder_ConvLSTM(nn.Module):
    def __init__(self,
                 conv_rnn_cells,
                 conv_cells,
                 output_cells,
                 ):
        super(Decoder_ConvLSTM, self).__init__()
        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.models = []
        self.output_cells = output_cells
        self.layer_num = len(self.conv_rnn_cells)
        for idx in range(self.layer_num):
            self.models.append(
                ConvLSTM(
                    cell_param=self.conv_rnn_cells[idx],
                    return_sequence=True,
                    return_state=False,
                )
            )
            self.models.append(
                conv_seq.DeConv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )
        for output_cell in output_cells:
            self.models.append(
                conv_seq.Conv2D(
                    cell_param=output_cell
                ).cuda()
            )
        self.models = nn.ModuleList(self.models)

    def forward(self, input,state = None):
        assert state is not None
        for layer_idx in range(self.layer_num):
            current_conv_rnn_output = self.models[2*layer_idx](input,state[self.layer_num-1-layer_idx])
            current_conv_output = self.models[2*layer_idx+1](current_conv_rnn_output)

            if layer_idx == self.layer_num-1:
                pass
            else:
                input = current_conv_output

        output = current_conv_output
        for out_layer_idx in range(len(self.output_cells)):
            output = self.models[2*len(self.conv_rnn_cells)+out_layer_idx](output)
        # print('the size of output is:', output.size(), layer_idx)
        return output



class Encoder_TrajGRU(nn.Module):
    def __init__(self,
                 conv_rnn_cells,
                 conv_cells,
    ):
        super(Encoder_TrajGRU, self).__init__()

        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.models = []
        self.layer_num = len(self.conv_rnn_cells)
        for idx in range(self.layer_num):
            self.models.append(
                conv_seq.Conv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )
            self.models.append(
                TrajGRU(
                    cell_param=self.conv_rnn_cells[idx],
                    return_sequence=True,
                    return_state=True,
                )
            )

        self.models = nn.ModuleList(self.models)

    def forward(self, input,state = None):
        encode_state = []
        for layer_idx in range(self.layer_num):
            current_conv_output = self.models[2*layer_idx](input)
            current_conv_rnn_output,state = self.models[2*layer_idx+1](current_conv_output)
            encode_state.append(state)
            if layer_idx == self.layer_num-1:
                pass
            else:
                input = current_conv_rnn_output
        return encode_state


class Encoder_ConvGRU(nn.Module):
    def __init__(self,
                 conv_rnn_cells,
                 conv_cells,
    ):
        super(Encoder_ConvGRU, self).__init__()

        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.models = []
        self.layer_num = len(self.conv_rnn_cells)
        for idx in range(self.layer_num):
            self.models.append(
                conv_seq.Conv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )
            self.models.append(
                ConvGRU(
                    cell_param=self.conv_rnn_cells[idx],
                    return_sequence=True,
                    return_state=True,
                )
            )

        self.models = nn.ModuleList(self.models)

    def forward(self, input,state = None):
        encode_state = []
        for layer_idx in range(self.layer_num):
            current_conv_output = self.models[2*layer_idx](input)
            current_conv_rnn_output,state = self.models[2*layer_idx+1](current_conv_output)
            encode_state.append(state)
            if layer_idx == self.layer_num-1:
                pass
            else:
                input = current_conv_rnn_output
        return encode_state

class Encoder_ConvLSTM(nn.Module):
    def __init__(self,
                 conv_rnn_cells,
                 conv_cells,
    ):
        super(Encoder_ConvLSTM, self).__init__()
        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.models = []
        self.layer_num = len(self.conv_rnn_cells)
        for idx in range(self.layer_num):
            self.models.append(
                conv_seq.Conv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )
            self.models.append(
                ConvLSTM(
                    cell_param=self.conv_rnn_cells[idx],
                    return_sequence=True,
                    return_state=True,
                )
            )
        self.models = nn.ModuleList(self.models)


    def forward(self, input,state = None):
        encode_state = []
        for layer_idx in range(self.layer_num):

            current_conv_output = self.models[2*layer_idx](input)

            current_conv_rnn_output,state = self.models[2*layer_idx+1](current_conv_output)
            encode_state.append(state)
            if layer_idx == self.layer_num-1:
                pass
            else:
                input = current_conv_rnn_output
        return encode_state

