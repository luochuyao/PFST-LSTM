import torch
import torch.nn as nn
from torch.autograd import Variable

class ConvRNN(nn.Module):
    def __init__(self,
                 cell_param,
                 return_state,
                 return_sequence,
                 ):
        super(ConvRNN, self).__init__()
        self.cell_param = cell_param
        self.return_state =return_state
        self.return_sequence = return_sequence

    def init_paramter(self,shape):
        return Variable(torch.zeros(shape).cuda())

    def forward(self, input, state=None):

        if state is None:
            state = self.init_hidden(input)
        else:
            state = state

        self.minibatch_size = input.size()[0]
        self.n_step = input.size()[1]
        # outputs = Variable(torch.zeros(self.minibatch_size, self.n_step, self.cell_param['output_channels'], input.size()[-2], input.size()[-1]).cuda())
        outputs = []

        for i in range(self.n_step):
            x_t = input[:, i, :, :, :]
            state = self.cell(x_t, state)
            if type(state) == type((1,2)):
                outputs.append(state[0])
            else:
                outputs.append(state)

        outputs = torch.stack(outputs,1)
        self.outputs = outputs
        if self.return_sequence:
            if self.return_state:
                return outputs, state
            else:
                return outputs
        else:
            if self.return_state:
                return state
            else:
                if type(state) == type((1)):
                    return state[0]
                else:
                    return state


class ConvRNNCell(nn.Module):
    def __init__(self, cell_param):
        super(ConvRNNCell, self).__init__()
        self.input_dim = cell_param['input_channels']
        self.output_dim = cell_param['output_channels']
        self.input_to_state_kernel_size = cell_param['input_to_state_kernel_size']
        self.state_to_state_kernel_size = cell_param['state_to_state_kernel_size']
