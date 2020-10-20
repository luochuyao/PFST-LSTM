import torch.nn as nn
import torch
class Conv2D(nn.Module):
    def __init__(self,
                 cell_param
                 ):
        super(Conv2D, self).__init__()
        self.cell_param = cell_param
        self.net = nn.Sequential()
        self.act_conv2d = nn.Conv2d(
            in_channels=self.cell_param['in_channel'],
            out_channels=self.cell_param['out_channel'],
            kernel_size=self.cell_param['kernel_size'],
            stride=self.cell_param['stride'],
            padding=self.cell_param['padding']
        )
        torch.nn.init.xavier_uniform_(self.act_conv2d.weight)
        torch.nn.init.constant_(self.act_conv2d.bias, 0)
        self.net.add_module('conv',self.act_conv2d)
        if self.cell_param['activate'] == None:
            pass
        elif self.cell_param['activate'] == 'relu':
            self.net.add_module('activate',nn.ReLU())
        elif self.cell_param['activate'] == 'tanh':
            self.net.add_module('activate', nn.Tanh())

    def forward(self, input):

        output = []
        for t in range(input.size()[1]):
            output.append(self.net(input[:,t,:,:,:,]))

        return torch.stack(output,1)


class DeConv2D(nn.Module):
    def __init__(self,
                 cell_param
                 ):
        super(DeConv2D, self).__init__()
        self.cell_param = cell_param
        self.net = nn.Sequential()
        self.act_de_conv2d = nn.ConvTranspose2d(
            in_channels=self.cell_param['in_channel'],
            out_channels=self.cell_param['out_channel'],
            kernel_size=self.cell_param['kernel_size'],
            stride=self.cell_param['stride'],
            padding=self.cell_param['padding'],
            output_padding=self.cell_param['output_padding']
        )
        torch.nn.init.xavier_uniform_(self.act_de_conv2d.weight)
        torch.nn.init.constant_(self.act_de_conv2d.bias, 0)
        self.net.add_module('de_conv',self.act_de_conv2d)
        if self.cell_param['activate'] == None:
            pass
        elif self.cell_param['activate'] == 'relu':
            self.act_de_conv2d.add_module('activate',nn.ReLU())
        elif self.cell_param['activate'] == 'tanh':
            self.act_de_conv2d.add_module('activate', nn.Tanh())

    def forward(self, input):

        output = []
        for t in range(input.size()[1]):
            output.append(self.net(input[:,t,:,:,:,]))

        return torch.stack(output,1)