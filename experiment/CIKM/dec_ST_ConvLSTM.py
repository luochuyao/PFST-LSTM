import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
rootPath = os.path.split(rootPath)[0]
sys.path.append(rootPath)

import yaml
import torch.optim as optim
from torchsummary import summary
from util.utils import *
from util.bad_grad_viz import *
# from experiment.CIKM.BaseSequenceModel import *
from model.EncodeDecode import *
from data.CIKM.data_iterator import *


class sequence_model(object):
    def __init__(self,name,encoder_decoder_model,info):
        self.test_save_root = '/mnt/A/meteorological/2500_ref_seq/CIKM_dec_ST_ConvLSTM_test/'
        self.validation_save_root = '/mnt/A/meteorological/2500_ref_seq/CIKM_dec_ST_ConvLSTM_validation/'
        self.name = name
        self.encoder_decoder_model = encoder_decoder_model
        self.info = info


    def train(self):
        self.optimizer = optim.Adam(self.encoder_decoder_model.parameters(), lr=self.info['TRAIN']['LEARNING_RATE'])
        tolerate_iter = 0
        best_mse = math.inf
        self.encoder_decoder_model.train()

        for train_iter in range(self.info['TRAIN']['EPOCHES']):
            frame_dat = sample(
                batch_size=self.info['TRAIN']['BATCH_SIZE']
            )

            frame_dat = normalization(frame_dat,255.0)
            in_frame_dat = frame_dat[:,:5]
            target_frame_dat = frame_dat[:,5:]
            in_frame_dat = in_frame_dat.transpose(0,1,4,2,3)
            target_frame_dat = target_frame_dat.transpose(0, 1, 4, 2, 3)

            if torch.cuda.is_available():
                self.encoder_decoder_model = self.encoder_decoder_model.cuda()
                in_frame_dat = Variable(torch.from_numpy(in_frame_dat).float().cuda(),requires_grad=True)
                target_frame_dat = Variable(torch.from_numpy(target_frame_dat).float().cuda())

            self.optimizer.zero_grad()
            output = self.encoder_decoder_model(in_frame_dat)
            criterion = torch.nn.MSELoss()
            loss = criterion(output, target_frame_dat)
            loss.backward()
            self.optimizer.step()


            if (train_iter+1)%self.info['TRAIN']['DISTPLAY_STEP'] == 0:
                print('Train iter is:',(train_iter+1),'current loss is:',float(loss.cpu().data.numpy()))

            if (train_iter+1) % self.info['TRAIN']['TEST_STEP'] == 0:
                cur_loss = self.validation()
                print('validation loss is :', cur_loss)
                if cur_loss < best_mse:
                    best_mse = cur_loss
                    tolerate_iter = 0
                    self.save_model()
                else:
                    tolerate_iter += 1
                    if tolerate_iter == self.info['TRAIN']['LOSS_LIMIT']:
                        print('the best validation loss is:', best_mse)
                        self.load_model()
                        test_loss = self.test()
                        print('the best test loss is:', test_loss)
                        break

    def test(self,is_save=True):
        self.encoder_decoder_model.eval()
        assert self.test_save_root is not None
        test_save_root = self.test_save_root
        clean_fold(test_save_root)
        batch_size = 4
        flag = True
        index = 1
        loss = 0
        count = 0
        while flag:
            dat, (index, b_cup) = sample(batch_size, data_type='test', index=index)
            frame_dat = normalization(dat, 255.0)
            in_frame_dat = frame_dat[:, :5]
            target_frame_dat = frame_dat[:, 5:]
            in_frame_dat = in_frame_dat.transpose(0, 1, 4, 2, 3)
            target_frame_dat = target_frame_dat.transpose(0, 1, 4, 2, 3)

            if torch.cuda.is_available():
                self.encoder_decoder_model = self.encoder_decoder_model.cuda()
                in_frame_dat = Variable(torch.from_numpy(in_frame_dat).float().cuda())
                target_frame_dat = Variable(torch.from_numpy(target_frame_dat).float().cuda())

            output_frames = self.encoder_decoder_model(in_frame_dat)
            output_frames = denormalization(output_frames.data.cpu().numpy(), 255.0).astype(np.float32) / 255.0
            target_frames = denormalization(target_frame_dat.data.cpu().numpy(), 255.0).astype(np.float32) / 255.0
            current_loss = np.mean(np.square(output_frames - target_frames))
            loss = loss + current_loss
            count = count + 1

            if is_save:
                output_frames = (output_frames * 255.0).astype(np.uint8)
                bat_ind = 0
                for ind in range(index - batch_size, index, 1):
                    save_fold = test_save_root + 'sample_' + str(ind) + '/'
                    clean_fold(save_fold)
                    for t in range(6, 16, 1):
                        imsave(save_fold + 'img_' + str(t) + '.png', output_frames[bat_ind, t - 6,0, :, :,])
                    bat_ind = bat_ind + 1
            if b_cup == 3:
                pass
            else:
                flag = False

        loss = loss / count
        return loss

    def validation(self,is_save=True):
        self.encoder_decoder_model.eval()
        assert self.validation_save_root is not None
        validation_save_root = self.validation_save_root
        clean_fold(validation_save_root)
        batch_size = 4
        flag = True
        index = 1
        loss = 0
        count = 0
        while flag:

            dat, (index, b_cup) = sample(batch_size, data_type='validation', index=index)
            frame_dat = normalization(dat, 255.0)
            in_frame_dat = frame_dat[:, :5]
            target_frame_dat = frame_dat[:, 5:]
            in_frame_dat = in_frame_dat.transpose(0, 1, 4, 2, 3)
            target_frame_dat = target_frame_dat.transpose(0, 1, 4, 2, 3)

            if torch.cuda.is_available():
                self.encoder_decoder_model = self.encoder_decoder_model.cuda()
                in_frame_dat = Variable(torch.from_numpy(in_frame_dat).float().cuda())
                target_frame_dat = Variable(torch.from_numpy(target_frame_dat).float().cuda())
            output_frames = self.encoder_decoder_model(in_frame_dat)
            output_frames = denormalization(output_frames.data.cpu().numpy(), 255.0).astype(np.float32) / 255.0
            target_frames = denormalization(target_frame_dat.data.cpu().numpy(), 255.0).astype(np.float32) / 255.0
            current_loss = np.mean(np.square(output_frames - target_frames))


            loss = current_loss + loss
            count = count + 1
            if is_save:
                output_frames = (output_frames*255.0).astype(np.uint8)
                bat_ind = 0
                for ind in range(index - batch_size, index, 1):
                    save_fold = validation_save_root + 'sample_' + str(ind) + '/'
                    clean_fold(save_fold)
                    for t in range(6, 16, 1):
                        imsave(save_fold + 'img_' + str(t) + '.png', output_frames[bat_ind, t - 6,0, :, :, ])
                    bat_ind = bat_ind + 1

            if b_cup == 3:
                pass
            else:
                flag = False
        loss = loss / count
        return loss

    def save_model(self):
        if not os.path.exists(os.path.split(self.info['MODEL_SAVE_PATH'])[0]):
            os.makedirs(os.path.split(self.info['MODEL_SAVE_PATH'])[0])
        torch.save(
            self.encoder_decoder_model,
            self.info['MODEL_SAVE_PATH']
        )
        print('model saved')

    def load_model(self):
        if not os.path.exists(os.path.split(self.info['MODEL_SAVE_PATH'])[0]):
            raise ('there are not model in ', os.path.split(self.info['MODEL_SAVE_PATH'])[0])
        self.encoder_decoder_model = torch.load(
            self.info['MODEL_SAVE_PATH']
        )
        print('model has been loaded')

if __name__ == '__main__':
    path = 'config/dec_ST_ConvLSTM_CIKM.yml'
    f = open(path)
    configuration = yaml.safe_load(f)

    encode_conv_rnn_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['ENCODE_CELLS']):
        param = get_cell_param(cell)
        param['m_channels'] = configuration['MODEL_NETS']['m_channels'][idx]
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
        param['m_channels'] = configuration['MODEL_NETS']['m_channels'][idx]
        decode_conv_rnn_cells.append(param)

    upsample_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['UPSAMPLE_CONVS']):
        if idx == len(configuration['MODEL_NETS']['UPSAMPLE_CONVS']) - 1:
            upsample_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['DECODE_PADDING'][idx], activate='tanh'))
        else:
            upsample_cells.append(get_conv_param(cell, padding=configuration['MODEL_NETS']['DECODE_PADDING'][idx]))

    output_conv_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['OUTPUT_CONV']):
        if idx == len(configuration['MODEL_NETS']['OUTPUT_CONV']) - 1:
            output_conv_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['OUTPUT_PADDING'][idx], activate='tanh'))
        else:
            output_conv_cells.append(get_conv_param(cell, padding=configuration['MODEL_NETS']['OUTPUT_PADDING'][idx]))

    encode_conv_m_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['M_ENCODE']):
        if idx == len(configuration['MODEL_NETS']['M_ENCODE']) - 1:
            encode_conv_m_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['M_ENCODE_PADDING'][idx], activate='tanh'))
        else:
            encode_conv_m_cells.append(get_conv_param(cell, padding=configuration['MODEL_NETS']['M_ENCODE_PADDING'][idx]))

    decode_conv_m_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['M_DECODE']):
        if idx == len(configuration['MODEL_NETS']['M_DECODE']) - 1:
            decode_conv_m_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['M_DECODE_PADDING'][idx], activate='tanh'))
        else:
            decode_conv_m_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['M_DECODE_PADDING'][idx]))

    encode_conv_reset_m_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['M_DECODE']):
        if idx == len(configuration['MODEL_NETS']['M_DECODE']) - 1:
            encode_conv_reset_m_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['M_DECODE_PADDING'][idx], activate='tanh'))
        else:
            encode_conv_reset_m_cells.append(get_conv_param(cell, padding=configuration['MODEL_NETS']['M_DECODE_PADDING'][idx]))

    decode_conv_reset_m_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['M_ENCODE']):
        if idx == len(configuration['MODEL_NETS']['M_ENCODE']) - 1:
            decode_conv_reset_m_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['M_ENCODE_PADDING'][idx], activate='tanh'))
        else:
            decode_conv_reset_m_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['M_ENCODE_PADDING'][idx]))

    encoder = Encoder_ST_ConvLSTM(
        conv_rnn_cells = encode_conv_rnn_cells,
        conv_cells = downsample_cells,
        conv_m_cells = encode_conv_m_cells,
        conv_reset_m_cells = encode_conv_reset_m_cells,
        info = configuration
    ).cuda()
    decoder = Decoder_ST_ConvLSTM(
        conv_rnn_cells = decode_conv_rnn_cells,
        conv_cells = upsample_cells,
        conv_m_cells = decode_conv_m_cells,
        conv_reset_m_cells = decode_conv_reset_m_cells,
        output_cells = output_conv_cells,
        info = configuration
    ).cuda()

    encoder_decoder_model = Encode_Decode_ST_ConvLSTM(
        encoder=encoder,
        decoder=decoder,
        info=configuration
    ).cuda()

    model = sequence_model(
        name=configuration['NAME'],
        encoder_decoder_model = encoder_decoder_model,
        info = configuration
    )
    model.load_model()
    model.test()
    # model.train()

