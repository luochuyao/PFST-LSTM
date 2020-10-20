import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from thop import profile
import time

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
rootPath = os.path.split(rootPath)[0]
sys.path.append(rootPath)
import yaml
import torch.optim as optim
from util.utils import *
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from model.PFSTEncodeDecode import *
from skimage.measure import compare_ssim
from skimage.transform import resize,rescale

from data.CIKM.data_iterator import *
from util.color_map import *
from util.utils import *
from model.varflow import *
import skimage
import cv2

def to_device(data):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return data.to(device,non_blocking = True)

class sequence_model(object):
    def __init__(self,
                 name,
                 encoder_decoder_model,
                 info,
                 ):
        self.name = name
        self.encoder_decoder_model = encoder_decoder_model
        self.info = info
        self.test_save_root = '/mnt/A/meteorological/2500_ref_seq/CIKM_dec_PFST_ConvLSTM_test/'
        self.validation_save_root = '/mnt/A/meteorological/2500_ref_seq/CIKM_dec_PFST_ConvLSTM_validation/'
        self.optimizer = optim.Adam(self.encoder_decoder_model.parameters(), lr=self.info['TRAIN']['LEARNING_RATE'])

    def train(self):
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
                in_frame_dat = Variable(torch.from_numpy(in_frame_dat).float().cuda())
                target_frame_dat = Variable(torch.from_numpy(target_frame_dat).float().cuda())

            self.optimizer.zero_grad()
            output = self.encoder_decoder_model(in_frame_dat)
            criterion = torch.nn.MSELoss()
            loss = criterion(output,target_frame_dat)
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

    def validation(self,is_save=True):
        self.encoder_decoder_model.eval()
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






    # def d_show(self):
    #     self.encoder_decoder_model.eval()
    #     batch_size = 4
    #     flag = True
    #     index = 1
    #     varflow_factory = VarFlowFactory(max_level=4, start_level=0, n1=2, n2=2, rho=2.8, alpha=1400,sigma=1.5)
    #
    #     while flag:
    #         dat, (index, b_cup) = sample(batch_size, data_type='test', index=index)
    #         if index<2361:
    #             continue
    #         print('index is:',str(index))
    #         b_id = 0
    #         frame_dat = normalization(dat, 255.0)
    #         in_frame_dat = frame_dat[:, :5]
    #         target_frame_dat = frame_dat[:, 5:]
    #         in_frame_dat = in_frame_dat.transpose(0, 1, 4, 2, 3)
    #         target_frame_dat = target_frame_dat.transpose(0, 1, 4, 2, 3)
    #
    #         if torch.cuda.is_available():
    #             self.encoder_decoder_model = self.encoder_decoder_model.cuda()
    #             in_frame_dat = Variable(torch.from_numpy(in_frame_dat).float().cuda())
    #             target_frame_dat = Variable(torch.from_numpy(target_frame_dat).float().cuda())
    #
    #         output_frames = self.encoder_decoder_model(in_frame_dat)
    #         output_frames_show,d = self.encoder_decoder_model.d_show(in_frame_dat)
    #         output_frames = output_frames.data.cpu().numpy()[:,:,0,:,:,]
    #         output_frames_show = output_frames_show.data.cpu().numpy()[:,:,0,:,:,]
    #         output_frames = denormalization(output_frames,255.0)
    #         output_frames_show = denormalization(output_frames_show,255.0)
    #         d1,d2 = d
    #         d1.extend(d2)
    #         d = d1
    #         t_length = len(d)
    #
    #         fig = plt.figure(figsize=(t_length,6))
    #         gs = GridSpec(6, t_length)
    #         new_dat = np.concatenate([dat[:,:5],output_frames[:,:,:,:,np.newaxis]],1)
    #
    #         for t in range(15):
    #             if t<14:
    #                 I1 = new_dat.transpose(0, 1, 4, 2, 3)[:,t,0]
    #                 I2 = new_dat.transpose(0, 1, 4, 2, 3)[:,t+1,0]
    #                 flow = varflow_factory.batch_calc_flow(I1=I1,I2=I2)
    #                 flow = np.concatenate((flow[:, :1, :, :], -flow[:, 1:, :, :]), axis=1)[b_id,:,:,:].transpose((1,2,0))
    #                 flow_img = flow_to_image(flow)
    #                 ax = plt.subplot2grid((6, t_length), (1, t))
    #                 ax.set_xticks([])
    #                 ax.set_yticks([])
    #                 ax.imshow(flow_img)
    #
    #             cur_output_frames = new_dat.transpose(0, 1, 4, 2, 3)[b_id,t,0]
    #             cur_output_frames = mapping(pixel_to_dBZ(cur_output_frames))
    #             ax = plt.subplot2grid((6, t_length), (0, t))
    #             ax.set_xticks([])
    #             ax.set_yticks([])
    #             ax.imshow(cur_output_frames)
    #
    #         l_flow = []
    #         for l in range(3):
    #             t_flow = []
    #             for t in range(t_length):
    #                 cur_layer_t_flow = d1[t][l].data.cpu().numpy()
    #                 batch_one_flow = cur_layer_t_flow[b_id]
    #                 for p in range(l+1):
    #                     batch_one_flow = np.stack([cv2.pyrUp(batch_one_flow[0], (101,101)),cv2.pyrUp(batch_one_flow[1], (101, 101))],0)
    #                 if l>0:
    #                     batch_one_flow = batch_one_flow[:,1:-1,1:-1]
    #                 batch_one_flow = batch_one_flow[np.newaxis,:,:,:]
    #                 batch_one_flow = np.concatenate((batch_one_flow[:, :1, :, :], batch_one_flow[:, 1:, :, :]), axis=1)[0,:,:,:].transpose((1,2,0))
    #                 t_flow.append(batch_one_flow)
    #                 if t == 0:
    #                     pass
    #                 else:
    #                     flow_img = flow_to_image(batch_one_flow)
    #                     ax = plt.subplot2grid((6, t_length), (l + 2, t - 1))
    #                     ax.set_xticks([])
    #                     ax.set_yticks([])
    #                     ax.imshow(flow_img)
    #             t_flow = np.array(t_flow)
    #             l_flow.append(t_flow)
    #
    #
    #
    #         l_flow = np.array(l_flow)[:,1:,:-1,:-1,:]
    #         groun_truth_imgs = new_dat[b_id]
    #         for t in range(14):
    #             I1_img = groun_truth_imgs[t]
    #             cur_img = I1_img.copy()
    #             next_imgs = []
    #             flow_lists_ = []
    #             for layer_idx in range(len(l_flow)):
    #                 if t<5:
    #                     cur_flow = l_flow[layer_idx,t]
    #                 else:
    #                     cur_flow = l_flow[len(l_flow)-1-layer_idx, t]
    #                 flow_lists_.append(cur_flow)
    #                 cur_flow = cur_flow.transpose((2,0,1))[np.newaxis,:,:,:]
    #                 if layer_idx==0:
    #                     cur_img = cur_img.transpose((2,0,1))[np.newaxis,:,:,:]
    #                 next_img = nearest_neighbor_advection(cur_img,-cur_flow)[0,0]
    #                 next_imgs.append(next_img)
    #             next_imgs = np.sum(np.array(next_imgs),0)
    #             I2_img = next_imgs[np.newaxis,:,:]
    #             I1_img = I1_img[np.newaxis,:,:,0]
    #             flow = varflow_factory.batch_calc_flow(I1=I1_img,I2=I2_img)[0]
    #
    #             flow = np.concatenate((flow[:1, :, :], -flow[1:, :, :]), axis=0)
    #             flow = flow.transpose((1,2,0))
    #
    #             cur_flow_img = flow_to_image(flow)
    #             ax = plt.subplot2grid((6, t_length), (-1, t))
    #             ax.set_xticks([])
    #             ax.set_yticks([])
    #             ax.imshow(cur_flow_img)
    #             pass
    #
    #         plt.savefig('flow.png')
    #         break

    def test(self,is_save=True):
        self.encoder_decoder_model.eval()
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
        print('test loss is:',str(loss))
        return loss

    def save_model(self):
        if not os.path.exists(os.path.split(self.info['MODEL_SAVE_PATH'])[0]):
            os.makedirs(os.path.split(configuration['MODEL_SAVE_PATH'])[0])
        torch.save(
            self.encoder_decoder_model,
            self.info['MODEL_SAVE_PATH']
        )
        print('model saved')

    def load_model(self):
        if not os.path.exists(os.path.split(configuration['MODEL_SAVE_PATH'])[0]):
            raise ('there are not model in ', os.path.split(configuration['MODEL_SAVE_PATH'])[0])
        self.encoder_decoder_model = torch.load(
            self.info['MODEL_SAVE_PATH']
        )
        print('model has been loaded')


if __name__ == '__main__':
    path = 'config/dec_PFST_ConvLSTM_CIKM.yml'
    f = open(path)
    configuration = yaml.safe_load(f)
    print(configuration)

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

    encode_conv_m_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['M_ENCODE']):
        if idx == len(configuration['MODEL_NETS']['M_ENCODE']) - 1:
            encode_conv_m_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['M_ENCODE_PADDING'][idx], activate='tanh'))
        else:
            encode_conv_m_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['M_ENCODE_PADDING'][idx]))

    encode_conv_reset_m_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['M_DECODE']):
        if idx == len(configuration['MODEL_NETS']['M_DECODE']) - 1:
            encode_conv_reset_m_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['M_DECODE_PADDING'][idx], activate='tanh'))
        else:
            encode_conv_reset_m_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['M_DECODE_PADDING'][idx]))

    decode_conv_rnn_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['DECODE_CELLS']):
        param = get_cell_param(cell)
        param['m_channels'] = configuration['MODEL_NETS']['m_channels'][idx]
        decode_conv_rnn_cells.append(param)

    output_conv_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['OUTPUT_CONV']):
        if idx == len(configuration['MODEL_NETS']['OUTPUT_CONV']) - 1:
            output_conv_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['OUTPUT_PADDING'][idx], activate='tanh'))
        else:
            output_conv_cells.append(get_conv_param(cell, padding=configuration['MODEL_NETS']['OUTPUT_PADDING'][idx]))

    decode_conv_m_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['M_DECODE']):
        if idx == len(configuration['MODEL_NETS']['M_DECODE']) - 1:
            decode_conv_m_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['M_DECODE_PADDING'][idx], activate='tanh'))
        else:
            decode_conv_m_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['M_DECODE_PADDING'][idx]))

    decode_conv_reset_m_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['M_ENCODE']):
        if idx == len(configuration['MODEL_NETS']['M_ENCODE']) - 1:
            decode_conv_reset_m_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['M_ENCODE_PADDING'][idx], activate='tanh'))
        else:
            decode_conv_reset_m_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['M_ENCODE_PADDING'][idx]))

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
            input_cells.append(get_conv_param(cell, padding=configuration['MODEL_NETS']['INPUT_PADDING'][idx], activate='tanh'))
        else:
            input_cells.append(get_conv_param(cell, padding=configuration['MODEL_NETS']['INPUT_PADDING'][idx]))

    encoder = Encoder_PFST_ConvLSTM(
        conv_rnn_cells=encode_conv_rnn_cells,
        conv_cells=downsample_cells,
        conv_m_cells=encode_conv_m_cells,
        conv_reset_m_cells=encode_conv_reset_m_cells,
        info=configuration
    ).cuda()

    decoder = Decoder_PFST_ConvLSTM(
        conv_rnn_cells = decode_conv_rnn_cells,
        conv_cells = upsample_cells,
        conv_m_cells = decode_conv_m_cells,
        conv_reset_m_cells = decode_conv_reset_m_cells,
        output_cells = output_conv_cells,
        input_cells = input_cells,
        info=configuration
    ).cuda()

    encoder_decoder_model = Encode_Decode_PFST_ConvLSTM(
        encoder=encoder,
        decoder=decoder,
        info=configuration
    ).cuda()

    model = sequence_model(
        name=configuration['NAME'],
        encoder_decoder_model = encoder_decoder_model,
        info = configuration
    )
    # model.train()
    model.load_model()
    # model.test()

