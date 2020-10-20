import cv2
import matplotlib.pyplot as plt
from data.CIKM.data_iterator import *


def distribution_analyze():
    test_dataset = CIKM_Datasets('/mnt/A/CIKM2017/CIKM_datasets/test/')
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)
    data_iter = iter(dataloader)
    count = 0
    imgs = {}
    count = 0
    for data in data_iter:
        in_data, tar_dat = data
        tar_dat = tar_dat.data.cpu().numpy()[0]
        for i in range(0,10):
            if count == 0:
                imgs[i] = tar_dat[i].reshape(-1)/4000
            else:
                imgs[i] = tar_dat[i].reshape(-1)/4000+imgs[i]

        count = count + 1
        print('complete is:',100.0*count/4000,'%')

    for i in range(10):
        n, bins, patches = plt.hist(x=imgs[i], bins='auto', color='#0504aa',
                                    alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Img'+str(i+1))
        plt.text(23, 45, r'$\mu=15, b=3$')
        maxfreq = n.max()
        # 设置y轴的上限
        plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        plt.show()


if __name__ == '__main__':
    distribution_analyze()

