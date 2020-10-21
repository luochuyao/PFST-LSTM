# PFST-LSTM

This is a Pytorhc implementation of PFST-LSTM, a recurrent model for precipitation nowcasting (radar echo extrapolation) as described in the following paper:

PFST-LSTM: a SpatioTemporal LSTM Model with Pseudo-flow Prediction for Precipitation Nowcasting, by Chuyao Luo, Xutao Li, Yunming Ye.

# Setup

Required python libraries: torch (>=1.4.0) + opencv + numpy + scipy (== 1.0.0) + jpype1.
Tested in ubuntu + nvidia 2080Ti with cuda (>=10.1).

# Datasets
We conduct experiments on CIKM AnalytiCup 2017 datasets:[CIKM_Radar](https://tianchi.aliyun.com/competition/entrance/231596/information)  

# Training
Use any '.py' script in the path of experiment/CIKM/ to train the models. To train the proposed model on the radar, we can simply run the experiment/CIKM/dec_PFST_ConvLSTM.py


You might want to change the parameter and setting, you can change the files in the path of experiment/CIKM/config/ for each model

The preprocess method and data root path can be modified in the data/data_iterator.py file

There are all trained models. You can download it following this address:[trained model](https://drive.google.com/drive/folders/1RB_V418msSLFSzplXYfzlnUZ7M79dt_l?usp=sharing)


# Evaluation
We give two approaches to evaluate our models. 


The first method is to check all predictions by running the java file in the path of CIKM_Eva/src (It is faster). You need to modify some information of path and make a .jar file to run

The second method is to run the evaluate.py in the path of evaluate/

# Prediction samples
5 frames are predicted given the last 10 frames.

![Prediction vislazation](https://github.com/luochuyao/PFST-LSTM/blob/master/evaluate/radar_res.png)

