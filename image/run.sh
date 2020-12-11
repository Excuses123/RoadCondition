#!/bin/sh

#linkid映射及word2vec训练
python ./src/Excuses/run_w2v.py -user_path="./data/user_data/Excuses"

#tfrecords训练数据生成
python ./src/Excuses/run_TFrecords.py -user_path="./data/user_data/Excuses"

#模型训练预测及相关结果输出
python ./src/Excuses/run_cnn.py -user_path="./data/user_data/Excuses" -is_train=1 -is_pred=1 -save_emb=1