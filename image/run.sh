#!/bin/sh

starttime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);

#执行程序
#word2vec训练
python ./src/Excuses/run_w2v.py -user_path="./data/user_data/Excuses"

#tfrecords训练数据生成
python ./src/Excuses/run_TFrecords.py -user_path="./data/user_data/Excuses"

#模型训练预测及相关结果输出
python ./src/Excuses/run_cnn.py -user_path="./data/user_data/Excuses" -is_train=1 -is_pred=1 -save_emb=1

endtime=`date +'%Y-%m-%d %H:%M:%S'`
end_seconds=$(date --date="$endtime" +%s);
echo "Excuses本次运行时间： "$((end_seconds-start_seconds))"s"
