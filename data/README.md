# RoadCondition

#### 介绍
2020CCF路况时空预测-宝可梦训练师团队比赛代码项目

地址：https://www.datafountain.cn/competitions/466

#### 项目依赖

python: 3
tensorflow-gpu: 1.14

CUDA Version: 10.0.130
CUDNN Version: 7.6.3

#### 比赛数据及文件名

    |-- data
        |-- raw_data
           | traffic
           | attr.txt
           | topo.txt
       

#### 总体流程

1. word2vec预训练得到linkid embedding
2. deepcnn模型finetune输出最终embedding
3. 训练lightgbm模型(人工特征工程+emdedding)得到预测结果
4. 基于训练数据的权重微调得到最终结果

#### 使用说明

从百度云盘下载镜像打包文件：baokemeng.tar
百度云盘地址: https://pan.baidu.com/s/14AX72R9zBfb04fPs6vHdWQ  密码: vqa3

把下载得到的baokemeng.tar放到您linux系统的某个目录下，然后在这个目录下，执行以下命令，把镜像打包文件导入到您本机中成为一个镜像。
sudo docker load -i baokemeng.tar

然后基于该镜像生成容器并开始跑代码的命令如下：
sudo nvidia-docker run -v /data:/data baokemeng:1 sh run.sh

#### lightGBM模型训练说明
##### ligthGBM模型训练可能出现的问题
lightGBM模型训练，会在最后执行。学习率设置的比较高，精度会下降。学习率设置的比较低，可能会导致训练轮数不足30轮就停止。正常训练的日志打印如下正常情况下，至少可以训练100轮以上。
```
[LightGBM] [Warning] num_threads is set with n_jobs=-1, nthread=-1 will be ignored. Current value: num_threads=-1
[LightGBM] [Warning] feature_fraction is set=0.8, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=0.8
[LightGBM] [Warning] min_data_in_leaf is set=20, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=20
[LightGBM] [Warning] bagging_fraction is set=0.8, subsample=1.0 will be ignored. Current value: bagging_fraction=0.8
[LightGBM] [Warning] bagging_freq is set=5, subsample_freq=0 will be ignored. Current value: bagging_freq=5
[LightGBM] [Warning] num_iterations is set=5000, num_boost_round=5000 will be ignored. Current value: num_iterations=5000
[LightGBM] [Warning] Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves
Training until validation scores don't improve for 10 rounds
[10]	training's multi_logloss: 0.452818	training's f1_score_sk: 0.20068	valid_1's multi_logloss: 0.412316	valid_1's f1_score_sk: 0.206637
[20]	training's multi_logloss: 0.403899	training's f1_score_sk: 0.507736	valid_1's multi_logloss: 0.367095	valid_1's f1_score_sk: 0.491168
[30]	training's multi_logloss: 0.382559	training's f1_score_sk: 0.564262	valid_1's multi_logloss: 0.347573	valid_1's f1_score_sk: 0.542965
[40]	training's multi_logloss: 0.371634	training's f1_score_sk: 0.584914	valid_1's multi_logloss: 0.337843	valid_1's f1_score_sk: 0.564284
[50]	training's multi_logloss: 0.365175	training's f1_score_sk: 0.595426	valid_1's multi_logloss: 0.332367	valid_1's f1_score_sk: 0.574643
[60]	training's multi_logloss: 0.360924	training's f1_score_sk: 0.60202	valid_1's multi_logloss: 0.329016	valid_1's f1_score_sk: 0.581289
[70]	training's multi_logloss: 0.357976	training's f1_score_sk: 0.606601	valid_1's multi_logloss: 0.326803	valid_1's f1_score_sk: 0.585167
[80]	training's multi_logloss: 0.355689	training's f1_score_sk: 0.610143	valid_1's multi_logloss: 0.325166	valid_1's f1_score_sk: 0.588474
[90]	training's multi_logloss: 0.353888	training's f1_score_sk: 0.613131	valid_1's multi_logloss: 0.323968	valid_1's f1_score_sk: 0.591058
[100]	training's multi_logloss: 0.352299	training's f1_score_sk: 0.615434	valid_1's multi_logloss: 0.322993	valid_1's f1_score_sk: 0.592842
[110]	training's multi_logloss: 0.350965	training's f1_score_sk: 0.616988	valid_1's multi_logloss: 0.32221	valid_1's f1_score_sk: 0.593499
[120]	training's multi_logloss: 0.349725	training's f1_score_sk: 0.618713	valid_1's multi_logloss: 0.321496	valid_1's f1_score_sk: 0.594902
[130]	training's multi_logloss: 0.348613	training's f1_score_sk: 0.620015	valid_1's multi_logloss: 0.320863	valid_1's f1_score_sk: 0.595516
[140]	training's multi_logloss: 0.347608	training's f1_score_sk: 0.621072	valid_1's multi_logloss: 0.320346	valid_1's f1_score_sk: 0.596219
[150]	training's multi_logloss: 0.346584	training's f1_score_sk: 0.62226	valid_1's multi_logloss: 0.319828	valid_1's f1_score_sk: 0.597182
[160]	training's multi_logloss: 0.345678	training's f1_score_sk: 0.622956	valid_1's multi_logloss: 0.319421	valid_1's f1_score_sk: 0.597712
[170]	training's multi_logloss: 0.344784	training's f1_score_sk: 0.623734	valid_1's multi_logloss: 0.319012	valid_1's f1_score_sk: 0.598815
[180]	training's multi_logloss: 0.34394	training's f1_score_sk: 0.62464	valid_1's multi_logloss: 0.318623	valid_1's f1_score_sk: 0.59917
[190]	training's multi_logloss: 0.343156	training's f1_score_sk: 0.625498	valid_1's multi_logloss: 0.318306	valid_1's f1_score_sk: 0.600021
[200]	training's multi_logloss: 0.342373	training's f1_score_sk: 0.626153	valid_1's multi_logloss: 0.317986	valid_1's f1_score_sk: 0.600126
[210]	training's multi_logloss: 0.341634	training's f1_score_sk: 0.626696	valid_1's multi_logloss: 0.317689	valid_1's f1_score_sk: 0.60038
[220]	training's multi_logloss: 0.340947	training's f1_score_sk: 0.627403	valid_1's multi_logloss: 0.317481	valid_1's f1_score_sk: 0.600974
[230]	training's multi_logloss: 0.340246	training's f1_score_sk: 0.6281	valid_1's multi_logloss: 0.317234	valid_1's f1_score_sk: 0.601043
[240]	training's multi_logloss: 0.339585	training's f1_score_sk: 0.628778	valid_1's multi_logloss: 0.317042	valid_1's f1_score_sk: 0.601515
[250]	training's multi_logloss: 0.338991	training's f1_score_sk: 0.629507	valid_1's multi_logloss: 0.316889	valid_1's f1_score_sk: 0.601844
[260]	training's multi_logloss: 0.338378	training's f1_score_sk: 0.630164	valid_1's multi_logloss: 0.316704	valid_1's f1_score_sk: 0.60209

```

如果出现这种情况，评审老师，可以把run.sh中，修改成如下，并运行这个文件。（之前的脚本运行过，就没有必要再次运行了）（本问题解释的不清楚的地方，可以联系队员朱思程，电话：13573131139）

```shell
# lightGBM学习率过低，训练轮数少于30轮的备用命令
python -u /data/code/zhusc/entry_lightgbm_sk_learn.py 1.0
```

#### 参与贡献

1.  https://gitee.com/Excuses-j
2.  https://gitee.com/chu_zhixing
3.  https://gitee.com/liangzuan
