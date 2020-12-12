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

#### 参与贡献

1.  https://gitee.com/Excuses-j
2.  https://gitee.com/chu_zhixing
3.  https://gitee.com/liangzuan
