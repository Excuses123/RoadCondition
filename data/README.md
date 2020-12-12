# RoadCondition

#### 介绍
2020CCF路况时空预测-宝可梦训练师团队比赛代码项目

地址：https://www.datafountain.cn/competitions/466

#### 项目依赖

python: 3
tensorflow-gpu: 1.14

CUDA Version: 10.0.130
CUDNN Version: 7.6.3

#### 总体流程

1. word2vec预训练得到linkid embedding
2. deepcnn模型finetune输出最终embedding
3. lightgbm人工特征工程+emdedding得到预测结果
4. 基于训练数据的权重微调得到最终结果

#### 使用说明

1.  sh ./image/run.sh

#### 参与贡献

1.  https://gitee.com/Excuses-j
2.  https://gitee.com/chu_zhixing
3.  https://gitee.com/liangzuan
