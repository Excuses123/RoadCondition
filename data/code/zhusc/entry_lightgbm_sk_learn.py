# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import f1_score
import pickle
import os
import logging
import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
from scipy.optimize import minimize
from functools import partial
import feature_engineer_all as fea
import sys


def init_pandas_show():
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)

    # 设置value的显示长度为100，默认为50
    pd.set_option('max_colwidth', 120)


def init_logger():
    """
    初始化logger类，使它能够同时往文件与控制台上输出日志
    :return:
    """

    # 配置日志信息
    logging.basicConfig(level=logging.DEBUG, filename='myapp.log',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    # 定义一个Handler打印INFO及以上级别的日志到sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # 设置日志打印格式
    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    # 将定义好的console日志handler添加到root logger
    logging.getLogger('').addHandler(console);

    return logging;


def findAllIndex(str, goal_char):
    """
    查找str中所有字符为goal_char的元素下标
    :param str:
    :param goal_char:
    :return:
    """
    str_len = len(str);
    idx = -1;
    res = [];
    while idx < str_len:
        idx = str.find(goal_char, idx + 1);
        if idx == -1:
            return res;
        else:
            res.append(idx);

    return res;


def current_history_lukuang_2_row(str):
    '''
    分解前 ==>
    总体格式：linkid label current_slice_id future_slice_id;recent_feature;history_feature
    348288 2 666 687;662:28.80,14.90,1,1 663:25.60,25.60,1,4 664:19.30,28.10,2,1 665:21.20,28.10,2,1 666:19.30,23.80,2,2;687:31.70,19.50,1,4 688:20.00,17.30,2,5 689:26.40,21.60,1,5 690:26.30,23.80,1,5 691:33.10,21.50,1,4;687:32.70,9.80,1,1 688:32.20,10.10,1,1 689:34.70,30.40,1,2 690:33.00,30.40,1,2 691:30.30,30.40,1,2;687:28.40,23.70,1,4 688:28.10,27.10,1,2 689:28.10,27.30,1,1 690:25.00,23.70,1,3 691:19.60,24.40,2,5;687:31.30,27.80,1,6 688:29.00,27.70,1,7 689:21.90,26.70,2,6 690:27.60,22.00,1,4 691:28.20,21.50,1,3

    分解后 ==>
    总体格式：
    link_id label current_slice_id future_slice_id;recent_feature;history_feature
    348288 2     666              687;662:28.80,14.90,1,1 663:25.60,25.60,1,4 664:19.30,28.10,2,1 665:21.20,28.10,2,1 666:19.30,23.80,2,2;

    history_feature
    687:31.70,19.50,1,4 688:20.00,17.30,2,5 689:26.40,21.60,1,5 690:26.30,23.80,1,5 691:33.10,21.50,1,4;
    687:32.70,9.80,1,1 688:32.20,10.10,1,1 689:34.70,30.40,1,2 690:33.00,30.40,1,2 691:30.30,30.40,1,2;
    687:28.40,23.70,1,4 688:28.10,27.10,1,2 689:28.10,27.30,1,1 690:25.00,23.70,1,3 691:19.60,24.40,2,5;
    687:31.30,27.80,1,6 688:29.00,27.70,1,7 689:21.90,26.70,2,6 690:27.60,22.00,1,4 691:28.20,21.50,1,3
    '''

    blank_idx_list = findAllIndex(str, ' ');

    res = [];
    link_id = str[0:blank_idx_list[0]];
    label = str[blank_idx_list[0] + 1: blank_idx_list[1]];
    current_slice_id = str[blank_idx_list[1] + 1: blank_idx_list[2]];

    # 按照";"整体拆分
    rest_str = str[blank_idx_list[2] + 1: len(str)];
    fenhao_idx_list = findAllIndex(rest_str, ';');
    future_slice_id = rest_str[0:fenhao_idx_list[0]];
    recent_feature = rest_str[fenhao_idx_list[0] + 1:fenhao_idx_list[1]];
    history_feature = rest_str[fenhao_idx_list[1] + 1:len(rest_str)];

    # 总体格式：linkid label current_slice_id future_slice_id;recent_feature;history_feature
    # return pd.Series([link_id, label, current_slice_id, future_slice_id, recent_feature, history_feature]);
    return link_id, label, current_slice_id, future_slice_id, recent_feature, history_feature;


def current_history_lukuang_2_row_method02(str):
    """

    :param str:
    :return: 返回多个计算后的变量
    """
    '''
    分解前 ==>
    总体格式：linkid label current_slice_id future_slice_id;recent_feature;history_feature
    348288 2 666 687;662:28.80,14.90,1,1 663:25.60,25.60,1,4 664:19.30,28.10,2,1 665:21.20,28.10,2,1 666:19.30,23.80,2,2;687:31.70,19.50,1,4 688:20.00,17.30,2,5 689:26.40,21.60,1,5 690:26.30,23.80,1,5 691:33.10,21.50,1,4;687:32.70,9.80,1,1 688:32.20,10.10,1,1 689:34.70,30.40,1,2 690:33.00,30.40,1,2 691:30.30,30.40,1,2;687:28.40,23.70,1,4 688:28.10,27.10,1,2 689:28.10,27.30,1,1 690:25.00,23.70,1,3 691:19.60,24.40,2,5;687:31.30,27.80,1,6 688:29.00,27.70,1,7 689:21.90,26.70,2,6 690:27.60,22.00,1,4 691:28.20,21.50,1,3

    分解后 ==>
    总体格式：
    link_id label current_slice_id future_slice_id;recent_feature;history_feature
    348288 2     666              687;662:28.80,14.90,1,1 663:25.60,25.60,1,4 664:19.30,28.10,2,1 665:21.20,28.10,2,1 666:19.30,23.80,2,2;

    history_feature
    687:31.70,19.50,1,4 688:20.00,17.30,2,5 689:26.40,21.60,1,5 690:26.30,23.80,1,5 691:33.10,21.50,1,4;
    687:32.70,9.80,1,1 688:32.20,10.10,1,1 689:34.70,30.40,1,2 690:33.00,30.40,1,2 691:30.30,30.40,1,2;
    687:28.40,23.70,1,4 688:28.10,27.10,1,2 689:28.10,27.30,1,1 690:25.00,23.70,1,3 691:19.60,24.40,2,5;
    687:31.30,27.80,1,6 688:29.00,27.70,1,7 689:21.90,26.70,2,6 690:27.60,22.00,1,4 691:28.20,21.50,1,3
    '''
    # print('str = ' + str);
    blank_idx_list = findAllIndex(str, ' ');

    res = [];
    link_id = str[0:blank_idx_list[0]];
    label = str[blank_idx_list[0] + 1: blank_idx_list[1]];
    current_slice_id = str[blank_idx_list[1] + 1: blank_idx_list[2]];

    # 按照";"整体拆分
    rest_str = str[blank_idx_list[2] + 1: len(str)];
    fenhao_idx_list = findAllIndex(rest_str, ';');
    future_slice_id = rest_str[0:fenhao_idx_list[0]];
    recent_feature = rest_str[fenhao_idx_list[0] + 1:fenhao_idx_list[1]];
    history_feature = rest_str[fenhao_idx_list[1] + 1:len(rest_str)];

    # 总体格式：linkid label current_slice_id future_slice_id;recent_feature;history_feature
    # return pd.Series([link_id, label, current_slice_id, future_slice_id, recent_feature, history_feature]);

    '''
    recent_feature的拆分解析
    340:33.50,38.50,1,7 341:32.00,36.80,1,7 342:31.90,36.60,1,8 343:30.20,35.10,1,7 344:31.60,36.60,1,5;
    '''
    recent_feature_dict = nfeatrue_2_multi_vars_dict(recent_feature, 0);

    '''
    history_feature的拆分处理
    总体格式：linkid label current_slice_id future_slice_id;recent_feature;history_feature

    history_feature
    369:33.30,36.70,1,2 370:33.30,34.00,1,3 371:33.30,34.10,1,3 372:33.30,34.10,1,3 373:33.80,36.20,1,4;
    369:41.80,44.10,1,1 370:33.30,35.00,1,2 371:34.20,35.70,1,4 372:33.90,37.10,1,5 373:33.40,36.70,1,6;
    369:33.90,35.60,1,5 370:32.40,32.10,1,3 371:28.60,29.10,1,2 372:30.10,33.30,1,2 373:33.70,36.50,1,5;
    369:35.30,45.60,1,1 370:25.60,36.50,1,2 371:30.50,36.00,1,3 372:33.50,36.50,1,2 373:30.30,30.80,1,3    
    '''
    # history_feature = row['history_feature'];
    date_idx = [-28, -21, -14, -7];
    history_feature_dict = {};
    for idx, history_feature_str in enumerate(history_feature.split(';')):
        tmp = nfeatrue_2_multi_vars_dict(history_feature_str, date_idx[idx])
        # row_feature_df = row_feature_df.append(tmp, ignore_index=True);
        history_feature_dict.update(tmp);

    return link_id, label, current_slice_id, future_slice_id, recent_feature_dict, history_feature_dict;


def nfeatrue2df(feature_str, date_id):
    """
    把若干个时间点对应的路况信息组成的str，拆分开，返回一个dataframe
    :param feature_str:
    :return:
    """

    list_tmp = feature_str.split(' ');
    rows_list = []
    for one_time_road_feature_str in list_tmp:
        # 232:29.80,32.40,1,4
        row_dict = {};
        row_dict['date_id'] = date_id;

        maohao_idx = one_time_road_feature_str.find(':');

        # 'slice_id', 'road_speed', 'eta_speed', 'label', 'road_calc_car_num'
        row_dict['slice_id'] = one_time_road_feature_str[0:maohao_idx];

        rest_str = one_time_road_feature_str[maohao_idx + 1:len(one_time_road_feature_str)];
        rest_list = rest_str.split(',');

        # 'road_speed', 'eta_speed', 'label', 'road_calc_car_num'
        row_dict['road_speed'] = float(rest_list[0]);
        row_dict['eta_speed'] = float(rest_list[1]);
        row_dict['past_label'] = rest_list[2];
        row_dict['road_calc_car_num'] = float(rest_list[3]);

        rows_list.append(row_dict);

    df = pd.DataFrame(rows_list,
                      columns=['date_id', 'slice_id', 'road_speed', 'eta_speed', 'past_label', 'road_calc_car_num']);
    return df;


def nfeatrue_2_multi_vars_dict(feature_str, date_id):
    """
    把若干个时间点对应的路况信息组成的str，拆分开，返回一个dataframe
    :param feature_str:
    :param date_id: 描述当前字符串对应的路况信息，相对于当前slice_id的时间;
    :return: 返回一个list
    """

    list_tmp = feature_str.split(' ');
    part_suffix = '_' + str(date_id) + '_';
    row_dict = {};
    for idx, one_time_road_feature_str in enumerate(list_tmp):
        # 字典键名称的完整前缀;
        suffix = part_suffix + str(idx);

        # 232:29.80,32.40,1,4
        # row_dict['date_id'] = date_id;
        maohao_idx = one_time_road_feature_str.find(':');

        # 'slice_id', 'road_speed', 'eta_speed', 'label', 'road_calc_car_num'
        row_dict['slice_id' + suffix] = int(one_time_road_feature_str[0:maohao_idx]);

        # 'road_speed', 'eta_speed', 'label', 'road_calc_car_num'
        rest_str = one_time_road_feature_str[maohao_idx + 1:len(one_time_road_feature_str)];
        rest_list = rest_str.split(',');
        row_dict['road_speed' + suffix] = float(rest_list[0]);
        row_dict['eta_speed' + suffix] = float(rest_list[1]);
        row_dict['label' + suffix] = int(rest_list[2]);
        row_dict['calc_car_num' + suffix] = float(rest_list[3]);

    return row_dict;


def featrue2df(row):
    """

    :param row: 一行，当前一个link对应的当前路况，历史路况信息
    :return:
    """

    '''
    print(row['recent_feature'] + '\r\n\r\n' + row['history_feature']);

    recent_feature的拆分解析
    recent_feature;
    340:33.50,38.50,1,7 341:32.00,36.80,1,7 342:31.90,36.60,1,8 343:30.20,35.10,1,7 344:31.60,36.60,1,5;
    '''
    recent_feature = row['recent_feature'];
    row_feature_df = nfeatrue2df(recent_feature, 0);
    # print(feature_df);

    '''
    history_feature的拆分处理
    总体格式：linkid label current_slice_id future_slice_id;recent_feature;history_feature

    history_feature
    369:33.30,36.70,1,2 370:33.30,34.00,1,3 371:33.30,34.10,1,3 372:33.30,34.10,1,3 373:33.80,36.20,1,4;
    369:41.80,44.10,1,1 370:33.30,35.00,1,2 371:34.20,35.70,1,4 372:33.90,37.10,1,5 373:33.40,36.70,1,6;
    369:33.90,35.60,1,5 370:32.40,32.10,1,3 371:28.60,29.10,1,2 372:30.10,33.30,1,2 373:33.70,36.50,1,5;
    369:35.30,45.60,1,1 370:25.60,36.50,1,2 371:30.50,36.00,1,3 372:33.50,36.50,1,2 373:30.30,30.80,1,3
    '''
    history_feature = row['history_feature'];
    date_idx = [-28, -21, -14, -7];
    for idx, history_feature_str in enumerate(history_feature.split(';')):
        tmp = nfeatrue2df(history_feature_str, date_idx[idx])
        row_feature_df = row_feature_df.append(tmp, ignore_index=True);

    return row_feature_df;


def basic_001():
    starttime = datetime.datetime.now()
    print('hello lu kuang ....');

    init_pandas_show();
    dir = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/'

    ''' '''
    attr_path = dir + "20201012150828attr.txt"
    attr_df = pd.read_csv(attr_path, sep='	');
    print('attr_df.head(5) =\r\n', attr_df.head(5));

    topo_path = dir + "20201012151101topo.txt";
    topo_df = pd.read_csv(topo_path, sep='	');
    print('topo_df.head(5) =\r\n', topo_df.head(5));

    topo_df.xiayou_link_ids = topo_df.xiayou_link_ids.str.split(',', expand=False);
    print('topo_df.head(5) =\r\n', topo_df.head(5));
    print(topo_df.xiayou_link_ids[2][0])

    # 处理某一个的路况数据
    current_history_lukuang_path = dir + '20201015181036traffic-fix/20190701.txt';

    # 整体读取成仅有一列的df
    current_history_lukuang = pd.read_csv(current_history_lukuang_path, sep='\t', header=None);
    # print(current_history_lukuang.head(3));

    current_history_road_df = current_history_lukuang.iloc[0:3, 0].apply(current_history_lukuang_2_row).apply(
        pd.Series);
    current_history_road_df.columns = ['link_id', 'label', 'current_slice_id', 'future_slice_id', 'recent_feature',
                                       'history_feature'];
    print(current_history_road_df.head(1));

    submission_list = [];
    for index, road_row in current_history_road_df.iterrows():
        # print(row["c1"], row["c2"])
        # 对其中一行，拆分路况字符串成一个dataframe
        # feature_df = featrue2df(current_history_road_df.loc[current_history_road_df.index[0], :]);
        link_id = road_row.link_id;
        current_slice_id = road_row.current_slice_id;
        feature_df = featrue2df(road_row);
        print('feature_df =\r\n', feature_df);

        # 对一行数据，一个link_id预测的baseline
        # road_row = current_history_road_df.loc[current_history_road_df.index[0], :];
        future_slice_id = road_row.future_slice_id;
        history_road_speed_mean = feature_df.loc[feature_df.slice_id == future_slice_id, 'road_speed'].mean();
        current_speed_mean = feature_df.loc[feature_df.date_id == 0, 'road_speed'].mean();
        road_speed_mean = history_road_speed_mean * 0.3 + current_speed_mean * 0.7;
        if road_speed_mean > 36:
            future_lable = 0;
        elif road_speed_mean > 26:
            future_lable = 1;
        else:
            future_lable = 2;

        # linkid label current_slice_id future_slice_id;recent_feature;history_feature
        submission_list.append({'link_id': link_id,
                                'label': future_lable,
                                'current_slice_id': current_slice_id,
                                'future_slice_id': future_slice_id});

        print('future_lable =', future_lable);
        if index >= 2:
            break;

    submission_df = pd.DataFrame(submission_list);
    new_columns_order_list = ['link_id', 'current_slice_id', 'future_slice_id', 'label'];
    submission_df = submission_df.reindex(columns=new_columns_order_list);

    submission_df.to_csv('submission.csv', index=False);
    endtime = datetime.datetime.now();
    print('total run time:' + str((endtime - starttime).seconds));


# def xgbt_model():
#     pass;
def split_2_struct_data(current_history_road, attr_df, save_path):
    """
    之前存在的问题，行数太多，会OOM
    :param current_history_road:
    :param attr_df: 路段的属性集合
    :return:
    """

    row_num = current_history_road.shape[0];
    batch_size = 10000;
    total_batch = math.ceil(row_num / batch_size);
    # res_df = pd.DataFrame({});

    start_row_id = 0;
    end_row_id = 0;
    for batch_id in range(total_batch + 1):
        print('batch_id = ' + str(batch_id) + ', total_batch = ' + str(total_batch));
        start_row_id = batch_id * batch_size;

        # 右区间远大于总行数，没有问题的。
        end_row_id = start_row_id + batch_size;

        # 只选取n行，做拆分列
        # current_history_road_df = current_history_lukuang.iloc[0:20, 0].apply(current_history_lukuang_2_row_method02).apply( pd.Series);
        current_history_road_df = current_history_road.iloc[start_row_id: end_row_id, 0].apply(
            current_history_lukuang_2_row_method02).apply(
            pd.Series);

        # 说明此时是最末一轮，且查询出来为空的情况，结束循环;
        if len(current_history_road_df) == 0:
            break;

        current_history_road_df.columns = ['link_id', 'label', 'current_slice_id', 'future_slice_id', 'recent_feature',
                                           'history_feature'];
        # current_history_road_df
        # print(current_history_road_df.head(3));

        # 把recent_feature、history_feature对应的字典展开成多列
        recent_feature_df = current_history_road_df.recent_feature.apply(pd.Series);
        history_feature_df = current_history_road_df.history_feature.apply(pd.Series);
        current_history_road_df = pd.concat([current_history_road_df, recent_feature_df, history_feature_df], axis=1);
        current_history_road_df = current_history_road_df.drop(columns=['recent_feature', 'history_feature']);

        # print(current_history_road_df.head(3));
        # print(current_history_road_df.columns);

        # join上路段属性信息
        current_history_road_df[['link_id']] = current_history_road_df[['link_id']].astype('int64');
        current_history_attr_road_df = pd.merge(current_history_road_df, attr_df, on=['link_id'], how='left');
        # print('current_history_attr_road_df =\n', current_history_attr_road_df.head(3));

        # 保存结果
        # res_df = res_df.append(copy.deepcopy(current_history_attr_road_df), ignore_index=True)

        if batch_id == 0:
            current_history_attr_road_df.to_csv(save_path, index=False, mode='a', header=True);
        else:
            current_history_attr_road_df.to_csv(save_path, index=False, mode='a', header=False)

        # 删除临时变量，腾出空间
        del current_history_attr_road_df;
        del current_history_road_df;
        del history_feature_df;
        del recent_feature_df;

    # return res_df;


def split_2_struct_data_02(current_history_road, attr_df, save_path):
    """
    之前存在的问题，行数太多，会OOM
    :param current_history_road:
    :param attr_df: 路段的属性集合
    :return:
    """

    row_num = current_history_road.shape[0];
    batch_size = 20000;
    if MODE == 'DEV':
        batch_size = 100;
    total_batch = math.ceil(row_num / batch_size);
    # res_df = pd.DataFrame({});

    start_row_id = 0;
    end_row_id = 0;
    for batch_id in range(total_batch + 1):
        print('batch_id = ' + str(batch_id) + ', total_batch = ' + str(total_batch));
        start_row_id = batch_id * batch_size;

        # 右区间远大于总行数，没有问题的。
        end_row_id = start_row_id + batch_size;

        # 只选取n行，做拆分列 current_history_road_df = current_history_lukuang.iloc[0:20, 0].apply(
        # current_history_lukuang_2_row_method02).apply( pd.Series);
        current_history_road_df = current_history_road.iloc[start_row_id: end_row_id, 0].apply(
            current_history_lukuang_2_row_method02).apply(
            pd.Series);

        # 说明此时是最末一轮，且查询出来为空的情况，结束循环;
        if len(current_history_road_df) == 0:
            break;

        current_history_road_df.columns = ['link_id', 'label', 'current_slice_id', 'future_slice_id', 'recent_feature',
                                           'history_feature'];
        # current_history_road_df
        # print(current_history_road_df.head(3));

        # 把recent_feature、history_feature对应的字典展开成多列
        recent_feature_df = current_history_road_df.recent_feature.apply(pd.Series);
        history_feature_df = current_history_road_df.history_feature.apply(pd.Series);
        current_history_road_df = pd.concat([current_history_road_df, recent_feature_df, history_feature_df], axis=1);
        current_history_road_df = current_history_road_df.drop(columns=['recent_feature', 'history_feature']);

        # 新增time_diff这一特征;
        current_history_road_df[['current_slice_id']] = current_history_road_df[['current_slice_id']].astype(int);
        current_history_road_df[['future_slice_id']] = current_history_road_df[['future_slice_id']].astype(int);
        current_history_road_df['time_diff'] = current_history_road_df['future_slice_id'] - current_history_road_df[
            'current_slice_id'];

        # print(current_history_road_df.head(3));
        # print(current_history_road_df.columns);

        # join上路段属性信息
        current_history_road_df[['link_id']] = current_history_road_df[['link_id']].astype('int64');
        current_history_attr_road_df = pd.merge(current_history_road_df, attr_df, on=['link_id'], how='left');
        # print('current_history_attr_road_df =\n', current_history_attr_road_df.head(3));

        # 保存结果
        # res_df = res_df.append(copy.deepcopy(current_history_attr_road_df), ignore_index=True)

        if batch_id == 0:
            current_history_attr_road_df.to_csv(save_path, index=False, mode='a', header=True);
        else:
            current_history_attr_road_df.to_csv(save_path, index=False, mode='a', header=False)

        # 删除临时变量，腾出空间
        del current_history_attr_road_df;
        del current_history_road_df;
        del history_feature_df;
        del recent_feature_df;


def model_meric(y_test, y_pred):
    y_test.loc[y_test.label == 4, :] = 3;
    y_pred[y_pred == 4] = 3;
    f1_score_arr = f1_score(y_test.values, y_pred, average=None);
    class_type_np = np.sort(np.unique(y_test.values), 0);
    weight_arr = [0.2, 0.2, 0.6];
    f1_score_val = 0;
    for class_type_idx, class_type in enumerate(class_type_np):
        f1_score_val = f1_score_val + weight_arr[class_type - 1] * f1_score_arr[class_type_idx];

    print('f1_score_val =', f1_score_val);
    logging.info('f1_score_val =', f1_score_val)
    return f1_score_val;


def model_meric2(y_test, y_pred):
    y_test[y_test == 4] = 3;
    y_pred[y_pred == 4] = 3;
    f1_score_arr = f1_score(y_test, y_pred, average=None);
    class_type_np = np.sort(np.unique(y_test), 0);
    weight_arr = [0.2, 0.2, 0.6];
    f1_score_val = 0;
    for class_type_idx, class_type in enumerate(class_type_np):
        f1_score_val = f1_score_val + weight_arr[class_type - 1] * f1_score_arr[class_type_idx];

    return f1_score_val;


def model_meric3(y_test, y_pred):
    """

    :param y_test: 是个m个元素的1维numpy
    :param y_pred: 是个m*n的numpy n为分类的种类
    :return:
    """

    pred_class = np.argmax(y_pred, axis=1);

    y_test[y_test == 4] = 3;
    pred_class[pred_class == 4] = 3;

    f1_score_arr = f1_score(y_test, pred_class, average=None);
    class_type_np = np.sort(np.unique(y_test), 0);
    weight_arr = [0, 0.2, 0.2, 0.6];  # 依次对应类别0(不存在), 类别1，类别2，类别3的权重
    f1_score_val = 0;
    for class_type_idx, class_type in enumerate(class_type_np):
        f1_score_val = f1_score_val + weight_arr[int(class_type)] * f1_score_arr[class_type_idx];

    return f1_score_val;


def eval_error(preds, dMatrix):
    """
    自定义模型评估函数
    :param preds: 预测结果
    :param dMatrix: watchlist中的训练集与测试集
    :return:
    """
    labels = dMatrix.get_label()
    labels_copy = labels.copy();

    pred_class = np.argmax(preds, axis=1);
    # pred_class = [pred_max_idx + 1 for value in pred_max_idx];

    labels_copy[labels_copy == 4] = 3;
    pred_class[pred_class == 4] = 3;

    f1_score_arr = f1_score(labels_copy, pred_class, average=None);
    class_type_np = np.sort(np.unique(labels_copy), 0);
    weight_arr = [0, 0.2, 0.2, 0.6];  # 依次对应类别0(不存在), 类别1，类别2，类别3的权重
    f1_score_val = 0;
    for class_type_idx, class_type in enumerate(class_type_np):
        f1_score_val = f1_score_val + weight_arr[int(class_type)] * f1_score_arr[class_type_idx];

    # return a pair metric_name, result. The metric name must not contain a colon (:) or a space
    return 'my-error', -f1_score_val


def f1_score_eval(preds, valid_df):
    labels = valid_df.get_label()
    preds = np.argmax(preds.reshape(3, -1), axis=0)
    # preds[preds == 4] = 3;
    f1_score_arr = f1_score(y_true=labels, y_pred=preds, average=None)
    # f1_score
    # scores = scores[0] * 0.2 + scores[1] * 0.2 + scores[2] * 0.6

    class_type_np = np.sort(np.unique(labels), 0);
    weight_arr = [0.2, 0.2, 0.6];  # 依次对应类别0(不存在), 类别1，类别2，类别3的权重
    f1_score_val = 0;
    # 为了适应lightGBM的label从0开始做的调整，即类别1==> class=0; 类别2==>class=1; 类别3==>class=2;
    for class_type_idx, class_type in enumerate(class_type_np):
        f1_score_val = f1_score_val + weight_arr[int(class_type)] * f1_score_arr[class_type_idx];

    return 'f1_score', f1_score_val, True


def f1_score_eval_final_test(preds, valid_df):
    labels = valid_df.get_label()
    preds = np.argmax(preds, axis=1)
    # preds[preds == 4] = 3;
    f1_score_arr = f1_score(y_true=labels, y_pred=preds, average=None)
    # f1_score
    # scores = scores[0] * 0.2 + scores[1] * 0.2 + scores[2] * 0.6

    class_type_np = np.sort(np.unique(labels), 0);
    weight_arr = [0.2, 0.2, 0.6];  # 依次对应类别0(不存在), 类别1，类别2，类别3的权重
    f1_score_val = 0;
    for class_type_idx, class_type in enumerate(class_type_np):
        f1_score_val = f1_score_val + weight_arr[int(class_type)] * f1_score_arr[class_type_idx];

    print('next print prediction class report ......');
    # target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4']
    target_names = ['class 1', 'class 2', 'class 3'];
    # if (labels == 0).sum() != 0:
    #     target_names.append('class 0');
    # target_names.extend(['class 1', 'class 2', 'class 3'])
    # if (labels == 4).sum() != 0:
    #     target_names.append('class 4');
    print(classification_report(labels, preds, target_names=target_names))
    return 'f1_score', f1_score_val, True


def f1_score_eval_final_test_sk(y_true, y_pred):
    f1_score_arr = f1_score(y_true=y_true, y_pred=y_pred, average=None)
    class_type_np = np.sort(np.unique(y_true), 0);
    weight_arr = [0.2, 0.2, 0.6];  # 依次对应类别0(不存在), 类别1，类别2，类别3的权重
    f1_score_val = 0;
    # 为了适应lightGBM的label从0开始做的调整，即类别1==> class=0; 类别2==>class=1; 类别3==>class=2;
    for class_type_idx, class_type in enumerate(class_type_np):
        f1_score_val = f1_score_val + weight_arr[int(class_type)] * f1_score_arr[class_type_idx];

    print('list scores as follows ...')
    target_names = ['class 1', 'class 2', 'class 3'];
    # print(classification_report(labels, preds, target_names=target_names))
    print(classification_report(y_true, y_pred, target_names=target_names))
    return 'f1_score_sk', f1_score_val, True


def f1_score_eval_sk(y_true, y_pred):
    y_pred = np.argmax(y_pred.reshape(3, -1), axis=0)
    f1_score_arr = f1_score(y_true=y_true, y_pred=y_pred, average=None)
    class_type_np = np.sort(np.unique(y_true), 0);
    weight_arr = [0.2, 0.2, 0.6];  # 依次对应类别0(不存在), 类别1，类别2，类别3的权重
    f1_score_val = 0;
    # 为了适应lightGBM的label从0开始做的调整，即类别1==> class=0; 类别2==>class=1; 类别3==>class=2;
    for class_type_idx, class_type in enumerate(class_type_np):
        f1_score_val = f1_score_val + weight_arr[int(class_type)] * f1_score_arr[class_type_idx];

    return 'f1_score_sk', f1_score_val, True


# def custom_asymmetric_valid(y_true, y_pred):
#     residual = (y_true - y_pred).astype("float")
#     loss = np.where(residual < 0, (residual**2)*10.0, residual**2)
#     return "custom_asymmetric_eval", np.mean(loss), False

def train_by_light_gbm_00():
    logging.info('train_by_light_gbm, 训练开始......');
    dir = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/'

    current_history_attr_road_df_path = dir + '20201015181036traffic-fix/hd1/20190701.txt_hd1';
    current_history_attr_road_df = pd.read_csv(current_history_attr_road_df_path)
    x_df = current_history_attr_road_df.drop(columns=['label']);
    y_df = current_history_attr_road_df[['label']];

    if MODE == 'DEV':
        x_df = x_df.iloc[0:1000, :];
        y_df = y_df.iloc[0:1000, :];

    seed = 7
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=test_size, random_state=seed)
    # eval_set = [(X_test.values, y_test.values)]
    n_class = 5;

    params = {
        'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'None',
        'num_leaves': 31,
        'num_class': n_class,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'nthread': -1,
        'verbose': -1
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_test, label=y_test)

    clf = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=5000,
        valid_sets=[dvalid],
        early_stopping_rounds=10,
        verbose_eval=100,
        feval=f1_score_eval
    )

    logging.info('训练完毕......');


def train_by_light_gbm_01():
    logging.info('train_by_light_gbm, 训练开始......');
    dir = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/'

    current_history_attr_road_df_path = dir + '20201015181036traffic-fix/hd1/20190701.txt_hd1';
    current_history_attr_road_df = pd.read_csv(current_history_attr_road_df_path)
    X_train = current_history_attr_road_df.drop(columns=['label']);
    y_train = current_history_attr_road_df[['label']];

    current_history_attr_road_df_path_test = dir + '20201015181036traffic-fix/hd1/20190730.txt_hd1';
    current_history_attr_road_test_df = pd.read_csv(current_history_attr_road_df_path_test)
    X_test = current_history_attr_road_test_df.drop(columns=['label']);
    y_test = current_history_attr_road_test_df[['label']];

    # if MODE == 'DEV':
    #     x_df = x_df.iloc[0:1000, :];
    #     y_df = y_df.iloc[0:1000, :];

    # seed = 7
    # test_size = 0.2
    # X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=test_size, random_state=seed)
    # eval_set = [(X_test.values, y_test.values)]
    n_class = 5;

    params = {
        'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'None',
        'num_leaves': 31,
        'num_class': n_class,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'nthread': -1,
        'verbose': -1
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_test, label=y_test)

    clf = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=5000,
        valid_sets=[dvalid],
        early_stopping_rounds=10,
        verbose_eval=10,
        feval=f1_score_eval
    )

    logging.info('训练完毕......');


def train_by_light_gbm_by_one_day(train_file_name, init_gbm_model):
    # logging.info('训练开始, train_file_name=', train_file_name);
    # dir = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/'

    # 训练集的加载
    # current_history_attr_road_df_train_path = dir + '20201015181036traffic-fix/hd1/20190701.txt_hd1';
    current_history_attr_road_df_train_path = dir + '20201015181036traffic-fix/hd1/' + train_file_name;
    current_history_attr_road_train_df = pd.read_csv(current_history_attr_road_df_train_path)
    X_train = current_history_attr_road_train_df.drop(columns=['label']);
    y_train = current_history_attr_road_train_df[['label']];

    # 将对验证集的加载
    current_history_attr_road_df_test_path = dir + '20201015181036traffic-fix/hd1/20190730.txt_hd1';
    current_history_attr_road_test_df = pd.read_csv(current_history_attr_road_df_test_path)
    X_test = current_history_attr_road_test_df.drop(columns=['label']);
    y_test = current_history_attr_road_test_df[['label']];

    if MODE == 'DEV':
        X_train = X_train.iloc[0:1000, :];
        y_train = y_train.iloc[0:1000, :];
        X_test = X_test.iloc[0:1000, :];
        y_test = y_test.iloc[0:1000, :];

    # seed = 7
    # test_size = 0.2
    # X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=test_size, random_state=seed)
    # eval_set = [(X_test.values, y_test.values)]
    n_class = 5;

    params = {
        'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'None',
        'num_leaves': 31,
        'num_class': n_class,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'nthread': -1,
        'verbose': -1
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_test, label=y_test)

    gbm_model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=5000,
        valid_sets=[dvalid],
        early_stopping_rounds=10,
        verbose_eval=10,
        feval=f1_score_eval,
        init_model=init_gbm_model,  # 如果init_model不为None，那么就是在init_model基础上接着训练
        keep_training_booster=True  # 增量训练
    )

    # logging.info('训练完毕');
    # val_preds_loaded = gbm_model.predict(val_X, num_iteration=gbm_model.best_iteration)  # 输出的是概率结果
    # print(gbm_model.best_score)
    # gbm_model.best_score[]

    # gbm_model.best_score['valid_0']['f1_score']
    if init_gbm_model is not None:
        old_model_score = init_gbm_model.best_score['valid_0']['f1_score'];
        new_model_score = gbm_model.best_score['valid_0']['f1_score'];
        logging.debug('old_model_score=' + str(old_model_score) + ', new_model_score=' + str(new_model_score));
        if new_model_score <= old_model_score:
            return init_gbm_model;

    return gbm_model


def my_fun(val):
    """
    按照这种写法，返回的结果是一列，apply函数返回的是一个dataframe
    :param val:
    :return:
    """
    try:
        val = int(float(val.strip('\n').strip('\t').strip('\r')));
    except ValueError as err:
        print('err =\n', err);
        print('val =', val);
    return val;


def change_df_type(train_df):
    for column_name in train_df.columns:
        if column_name.find('speed') != -1:
            train_df[[column_name]] = train_df[[column_name]].astype(float);
        else:
            # line = line.strip("\n")
            # line = line.split(" ")
            # train_df.loc[:, column_name] = train_df.loc[:, column_name].strip('\n');
            # train_df[[column_name]] = train_df[[column_name]].astype(float);
            # train_df[[column_name]] = train_df[[column_name]].astype(int);
            if train_df[column_name].dtypes == 'object':

                # train_df.loc[:, column_name] = train_df.loc[:, column_name].map(lambda val: int(float(val.strip('\n')
                #     .strip('\t').strip('\r'))));
                # except ValueError as err:
                #     print()
                train_df.loc[:, column_name] = train_df.loc[:, column_name].apply(my_fun)
            else:
                train_df[[column_name]] = train_df[[column_name]].astype(int);


def add_history_speed_features(x_df):
    # ['road_speed_-28_0', 'road_speed_-21_0', 'road_speed_-14_0', 'road_speed_-7_0']
    x_df['his_road_speed_0_mean'] = \
        x_df.loc[:, ['road_speed_-28_0', 'road_speed_-21_0', 'road_speed_-14_0', 'road_speed_-7_0']].mean(1);
    # x_df['his_road_speed_0_std'] = \
    #     x_df.loc[:, ['road_speed_-28_0', 'road_speed_-21_0', 'road_speed_-14_0', 'road_speed_-7_0']].std(1);
    #
    # x_df['his_road_speed_1_mean'] = \
    #     x_df.loc[:, ['road_speed_-28_1', 'road_speed_-21_1', 'road_speed_-14_1', 'road_speed_-7_1']].mean(1);
    # x_df['his_road_speed_1_std'] = \
    #     x_df.loc[:, ['road_speed_-28_1', 'road_speed_-21_1', 'road_speed_-14_1', 'road_speed_-7_1']].std(1);
    #
    # x_df['his_road_speed_2_mean'] = \
    #     x_df.loc[:, ['road_speed_-28_2', 'road_speed_-21_2', 'road_speed_-14_2', 'road_speed_-7_2']].mean(1);
    # x_df['his_road_speed_2_std'] = \
    #     x_df.loc[:, ['road_speed_-28_2', 'road_speed_-21_2', 'road_speed_-14_2', 'road_speed_-7_2']].std(1);
    #
    # x_df['his_road_speed_3_mean'] = \
    #     x_df.loc[:, ['road_speed_-28_3', 'road_speed_-21_3', 'road_speed_-14_3', 'road_speed_-7_3']].mean(1);
    # x_df['his_road_speed_3_std'] = \
    #     x_df.loc[:, ['road_speed_-28_3', 'road_speed_-21_3', 'road_speed_-14_3', 'road_speed_-7_3']].std(1);
    #
    # x_df['his_road_speed_4_mean'] = \
    #     x_df.loc[:, ['road_speed_-28_4', 'road_speed_-21_4', 'road_speed_-14_4', 'road_speed_-7_4']].mean(1);
    # x_df['his_road_speed_4_std'] = \
    #     x_df.loc[:, ['road_speed_-28_4', 'road_speed_-21_4', 'road_speed_-14_4', 'road_speed_-7_4']].std(1);

    # x_df = x_df.drop(columns=['road_speed_-28_0', 'road_speed_-21_0', 'road_speed_-14_0', 'road_speed_-7_0',
    #                           'road_speed_-28_1', 'road_speed_-21_1', 'road_speed_-14_1', 'road_speed_-7_1',
    #                           'road_speed_-28_2', 'road_speed_-21_2', 'road_speed_-14_2', 'road_speed_-7_2',
    #                           'road_speed_-28_3', 'road_speed_-21_3', 'road_speed_-14_3', 'road_speed_-7_3',
    #                           'road_speed_-28_4', 'road_speed_-21_4', 'road_speed_-14_4', 'road_speed_-7_4',
    #                           'eta_speed_-28_0', 'eta_speed_-21_0', 'eta_speed_-14_0', 'eta_speed_-7_0',
    #                           'eta_speed_-28_1', 'eta_speed_-21_1', 'eta_speed_-14_1', 'eta_speed_-7_1',
    #                           'eta_speed_-28_2', 'eta_speed_-21_2', 'eta_speed_-14_2', 'eta_speed_-7_2',
    #                           'eta_speed_-28_3', 'eta_speed_-21_3', 'eta_speed_-14_3', 'eta_speed_-7_3',
    #                           'eta_speed_-28_4', 'eta_speed_-21_4', 'eta_speed_-14_4', 'eta_speed_-7_4',
    #     ]);

    # x_df['his_road_speed_min'] = \
    #     x_df.loc[:, ['road_speed_-28_0', 'road_speed_-21_0', 'road_speed_-14_0', 'road_speed_-7_0']].min(1);
    # x_df['his_road_speed_max'] = \
    #     x_df.loc[:, ['road_speed_-28_0', 'road_speed_-21_0', 'road_speed_-14_0', 'road_speed_-7_0']].max(1);

    #
    # x_df = x_df.drop(columns=['road_speed_-28_0', 'road_speed_-21_0', 'road_speed_-14_0', 'road_speed_-7_0']);

    # ['road_speed_-28_1', 'road_speed_-21_1', 'road_speed_-14_1', 'road_speed_-7_1']
    # ['road_speed_-28_2', 'road_speed_-21_2', 'road_speed_-14_2', 'road_speed_-7_2']
    # ['road_speed_-28_3', 'road_speed_-21_3', 'road_speed_-14_3', 'road_speed_-7_3']
    # ['road_speed_-28_4', 'road_speed_-21_4', 'road_speed_-14_4', 'road_speed_-7_4']
    #
    # ['eta_speed_-28_0', 'eta_speed_-21_0', 'eta_speed_-14_0', 'eta_speed_-7_0']
    # ['eta_speed_-28_1', 'eta_speed_-21_1', 'eta_speed_-14_1', 'eta_speed_-7_1']
    # ['eta_speed_-28_2', 'eta_speed_-21_2', 'eta_speed_-14_2', 'eta_speed_-7_2']
    # ['eta_speed_-28_3', 'eta_speed_-21_3', 'eta_speed_-14_3', 'eta_speed_-7_3']
    # ['eta_speed_-28_4', 'eta_speed_-21_4', 'eta_speed_-14_4', 'eta_speed_-7_4']

    return x_df;


def add_eta_over_road_speed_feature(x_df):
    """

    :param x_df:
    :return:
    """
    '''
    slice_id_0_0,road_speed_0_0,eta_speed_0_0,label_0_0,calc_car_num_0_0,
    slice_id_0_1,road_speed_0_1,eta_speed_0_1,label_0_1,calc_car_num_0_1,
    slice_id_0_2,road_speed_0_2,eta_speed_0_2,label_0_2,calc_car_num_0_2,
    slice_id_0_3,road_speed_0_3,eta_speed_0_3,label_0_3,calc_car_num_0_3,
    slice_id_0_4,road_speed_0_4,eta_speed_0_4,label_0_4,calc_car_num_0_4,
    '''

    date_flag_list = ['0', '-28', '-21', '-14', '-7']
    slice_flag_list = ['0', '1', '2', '3', '4']

    for date_flag in date_flag_list:
        for slice_flag in slice_flag_list:
            # 变量后缀
            var_suffix = '_' + date_flag + '_' + slice_flag
            x_df['eta_over_road_speed' + var_suffix] = x_df['eta_speed' + var_suffix] / \
                                                       (x_df['road_speed' + var_suffix] + 1e-10);
    return x_df;


def add_label_cnt(x_df):
    pass


def add_road_over_eta_speed_feature(x_df):
    """

    :param x_df:
    :return:
    """

    date_flag_list = ['0', '-28', '-21', '-14', '-7']
    slice_flag_list = ['0', '1', '2', '3', '4']

    for date_flag in date_flag_list:
        for slice_flag in slice_flag_list:
            # 变量后缀
            var_suffix = '_' + date_flag + '_' + slice_flag
            x_df['road_over_eta_speed' + var_suffix] = x_df['road_speed' + var_suffix] / \
                                                       (x_df['eta_speed' + var_suffix] + 1e-10);
    return x_df;


def my_feature_engineer(current_history_attr_road_train_df):
    current_history_attr_road_train_df = current_history_attr_road_train_df.fillna(0);
    if MODE == 'DEV':
        current_history_attr_road_train_df = current_history_attr_road_train_df.iloc[0:1000, :];
    elif MODE == 'PARAMS':
        current_history_attr_road_train_df = current_history_attr_road_train_df.iloc[0:10000, :];
        # current_history_attr_road_train_df = current_history_attr_road_train_df.iloc[0:150000, :];

    # 输入数据中， 删除'label'、 slice_id相关的列
    del_columns = ['label', 'level',
                   'slice_id_-28_0', 'slice_id_-28_1', 'slice_id_-28_2', 'slice_id_-28_3', 'slice_id_-28_4',
                   'slice_id_-21_0', 'slice_id_-21_1', 'slice_id_-21_2', 'slice_id_-21_3', 'slice_id_-21_4',
                   'slice_id_-14_0', 'slice_id_-14_1', 'slice_id_-14_2', 'slice_id_-14_3', 'slice_id_-14_4',
                   'slice_id_-7_0', 'slice_id_-7_1', 'slice_id_-7_2', 'slice_id_-7_3', 'slice_id_-7_4',
                   'slice_id_0_0', 'slice_id_0_1', 'slice_id_0_2', 'slice_id_0_3', 'slice_id_0_4'];

    # 去掉3个当前的label占比特征后，考查模型情况
    # del_columns.extend(['current_label_1_prop', 'current_label_2_prop', 'current_label_3_prop'])

    # 去掉历史label原始的label值，仅保留占比统计结果
    # del_columns.extend([
    #     'label_-28_0', 'label_-28_1', 'label_-28_2', 'label_-28_3', 'label_-28_4',
    #     'label_-21_0', 'label_-21_1', 'label_-21_2', 'label_-21_3', 'label_-21_4',
    #     'label_-14_0', 'label_-14_1', 'label_-14_2', 'label_-14_3', 'label_-14_4',
    #     'label_-7_0', 'label_-7_1', 'label_-7_2', 'label_-7_3', 'label_-7_4']);

    # 去掉link_id
    # del_columns.append('link_id');

    # current_history_attr_road_train_df = add_eta_over_road_speed_feature(current_history_attr_road_train_df);
    current_history_attr_road_train_df = add_road_over_eta_speed_feature(current_history_attr_road_train_df);

    current_history_attr_road_train_df['time_diff'] = \
        current_history_attr_road_train_df['future_slice_id'] - current_history_attr_road_train_df['current_slice_id'];

    x_df = current_history_attr_road_train_df.drop(columns=del_columns);

    # x_df['recent_road_speed_mean'] = x_df.loc[:, ['road_speed_0_0', 'road_speed_0_1', 'road_speed_0_2',
    #                                               'road_speed_0_3', 'road_speed_0_4']].mean(1);

    # x_df['flow_area_per_s_1'] = x_df['speed_limit'] * x_df['width'];
    # x_df['flow_area_per_s_2'] = x_df['recent_road_speed_mean'] * x_df['width'];

    # x_df = add_history_speed_features(x_df);

    # 通行总时间
    # x_df['road_access_time'] = x_df['length'] / x_df['speed_limit'];

    # 当前通行总时间
    # x_df['recent_access_time'] = x_df['length'] / x_df['recent_road_speed_mean'];

    # x_df['speed_limit'] = x_df['speed_limit'] * 3.6;
    # x_df['road_speed_0_0_over_speed_limit'] = x_df['road_speed_0_0'] / x_df['speed_limit'];
    # x_df['eta_speed_0_0_over_speed_limit'] = x_df['eta_speed_0_0'] / x_df['speed_limit'];

    # 是否是早高峰、晚高峰的特征构造
    # x_df['is_future_morning_peak'] = x_df['future_slice_id'].map(lambda id: 1 if (id >= 225) & (id <= 270) else 0);
    # x_df['is_future_evening_peak'] = x_df['future_slice_id'].map(lambda id: 1 if (id >= 540) & (id <= 600) else 0);
    x_df['is_future_peak'] = x_df['future_slice_id'].map(lambda _id: 1 if ((_id >= 225) & (_id <= 270)) |
                                                                          ((_id >= 540) & (_id <= 600)) else 0);

    # 删除future_slice_id
    # x_df = x_df.drop(columns=['future_slice_id']);

    # 为了适应lightGBM的label从0开始做的调整，在最终预测结果中，还需要加回去;
    y_df = current_history_attr_road_train_df[['label']].copy();
    y_df.loc[y_df.label == 0, 'label'] = 1;
    y_df.loc[y_df.label == 4, 'label'] = 3;
    y_df['label'] = y_df['label'] - 1;

    print('train dataset shape = ', x_df.shape)
    return x_df, y_df;


def loglikelihood(y_true, y_pred):
    # labels = train_data.get_label()
    # preds = 1. / (1. + np.exp(-preds))
    # grad = preds - labels
    # hess = preds * (1. - preds)

    # y_pred = np.argmax(y_pred.reshape(3, -1), axis=0)
    # y_pred = 1. / (1. + np.exp(-y_pred))
    # grad = y_pred - y_true
    # hess = y_pred * (1. - y_pred) + 1e-6
    # return grad, hess

    labels = y_true
    preds = np.reshape(y_pred, (len(labels), 3))
    preds = 1. / (1. + np.exp(-preds))
    preds = softmax(preds, axis=1)
    labels = OneHotEncoder(sparse=False).fit_transform(labels.reshape(-1, 1))
    grad = (preds - labels)
    hess = preds * (1.0 - preds)
    return grad.flatten("F"), hess.flatten("F")


def logregobj(labels, preds):
    preds = np.reshape(preds, (len(labels), 3), "F")
    preds = softmax(preds, axis=1)
    labels = OneHotEncoder(sparse=False).fit_transform(labels.reshape(-1, 1))
    grad = (preds - labels)
    hess = preds * (1.0 - preds)
    return grad.flatten('F'), hess.flatten('F')


def train_by_light_gbm_by_one_day_02(train_file_name):
    _dir_path = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/'

    # 训练集的加载
    current_history_attr_road_df_train_path = _dir_path + '20201015181036traffic-fix/hd1/' + train_file_name;
    current_history_attr_road_train_df = pd.read_csv(current_history_attr_road_df_train_path)

    # 训练集的加载
    # current_history_attr_road_df_train_path_01 = _dir_path + '20201015181036traffic-fix/hd1/20190701.txt_hd1';
    # current_history_attr_road_df_train_path_02 = _dir_path + '20201015181036traffic-fix/hd1/20190702.txt_hd1';
    # current_history_attr_road_df_train_path_03 = _dir_path + '20201015181036traffic-fix/hd1/20190703.txt_hd1';
    # # current_history_attr_road_df_train_path_04 = _dir_path + '20201015181036traffic-fix/hd1/20190704.txt_hd1';
    # current_history_attr_road_train_df = pd.read_csv(current_history_attr_road_df_train_path_01)
    #
    # current_history_attr_road_train_df_02 = pd.read_csv(current_history_attr_road_df_train_path_02)
    # current_history_attr_road_train_df = \
    #     current_history_attr_road_train_df.append(current_history_attr_road_train_df_02, ignore_index=True);
    # del current_history_attr_road_train_df_02;

    # current_history_attr_road_train_df_03 = pd.read_csv(current_history_attr_road_df_train_path_03)
    # current_history_attr_road_train_df = \
    #     current_history_attr_road_train_df.append(current_history_attr_road_train_df_03, ignore_index=True);
    # del current_history_attr_road_train_df_03;

    # current_history_attr_road_train_df_04 = pd.read_csv(current_history_attr_road_df_train_path_04)
    # current_history_attr_road_train_df = \
    #     current_history_attr_road_train_df.append(current_history_attr_road_train_df_04, ignore_index=True);
    # del current_history_attr_road_train_df_04;

    # 将对验证集的加载
    current_history_attr_road_df_test_path = _dir_path + '20201015181036traffic-fix/hd1/20190730.txt_hd1';
    current_history_attr_road_test_df = pd.read_csv(current_history_attr_road_df_test_path)

    X_train, y_train = my_feature_engineer(current_history_attr_road_train_df);
    X_test, y_test = my_feature_engineer(current_history_attr_road_test_df);

    n_class = 3
    # params = {
    #     'learning_rate': 0.05,
    #     'boosting_type': 'gbdt',
    #     'objective': 'multiclass',
    #     'metric': 'None',
    #     'num_leaves': 31,
    #     'num_class': n_class,
    #     'feature_fraction': 0.8,
    #     'bagging_fraction': 0.8,
    #     'bagging_freq': 5,
    #     'seed': 1,
    #     'bagging_seed': 1,
    #     'feature_fraction_seed': 7,
    #     'min_data_in_leaf': 20,
    #     'nthread': -1,
    #     'verbose': -1
    #     # 'is_unbalance': True
    #     # 'scale_pos_weight': 1
    # }

    # dtrain = lgb.Dataset(X_train, label=y_train)
    # dvalid = lgb.Dataset(X_test, label=y_test)
    #
    # gbm_model = lgb.train(
    #     params=params,
    #     train_set=dtrain,
    #     num_boost_round=5000,
    #     valid_sets=[dvalid],
    #     early_stopping_rounds=10,
    #     verbose_eval=10,
    #     feval=f1_score_eval,
    #     init_model=init_gbm_model,  # 如果init_model不为None，那么就是在init_model基础上接着训练
    #     keep_training_booster=True  # 增量训练
    #     # categorical_feature=['direction']
    # )

    model = lgb.LGBMClassifier(
        learning_rate=0.05,
        boosting_type='gbdt',
        objective='multiclass',

        num_leaves=31,
        max_depth=6,
        num_class=n_class,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        seed=1,  # 随机数种子
        bagging_seed=1,
        feature_fraction_seed=7,
        min_data_in_leaf=20,
        nthread=-1,
        verbose=-1,
        num_boost_round=5000
        # class_weight={0: 0.8247599007864205, 1: 0.1381701912565094, 2: 0.0370699079570701}
        # n_estimators=200,  # 使用多少个弱分类器
        # num_class=3,
        # booster='gbtree',
        # min_child_weight=2,
        # subsample=0.8,
        # colsample_bytree=0.8,
        # reg_alpha=0,
        # reg_lambda=1
    )

    # model.set_params(**{'objective': loglikelihood})
    # model.set_params(**{'objective': logregobj})

    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=10, early_stopping_rounds=10, eval_metric=f1_score_eval_sk)

    # 在提交测试集上预测
    # 对测试集进行预测
    y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
    y_pred_prob = model.predict_proba(X_test, num_iteration=model.best_iteration_)

    # 计算准确率
    # accuracy = accuracy_score(y_test, y_pred)
    # print('accuracy:%3.f%%' % (accuracy * 100))

    final_test_f1_score = f1_score_eval_final_test_sk(y_test, y_pred);
    print('final_test_f1_score =', final_test_f1_score);

    logging.info('训练完毕');

    # # gbm_model.best_score['valid_0']['f1_score']
    # if init_gbm_model is not None:
    #     old_model_score = init_gbm_model.best_score['valid_0']['f1_score'];
    #     new_model_score = gbm_model.best_score['valid_0']['f1_score'];
    #     logging.debug('old_model_score=' + str(old_model_score) + ', new_model_score=' + str(new_model_score));
    #     if new_model_score <= old_model_score:
    #         return init_gbm_model;
    return model


def model_metric_loss_func(w, y_true, y_pred_prob):
    m = y_true.shape[0];
    w_np = np.repeat(np.expand_dims(np.array(w), axis=1), repeats=m, axis=1).T

    rescale_y_pred = np.argmax(y_pred_prob * w_np, axis=1)

    f1_score_arr = f1_score(y_true=y_true, y_pred=rescale_y_pred, average=None)
    class_type_np = np.sort(np.unique(y_true), 0);
    weight_arr = [0.2, 0.2, 0.6];  # 依次对应类别0(不存在), 类别1，类别2，类别3的权重
    f1_score_val = 0;
    for class_type_idx, class_type in enumerate(class_type_np):
        f1_score_val = f1_score_val + weight_arr[int(class_type)] * f1_score_arr[class_type_idx];

    global MODE;
    if MODE == 'DEV':
        print("w = " + str(w) + ", -1*f1_score_val =" + str(-1 * f1_score_val))

    return -1 * f1_score_val;


def train_by_light_gbm_by_one_day_with_second_model_offline_base(train_file_name):
    """
    :param train_file_name:
    :return:
    """
    print("train_by_light_gbm_by_one_day_with_second_model_offline_base is startting ... ");
    _dir_path = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/20201015181036traffic-fix/hd4/'

    # 训练集的加载
    current_history_attr_road_df_train_path = _dir_path + train_file_name;
    current_history_attr_road_train_df = pd.read_csv(current_history_attr_road_df_train_path)

    # 训练集的加载
    # current_history_attr_road_df_train_path_01 = _dir_path + '20201015181036traffic-fix/hd1/20190701.txt_hd1';
    # current_history_attr_road_df_train_path_03 = _dir_path + '20201015181036traffic-fix/hd1/20190703.txt_hd1';
    # # current_history_attr_road_df_train_path_04 = _dir_path + '20201015181036traffic-fix/hd1/20190704.txt_hd1';
    # current_history_attr_road_train_df = pd.read_csv(current_history_attr_road_df_train_path_01)
    if MODE != 'DEV':
        current_history_attr_road_df_train_path_02 = _dir_path + '20190702.txt_hd1_hd4';
        current_history_attr_road_train_df_02 = pd.read_csv(current_history_attr_road_df_train_path_02)
        current_history_attr_road_train_df = \
            current_history_attr_road_train_df.append(current_history_attr_road_train_df_02, ignore_index=True);
        del current_history_attr_road_train_df_02;

    # current_history_attr_road_train_df_03 = pd.read_csv(current_history_attr_road_df_train_path_03)
    # current_history_attr_road_train_df = \
    #     current_history_attr_road_train_df.append(current_history_attr_road_train_df_03, ignore_index=True);
    # del current_history_attr_road_train_df_03;

    # current_history_attr_road_train_df_04 = pd.read_csv(current_history_attr_road_df_train_path_04)
    # current_history_attr_road_train_df = \
    #     current_history_attr_road_train_df.append(current_history_attr_road_train_df_04, ignore_index=True);
    # del current_history_attr_road_train_df_04;

    # 将对验证集的加载
    current_history_attr_road_df_test_path = _dir_path + '20190730.txt_hd1_hd4';
    current_history_attr_road_test_df = pd.read_csv(current_history_attr_road_df_test_path)

    X_train, y_train = my_feature_engineer(current_history_attr_road_train_df);
    X_test, y_test = my_feature_engineer(current_history_attr_road_test_df);

    n_class = 3
    model = lgb.LGBMClassifier(
        learning_rate=0.05,
        boosting_type='gbdt',
        objective='multiclass',
        num_leaves=31,
        max_depth=6,
        num_class=n_class,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        seed=1,  # 随机数种子
        bagging_seed=1,
        feature_fraction_seed=7,
        min_data_in_leaf=20,
        nthread=-1,
        verbose=-1,
        num_boost_round=5000,
        class_weight={0: 6.792153871, 1: 0.768032069, 2: 1}
        # class_weight={0: 1, 1: 1, 2: 3}
        # n_estimators=200,  # 使用多少个弱分类器
        # num_class=3,
        # booster='gbtree',
        # min_child_weight=2,
        # subsample=0.8,
        # colsample_bytree=0.8,
        # reg_alpha=0,
        # reg_lambda=1
    )

    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=10, early_stopping_rounds=10, eval_metric=f1_score_eval_sk)

    # 在提交测试集上预测
    # 对测试集进行预测
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration_)
    y_train_pred_prob = model.predict_proba(X_train, num_iteration=model.best_iteration_)

    final_test_f1_score = f1_score_eval_final_test_sk(y_test, y_test_pred);
    print('before optimize metric function, final_test_f1_score =', final_test_f1_score);
    del y_test_pred;

    # 针对评估函数，进行优化;
    bnds = ((0, 1000), (0, 1000), (0, 1000))
    w = [1.0, 1.0, 1.0];

    partial_model_metric_loss_func = partial(model_metric_loss_func, y_true=y_train, y_pred_prob=y_train_pred_prob);
    res = minimize(partial_model_metric_loss_func, w, method='Powell', bounds=bnds)
    print('res.fun = ' + str(res.fun) + ', res.success = ' + str(res.success) + " , res.x = " + str(res.x))
    del y_train;
    del y_train_pred_prob;

    # 使用对评估函数优化后的参数，重新缩放prediction，在测试集上验证
    rescale_weight = res.x;
    y_test_pred_prob = model.predict_proba(X_test, num_iteration=model.best_iteration_)
    m = y_test_pred_prob.shape[0];
    rescale_weight_np = np.repeat(np.expand_dims(np.array(rescale_weight), axis=1), repeats=m, axis=1).T
    rescale_y_test_pred = np.argmax(y_test_pred_prob * rescale_weight_np, axis=1)

    rescale_final_test_f1_score = f1_score_eval_final_test_sk(y_test, rescale_y_test_pred);
    print('after optimize metric function, final_test_f1_score =', rescale_final_test_f1_score);

    logging.info('train is end ... ');
    return model, rescale_weight


def train_by_light_gbm_by_one_day_with_second_model_with_same_week_online_base():
    """
    :param train_file_name:
    :return:
    """
    print("train_by_light_gbm_by_one_day_with_second_model_with_same_week_online_base is startting ... ");
    _dir_path = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/20201015181036traffic-fix/hd4/'

    # 训练集的加载
    # 20190704.txt_hd1  20190711.txt_hd1
    current_history_attr_road_df_train_path = _dir_path + '20190704.txt_hd1_hd4'
    current_history_attr_road_train_df = pd.read_csv(current_history_attr_road_df_train_path)

    # current_history_attr_road_train_df = pd.read_csv(current_history_attr_road_df_train_path_01)
    if MODE != 'DEV':
        current_history_attr_road_df_train_path_02 = _dir_path + '20190711.txt_hd1_hd4';
        current_history_attr_road_train_df_02 = pd.read_csv(current_history_attr_road_df_train_path_02)
        current_history_attr_road_train_df = \
            current_history_attr_road_train_df.append(current_history_attr_road_train_df_02, ignore_index=True);
        del current_history_attr_road_train_df_02;

        current_history_attr_road_df_train_path_03 = _dir_path + '20190718.txt_hd1_hd4';
        current_history_attr_road_train_df_03 = pd.read_csv(current_history_attr_road_df_train_path_03)
        current_history_attr_road_train_df = \
            current_history_attr_road_train_df.append(current_history_attr_road_train_df_03, ignore_index=True);
        del current_history_attr_road_train_df_03;

        current_history_attr_road_df_train_path_04 = _dir_path + '20190725.txt_hd1_hd4';
        current_history_attr_road_train_df_04 = pd.read_csv(current_history_attr_road_df_train_path_04)
        current_history_attr_road_train_df = \
            current_history_attr_road_train_df.append(current_history_attr_road_train_df_04, ignore_index=True);
        del current_history_attr_road_train_df_04;

    # 将对验证集的加载
    # current_history_attr_road_df_test_path = _dir_path + '20201015181036traffic-fix/hd1/20190725.txt_hd1';
    current_history_attr_road_df_test_path = _dir_path + '20190730.txt_hd1_hd4';
    current_history_attr_road_test_df = pd.read_csv(current_history_attr_road_df_test_path)

    X_train, y_train = my_feature_engineer(current_history_attr_road_train_df);
    X_test, y_test = my_feature_engineer(current_history_attr_road_test_df);

    n_class = 3
    model = lgb.LGBMClassifier(
        learning_rate=0.05,
        boosting_type='gbdt',
        objective='multiclass',
        num_leaves=31,
        max_depth=6,
        num_class=n_class,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        seed=1,  # 随机数种子
        bagging_seed=1,
        feature_fraction_seed=7,
        min_data_in_leaf=20,
        nthread=-1,
        verbose=-1,
        num_boost_round=5000
        # class_weight={0: 1, 1: 1, 2: 3}
        # n_estimators=200,  # 使用多少个弱分类器
        # num_class=3,
        # booster='gbtree',
        # min_child_weight=2,
        # subsample=0.8,
        # colsample_bytree=0.8,
        # reg_alpha=0,
        # reg_lambda=1
    )

    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=10, early_stopping_rounds=10, eval_metric=f1_score_eval_sk)

    # 在提交测试集上预测
    # 对测试集进行预测
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration_)
    y_train_pred_prob = model.predict_proba(X_train, num_iteration=model.best_iteration_)

    final_test_f1_score = f1_score_eval_final_test_sk(y_test, y_test_pred);
    print('before optimize metric function, final_test_f1_score =', final_test_f1_score);
    del y_test_pred;

    # 针对评估函数，进行优化;
    bnds = ((0, 1000), (0, 1000), (0, 1000))
    w = [1.0, 1.0, 1.0];

    partial_model_metric_loss_func = partial(model_metric_loss_func, y_true=y_train, y_pred_prob=y_train_pred_prob);
    res = minimize(partial_model_metric_loss_func, w, method='Powell', bounds=bnds)
    print('res.fun = ' + str(res.fun) + ', res.success = ' + str(res.success) + " , res.x = " + str(res.x))
    del y_train;
    del y_train_pred_prob;

    # 使用对评估函数优化后的参数，重新缩放prediction，在测试集上验证
    rescale_weight = res.x;
    y_test_pred_prob = model.predict_proba(X_test, num_iteration=model.best_iteration_)
    m = y_test_pred_prob.shape[0];
    rescale_weight_np = np.repeat(np.expand_dims(np.array(rescale_weight), axis=1), repeats=m, axis=1).T
    rescale_y_test_pred = np.argmax(y_test_pred_prob * rescale_weight_np, axis=1)

    rescale_final_test_f1_score = f1_score_eval_final_test_sk(y_test, rescale_y_test_pred);
    print('after optimize metirc function, final_test_f1_score =', rescale_final_test_f1_score);

    logging.info('trian is end ... ');
    return model, rescale_weight


def my_custom_gscv_score(y_true, y_pred):
    f1_score_arr = f1_score(y_true=y_true, y_pred=y_pred, average=None)
    class_type_np = np.sort(np.unique(y_true), 0);
    weight_arr = [0.2, 0.2, 0.6];  # 依次对应类别0(不存在), 类别1，类别2，类别3的权重
    f1_score_val = 0;
    # 为了适应lightGBM的label从0开始做的调整，即类别1==> class=0; 类别2==>class=1; 类别3==>class=2;
    for class_type_idx, class_type in enumerate(class_type_np):
        f1_score_val = f1_score_val + weight_arr[int(class_type)] * f1_score_arr[class_type_idx];

    print('my_custom_gscv_score, y_pred.shape=' + str(y_pred.shape) + 'f1_score_val =' + str(f1_score_val));
    return f1_score_val


def train_by_light_gbm_by_one_day_02_adjust_params():
    _dir_path = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/'

    # 训练集的加载
    current_history_attr_road_df_train_path_01 = _dir_path + '20201015181036traffic-fix/hd1/20190701.txt_hd1';
    current_history_attr_road_df_train_path_02 = _dir_path + '20201015181036traffic-fix/hd1/20190702.txt_hd1';
    current_history_attr_road_train_df = pd.read_csv(current_history_attr_road_df_train_path_01)
    current_history_attr_road_train_df_02 = pd.read_csv(current_history_attr_road_df_train_path_02)
    current_history_attr_road_train_df = \
        current_history_attr_road_train_df.append(current_history_attr_road_train_df_02, ignore_index=True);
    del current_history_attr_road_train_df_02;

    # 将对验证集的加载
    current_history_attr_road_df_test_path = _dir_path + '20201015181036traffic-fix/hd1/20190730.txt_hd1';
    current_history_attr_road_test_df = pd.read_csv(current_history_attr_road_df_test_path)

    X_train, y_train = my_feature_engineer(current_history_attr_road_train_df);
    X_test, y_test = my_feature_engineer(current_history_attr_road_test_df);

    n_class = 3
    # parameters = {
    #     'max_depth': [4, 6, 8],
    #     'num_leaves': [20, 30, 40],
    # }

    # # 当使用make_scorer把一个函数转换成一个scorer对象时，设置greater_is_better参数为False
    # my_custom_gscv_score
    # ftwo_scorer
    my_custom_gscv_scorer = make_scorer(my_custom_gscv_score, greater_is_better=True)

    # range(3, 8, 2)

    global MODE;
    if MODE == 'DEV':
        n_jobs = 1;
        max_depth = [4, 6]
        num_leaves = [31]
    else:
        n_jobs = 1;
        max_depth = [6];
        num_leaves = [31];

    parameters = {
        'max_depth': max_depth,
        'num_leaves': num_leaves
    }

    model = lgb.LGBMClassifier(
        learning_rate=0.05,
        boosting_type='gbdt',
        objective='multiclass',

        num_leaves=31,
        max_depth=-1,
        num_class=n_class,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        seed=1,  # 随机数种子
        bagging_seed=1,
        feature_fraction_seed=7,
        min_data_in_leaf=20,
        nthread=-1,
        verbose=-1,
        num_boost_round=5000
        # class_weight={0: 0.8247599007864205, 1: 0.1381701912565094, 2: 0.0370699079570701}
        # n_estimators=200,  # 使用多少个弱分类器
        # num_class=3,
        # booster='gbtree',
        # min_child_weight=2,
        # subsample=0.8,
        # colsample_bytree=0.8,
        # reg_alpha=0,
        # reg_lambda=1
    )

    # model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
    #           verbose=10, early_stopping_rounds=10, eval_metric=f1_score_eval_sk)

    # gsearch = GridSearchCV(model, param_grid=parameters, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=1)

    gsearch = GridSearchCV(model, param_grid=parameters, cv=2, scoring=my_custom_gscv_scorer, verbose=2, n_jobs=n_jobs)

    gsearch.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=10,
                early_stopping_rounds=10, eval_metric=f1_score_eval_sk)

    # print('参数的最佳取值:{0}'.format(gsearch.best_params_))
    # print('最佳模型得分:{0}'.format(gsearch.best_score_))
    # print(gsearch.cv_results_['mean_test_score'])
    # print(gsearch.cv_results_['params'])

    # 在提交测试集上预测
    # 对测试集进行预测
    y_pred = gsearch.predict(X_test)
    final_test_f1_score = f1_score_eval_final_test_sk(y_test, y_pred);
    print('final_test_f1_score =', final_test_f1_score);
    logging.info('训练完毕');

    # # gbm_model.best_score['valid_0']['f1_score']
    # if init_gbm_model is not None:
    #     old_model_score = init_gbm_model.best_score['valid_0']['f1_score'];
    #     new_model_score = gbm_model.best_score['valid_0']['f1_score'];
    #     logging.debug('old_model_score=' + str(old_model_score) + ', new_model_score=' + str(new_model_score));
    #     if new_model_score <= old_model_score:
    #         return init_gbm_model;
    gsearch
    return gsearch


def train_by_light_gbm_by_one_day_03():
    # logging.info('训练开始, train_file_name=', train_file_name);
    dir_path = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/20201015181036traffic-fix/hd1/'

    # 训练集的加载
    current_history_attr_road_train_df = pd.read_csv(dir_path + '20190701.txt_hd1')
    current_history_attr_road_train_df = current_history_attr_road_train_df.append(
        pd.read_csv(dir_path + '20190702.txt_hd1'), ignore_index=True);

    # 将对验证集的加载
    current_history_attr_road_df_test_path = dir_path + '20190730.txt_hd1';
    current_history_attr_road_test_df = pd.read_csv(current_history_attr_road_df_test_path)

    X_train, y_train = my_feature_engineer(current_history_attr_road_train_df);
    X_test, y_test = my_feature_engineer(current_history_attr_road_test_df);

    n_class = 5;
    params = {
        'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'None',
        'num_leaves': 31,
        'num_class': n_class,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'nthread': -1,
        'verbose': -1
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_test, label=y_test)

    gbm_model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=5000,
        valid_sets=[dvalid],
        early_stopping_rounds=10,
        verbose_eval=10,
        feval=f1_score_eval,
        init_model=None,  # 如果init_model不为None，那么就是在init_model基础上接着训练
        keep_training_booster=True  # 增量训练
    )

    # 在提交测试集上预测
    y_test_pred = gbm_model.predict(X_test, num_iteration=gbm_model.best_iteration)  # 输出的是概率结果
    final_test_f1_score = f1_score_eval_final_test(y_test_pred, dvalid);
    print('final_test_f1_score =', final_test_f1_score);

    return gbm_model


def train_by_light_gbm_entry():
    dir_path = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/20201015181036traffic-fix/hd1/'

    # 查询所有的预处理后的训练集文件
    # root_dir = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/20201015181036traffic-fix'
    file_name_list = os.listdir(dir_path);
    file_name_list.sort();
    # handled_file_list = [];
    # for file_name in file_name_list:
    #     # if file_name.startswith('2019') & file_name.endswith('_hd1'):
    #     handled_file_list.append(file_name);

    file_name_list.remove('20190730.txt_hd1');
    file_name_list.remove('test.txt_hd1');

    # 依次在不同训练集上训练;
    gbm_model = None;
    for idx, file_name in enumerate(file_name_list):
        logging.info('开始在' + file_name + '训练...');
        # handled_file_path = dir + file_name;

        # if idx != 0:
        #     old_xgb_model_path = 'model/' + handled_file_list[idx - 1] + '_road_predict.pickle.dat'
        # train_and_save_model_use_one_day_data(file_name, handled_file_path, old_xgb_model_path);
        gbm_model = train_by_light_gbm_by_one_day(file_name, gbm_model)


def train_by_light_gbm_entry_02():
    dir_path = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/20201015181036traffic-fix/hd1/'

    # 查询所有的预处理后的训练集文件
    # root_dir = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/20201015181036traffic-fix'
    file_name_list = os.listdir(dir_path);
    file_name_list.sort();
    # handled_file_list = [];
    # for file_name in file_name_list:
    #     # if file_name.startswith('2019') & file_name.endswith('_hd1'):
    #     handled_file_list.append(file_name);

    # file_name_list.remove('20190730.txt_fp1');

    # 依次在不同训练集上训练;
    gbm_model = None;
    for idx, file_name in enumerate(file_name_list):
        logging.info('开始在' + file_name + '训练...');
        # handled_file_path = dir + file_name;

        # if idx != 0:
        #     old_xgb_model_path = 'model/' + handled_file_list[idx - 1] + '_road_predict.pickle.dat'
        # train_and_save_model_use_one_day_data(file_name, handled_file_path, old_xgb_model_path);
        # gbm_model = train_by_light_gbm_by_one_day_02(file_name);
        # gbm_model = train_by_light_gbm_by_one_day_02_adjust_params(file_name)
        gbm_model = train_by_light_gbm_by_one_day_with_second_model_offline_base(file_name);

        # if MODE == 'DEV':
        break;
        # if idx >=6:
        #     break;

    ''' '''
    # 在提交测试集上预测
    current_history_attr_road_df_test_path = dir_path + 'test.txt_hd1';
    current_history_attr_road_test_df = pd.read_csv(current_history_attr_road_df_test_path)
    X_test, y_test = my_feature_engineer(current_history_attr_road_test_df);
    # y_test_pred = gbm_model.predict(X_test, num_iteration=gbm_model.best_iteration)  # 输出的是概率结果
    y_test_pred = gbm_model.predict(X_test, num_iteration=gbm_model.best_iteration_)

    # y_test_pred = np.argmax(y_test_pred, axis=1);
    submission_df = X_test.loc[:, ['link_id', 'current_slice_id', 'future_slice_id']];
    submission_df['label'] = y_test_pred;

    # 为了适应lightGBM的label从0开始做的调整，在最终预测结果中，还需要加回去;
    submission_df['label'] = submission_df['label'] + 1;
    submission_df = submission_df.rename(columns={'link_id': 'link'});
    submission_df.to_csv('submission.csv', index=False);

    # 打印特征重要性
    fi_df = pd.DataFrame({
        'column': gbm_model.feature_name_,
        'importance': gbm_model.feature_importances_,
    }).sort_values(by='importance', ascending=False).reset_index(drop=True);
    print('fi_df =\n', fi_df);


def train_by_light_gbm_entry_02_adjust_params():
    _dir_path = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/20201015181036traffic-fix/hd1/'

    # 查询所有的预处理后的训练集文件
    # root_dir = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/20201015181036traffic-fix'
    file_name_list = os.listdir(_dir_path);
    file_name_list.sort();

    # 依次在不同训练集上训练;
    gbm_model = None;
    for idx, file_name in enumerate(file_name_list):
        logging.info('开始在' + file_name + '训练...');
        # handled_file_path = dir + file_name;
        # if idx != 0:
        #     old_xgb_model_path = 'model/' + handled_file_list[idx - 1] + '_road_predict.pickle.dat'
        # train_and_save_model_use_one_day_data(file_name, handled_file_path, old_xgb_model_path);
        # gbm_model = train_by_light_gbm_by_one_day_02(file_name);
        # gbm_model = train_by_light_gbm_by_one_day_02_adjust_params(file_name)

        # 调参
        gbm_model = train_by_light_gbm_by_one_day_02_adjust_params();
        break;

    # 在提交测试集上预测
    current_history_attr_road_df_test_path = _dir_path + 'test.txt_hd1';
    current_history_attr_road_test_df = pd.read_csv(current_history_attr_road_df_test_path)
    X_test, y_test = my_feature_engineer(current_history_attr_road_test_df);
    y_test_pred = gbm_model.predict(X_test)

    submission_df = X_test.loc[:, ['link_id', 'current_slice_id', 'future_slice_id']];
    submission_df['label'] = y_test_pred;

    # 为了适应lightGBM的label从0开始做的调整，在最终预测结果中，还需要加回去;
    submission_df['label'] = submission_df['label'] + 1;
    submission_df = submission_df.rename(columns={'link_id': 'link'});
    submission_df.to_csv('submission.csv', index=False);


def train_by_light_gbm_entry_03():
    dir_path = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/20201015181036traffic-fix/hd1/'

    # 查询所有的预处理后的训练集文件
    file_name_list = os.listdir(dir_path);
    file_name_list.sort();

    gbm_model = train_by_light_gbm_by_one_day_03();

    # 在提交测试集上预测
    current_history_attr_road_df_test_path = dir_path + 'test.txt_hd1';
    current_history_attr_road_test_df = pd.read_csv(current_history_attr_road_df_test_path)
    X_test, y_test = my_feature_engineer(current_history_attr_road_test_df);
    y_test_pred = gbm_model.predict(X_test, num_iteration=gbm_model.best_iteration)  # 输出的是概率结果

    y_test_pred = np.argmax(y_test_pred, axis=1);
    submission_df = X_test.loc[:, ['link_id', 'current_slice_id', 'future_slice_id']];
    submission_df['label'] = y_test_pred;
    submission_df.loc[submission_df.label == 0, 'label'] = 1;
    submission_df.loc[submission_df.label == 4, 'label'] = 3;
    submission_df = submission_df.rename(columns={'link_id': 'link'});
    submission_df.to_csv('submission.csv', index=False);

    # 打印特征重要性
    fi_df = pd.DataFrame({
        'column': X_test.columns,
        'importance': gbm_model.feature_importance(),
    }).sort_values(by='importance', ascending=False).reset_index(drop=True);
    print('fi_df =\n', fi_df);


def train_by_light_gbm_entry_02_with_second_model():
    print("train_by_light_gbm_entry_02_with_second_model is start...... ");
    _dir_path = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/20201015181036traffic-fix/hd4/';

    # 查询所有的预处理后的训练集文件
    file_name_list = os.listdir(_dir_path);
    file_name_list.sort();

    # 移出提交集
    file_name_list.remove('20190801_testdata.txt_hd1_hd4');

    # 依次在不同训练集上训练;
    gbm_model = None;
    for idx, file_name in enumerate(file_name_list):
        logging.info('开始在' + file_name + '训练...');
        if MODE_OFFLINE:
            gbm_model, rescale_weight = train_by_light_gbm_by_one_day_with_second_model_offline_base(file_name);
        else:
            gbm_model, rescale_weight = train_by_light_gbm_by_one_day_with_second_model_with_same_week_online_base()
        break;

    # 在提交测试集上预测
    # 20190801_testdata.txt_hd1
    # current_history_attr_road_df_test_path = _dir_path + 'test.txt_hd1';
    current_history_attr_road_df_test_path = _dir_path + '20190801_testdata.txt_hd1_hd4';
    current_history_attr_road_test_df = pd.read_csv(current_history_attr_road_df_test_path)
    X_test, y_test = my_feature_engineer(current_history_attr_road_test_df);
    y_test_pred_prob = gbm_model.predict_proba(X_test, num_iteration=gbm_model.best_iteration_)

    # 使用再缩放权重，重新预测结果
    m = y_test_pred_prob.shape[0];
    rescale_weight_np = np.repeat(np.expand_dims(np.array(rescale_weight), axis=1), repeats=m, axis=1).T
    rescale_y_test_pred = np.argmax(y_test_pred_prob * rescale_weight_np, axis=1)

    submission_df = X_test.loc[:, ['link_id', 'current_slice_id', 'future_slice_id']];
    submission_df['label'] = rescale_y_test_pred;

    # 为了适应lightGBM的label从0开始做的调整，在最终预测结果中，还需要加回去;
    submission_df['label'] = submission_df['label'] + 1;
    submission_df = submission_df.rename(columns={'link_id': 'link'});
    submission_df.to_csv('submission.csv', index=False);

    # 打印特征重要性
    fi_df = pd.DataFrame({
        'column': gbm_model.feature_name_,
        'importance': gbm_model.feature_importances_,
    }).sort_values(by='importance', ascending=False).reset_index(drop=True);
    print('fi_df =\n', fi_df);


def prediction_and_rescale(rescale_weight, model, X):
    """
    :param rescale_weight:
    :param model:
    :param X:
    :return: 返回再缩放后的概率值 m*n, m是样本个数，n是类别个数
    """

    # 使用对评估函数优化后的参数，重新缩放prediction，在测试集上验证
    y_pred_prob = model.predict_proba(X, num_iteration=model.best_iteration_)
    m = y_pred_prob.shape[0];
    rescale_weight_np = np.repeat(np.expand_dims(np.array(rescale_weight), axis=1), repeats=m, axis=1).T
    return y_pred_prob * rescale_weight_np;


def train_by_light_gbm_with_second_model(_dir_path, train_file_name_list, test_file_name, subm_file_name, model_prefix,
                                         model_id, old_model=None):
    """
    :param _dir_path:
    :param test_file_name:
    :param train_file_name_list:
    :param model_id:
    :param train_file_name:
    :param old_model: 之前训练的模型，加载后的类，不传参的话默认值是None，相当于从头开始训练; 传了值的话，相关于基于旧的模型开始训练
    :return:
    """

    path_dict = fea.init();
    my_save_dir = path_dict['my_save_dir'];

    print("train_by_light_gbm_with_second_model is startting ... ");
    print('model is ' + model_prefix + str(model_id) + ', train and prediction start .... ');
    # _dir_path = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/20201015181036traffic-fix/hd4/'

    # 训练集的加载
    current_history_attr_road_df_train_path = _dir_path + train_file_name_list[0];
    current_history_attr_road_train_df = pd.read_csv(current_history_attr_road_df_train_path)
    X_train, y_train = my_feature_engineer(current_history_attr_road_train_df);
    del current_history_attr_road_train_df;
    for _idx, train_file_name in enumerate(train_file_name_list):
        if _idx == 0:
            continue

        # _idx不为0的情况
        current_history_attr_road_df_train_path_02 = _dir_path + train_file_name;
        current_history_attr_road_train_df_02 = pd.read_csv(current_history_attr_road_df_train_path_02)

        X_train_tmp, y_train_tmp = my_feature_engineer(current_history_attr_road_train_df_02);
        del current_history_attr_road_train_df_02;

        X_train = X_train.append(X_train_tmp, ignore_index=True);
        del X_train_tmp;

        y_train = y_train.append(y_train_tmp, ignore_index=True);
        del y_train_tmp;

        print('X_train.shape =', X_train.shape);
        print('y_train.shape =', y_train.shape);

    print('final X_train.shape =' + str(X_train.shape));
    print('final y_train.shape =' + str(y_train.shape));

    # 将对验证集的加载
    current_history_attr_road_df_test_path = _dir_path + test_file_name;
    current_history_attr_road_test_df = pd.read_csv(current_history_attr_road_df_test_path)
    X_test, y_test = my_feature_engineer(current_history_attr_road_test_df);
    del current_history_attr_road_test_df;

    n_class = 3
    model = lgb.LGBMClassifier(
        learning_rate=0.05,
        boosting_type='gbdt',
        objective='multiclass',
        num_leaves=31,
        max_depth=6,
        num_class=n_class,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        seed=1,  # 随机数种子
        bagging_seed=1,
        feature_fraction_seed=7,
        min_data_in_leaf=20,
        nthread=-1,
        verbose=-1,
        num_boost_round=5000
        # class_weight={0: 6.792153871, 1: 0.768032069, 2: 1}
        # class_weight={0: 1, 1: 1, 2: 3}
        # n_estimators=200,  # 使用多少个弱分类器
        # num_class=3,
        # booster='gbtree',
        # min_child_weight=2,
        # subsample=0.8,
        # colsample_bytree=0.8,
        # reg_alpha=0,
        # reg_lambda=1
    )

    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=10, early_stopping_rounds=10, eval_metric=f1_score_eval_sk, init_model=old_model)

    # 针对评估函数，进行优化;
    bnds = ((0, 1000), (0, 1000), (0, 1000))
    w = [1.0, 1.0, 1.0];

    y_train_pred_prob = model.predict_proba(X_train, num_iteration=model.best_iteration_)
    del X_train

    partial_model_metric_loss_func = partial(model_metric_loss_func, y_true=y_train, y_pred_prob=y_train_pred_prob);
    res = minimize(partial_model_metric_loss_func, w, method='Powell', bounds=bnds)
    print('res.fun = ' + str(res.fun) + ', res.success = ' + str(res.success) + " , res.x = " + str(res.x))
    del y_train;
    del y_train_pred_prob;

    # 在提交测试集上预测
    # 对测试集进行预测
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration_)
    final_test_f1_score = f1_score_eval_final_test_sk(y_test, y_test_pred);
    print('before optimize final_test_f1_score =', final_test_f1_score);
    del y_test_pred;

    # 使用对评估函数优化后的参数，重新缩放prediction，在测试集上验证
    rescale_weight = res.x;
    rescale_y_test_pred = np.argmax(prediction_and_rescale(rescale_weight, model, X_test), axis=1)
    rescale_final_test_f1_score = f1_score_eval_final_test_sk(y_test, rescale_y_test_pred);
    del X_test;
    del y_test;
    del rescale_y_test_pred;
    print('after optimize final_test_f1_score =', rescale_final_test_f1_score);

    print('save this batch model and rescale weight ..')
    print('model is ' + model_prefix + str(model_id) + ', rescale_weight=' + str(rescale_weight));

    model_dir = my_save_dir + 'model/';
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    pickle.dump(model, open(model_dir + "lightgbm_model_" + model_prefix + str(model_id) + ".pickle.dat", "wb"));

    # 在提交测试集上预测
    print('start to predict at submission data');
    current_history_attr_road_df_subm_path = _dir_path + subm_file_name;
    current_history_attr_road_subm_df = pd.read_csv(current_history_attr_road_df_subm_path)
    X_subm, y_subm = my_feature_engineer(current_history_attr_road_subm_df);
    if MODE == 'DEV':
        current_history_attr_road_subm_df = current_history_attr_road_subm_df.iloc[0:1000, :];

    # 使用再缩放权重，重新预测结果
    rescale_y_test_pred = np.argmax(prediction_and_rescale(rescale_weight, model, X_subm), axis=1)
    submission_df = current_history_attr_road_subm_df.loc[:, ['link_id', 'current_slice_id', 'future_slice_id']];
    submission_df['label'] = rescale_y_test_pred;

    # 为了适应lightGBM的label从0开始做的调整，在最终预测结果中，还需要加回去;
    submission_df['label'] = submission_df['label'] + 1;
    submission_df = submission_df.rename(columns={'link_id': 'link'});

    # path_dict['result_path'] = result_path;
    result_path = path_dict['result_path'];

    # /data/prediction_result/result.csv
    submission_df.to_csv(result_path + 'result.csv', index=False);
    del X_subm;
    del y_subm;
    del rescale_y_test_pred;

    # 打印特征重要性
    fi_df = pd.DataFrame({
        'column': model.feature_name_,
        'importance': model.feature_importances_,
    }).sort_values(by='importance', ascending=False).reset_index(drop=True);
    print('fi_df =\n', fi_df);
    print('model is ' + model_prefix + str(model_id) + ', train and prediction is over ...');


def resacle_weight_4_balanced_distribution(w1, w2, w3, y_pred_prob, X):
    """
    对训练集、测试集各预测数据分类， 分布不一致的预测概率微调
    :param w1:
    :param w2:
    :param w3:
    :param y_pred_prob:
    :param X:
    :return:
    """
    y_pred_prob[:, 0] = y_pred_prob[:, 0] * w1
    y_pred_prob[X['path_class'] == 5, 0] = y_pred_prob[X['path_class'] == 5, 0] * w2
    y_pred_prob[X['path_class'] == 2, 1] = y_pred_prob[X['path_class'] == 2, 1] * w3
    return y_pred_prob


def train_by_light_gbm_with_second_model_with_balanced_distribution(_dir_path, train_file_name_list, test_file_name,
                                                                    subm_file_name, model_prefix,
                                                                    model_id, old_model=None):
    """
    :param _dir_path:
    :param test_file_name:
    :param train_file_name_list:
    :param model_id:
    :param train_file_name:
    :param old_model: 之前训练的模型，加载后的类，不传参的话默认值是None，相当于从头开始训练; 传了值的话，相关于基于旧的模型开始训练
    :return:
    """

    path_dict = fea.init();
    my_save_dir = path_dict['my_save_dir'];

    print("train_by_light_gbm_with_second_model_with_balanced_distribution is startting ... ");
    print('model is ' + model_prefix + str(model_id) + ', train and prediction start .... ');
    # _dir_path = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/20201015181036traffic-fix/hd4/'

    # 训练集的加载
    current_history_attr_road_df_train_path = _dir_path + train_file_name_list[0];
    current_history_attr_road_train_df = pd.read_csv(current_history_attr_road_df_train_path)
    X_train, y_train = my_feature_engineer(current_history_attr_road_train_df);
    del current_history_attr_road_train_df;
    for _idx, train_file_name in enumerate(train_file_name_list):
        if _idx == 0:
            continue

        # _idx不为0的情况
        current_history_attr_road_df_train_path_02 = _dir_path + train_file_name;
        current_history_attr_road_train_df_02 = pd.read_csv(current_history_attr_road_df_train_path_02)

        X_train_tmp, y_train_tmp = my_feature_engineer(current_history_attr_road_train_df_02);
        del current_history_attr_road_train_df_02;

        X_train = X_train.append(X_train_tmp, ignore_index=True);
        del X_train_tmp;

        y_train = y_train.append(y_train_tmp, ignore_index=True);
        del y_train_tmp;

        print('X_train.shape =', X_train.shape);
        print('y_train.shape =', y_train.shape);

    print('final X_train.shape =' + str(X_train.shape));
    print('final y_train.shape =' + str(y_train.shape));

    # 将对验证集的加载
    current_history_attr_road_df_test_path = _dir_path + test_file_name;
    current_history_attr_road_test_df = pd.read_csv(current_history_attr_road_df_test_path)
    X_test, y_test = my_feature_engineer(current_history_attr_road_test_df);
    del current_history_attr_road_test_df;

    n_class = 3
    model = lgb.LGBMClassifier(
        learning_rate=LEARNNING_RATE,
        boosting_type='gbdt',
        objective='multiclass',
        num_leaves=31,
        max_depth=6,
        num_class=n_class,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        seed=1,  # 随机数种子
        bagging_seed=1,
        feature_fraction_seed=7,
        min_data_in_leaf=20,
        nthread=-1,
        verbose=-1,
        num_boost_round=5000
        # class_weight={0: 6.792153871, 1: 0.768032069, 2: 1}
        # class_weight={0: 1, 1: 1, 2: 3}
        # n_estimators=200,  # 使用多少个弱分类器
        # num_class=3,
        # booster='gbtree',
        # min_child_weight=2,
        # subsample=0.8,
        # colsample_bytree=0.8,
        # reg_alpha=0,
        # reg_lambda=1
    )

    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=10, early_stopping_rounds=10, eval_metric=f1_score_eval_sk, init_model=old_model)

    # 针对评估函数，进行优化;
    bnds = ((0, 1000), (0, 1000), (0, 1000))
    w = [1.0, 1.0, 1.0];

    y_train_pred_prob = model.predict_proba(X_train, num_iteration=model.best_iteration_)
    del X_train

    partial_model_metric_loss_func = partial(model_metric_loss_func, y_true=y_train, y_pred_prob=y_train_pred_prob);
    res = minimize(partial_model_metric_loss_func, w, method='Powell', bounds=bnds)
    print('res.fun = ' + str(res.fun) + ', res.success = ' + str(res.success) + " , res.x = " + str(res.x))
    del y_train;
    del y_train_pred_prob;

    # 在提交测试集上预测
    # 对测试集进行预测
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration_)
    final_test_f1_score = f1_score_eval_final_test_sk(y_test, y_test_pred);
    print('before optimize final_test_f1_score =', final_test_f1_score);
    del y_test_pred;

    # 使用对评估函数优化后的参数，重新缩放prediction，在测试集上验证
    rescale_weight = res.x;
    rescale_y_test_pred = np.argmax(prediction_and_rescale(rescale_weight, model, X_test), axis=1)
    rescale_final_test_f1_score = f1_score_eval_final_test_sk(y_test, rescale_y_test_pred);
    del X_test;
    del y_test;
    del rescale_y_test_pred;
    print('after optimize final_test_f1_score =', rescale_final_test_f1_score);

    print('save this batch model and rescale weight ..')
    print('model is ' + model_prefix + str(model_id) + ', rescale_weight=' + str(rescale_weight));

    model_dir = my_save_dir + 'model/';
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    pickle.dump(model, open(model_dir + "lightgbm_model_" + model_prefix + str(model_id) + ".pickle.dat", "wb"));

    # 在提交测试集上预测
    print('start to predict at submission data');
    current_history_attr_road_df_subm_path = _dir_path + subm_file_name;
    current_history_attr_road_subm_df = pd.read_csv(current_history_attr_road_df_subm_path)
    X_subm, y_subm = my_feature_engineer(current_history_attr_road_subm_df);
    if MODE == 'DEV':
        current_history_attr_road_subm_df = current_history_attr_road_subm_df.iloc[0:1000, :];

    # 使用再缩放权重，重新预测结果
    # rescale_y_test_pred = np.argmax(prediction_and_rescale(rescale_weight, model, X_subm), axis=1)
    y_sumb_pred_prob = prediction_and_rescale(rescale_weight, model, X_subm);

    # 均衡分布的调整
    w1, w2, w3 = 0.67, 1.29, 1.1
    y_sumb_pred_prob = resacle_weight_4_balanced_distribution(w1, w2, w3, y_sumb_pred_prob, X_subm)
    rescale_y_test_pred = np.argmax(y_sumb_pred_prob, axis=1)

    submission_df = current_history_attr_road_subm_df.loc[:, ['link_id', 'current_slice_id', 'future_slice_id']];
    submission_df['label'] = rescale_y_test_pred;

    # 为了适应lightGBM的label从0开始做的调整，在最终预测结果中，还需要加回去;
    submission_df['label'] = submission_df['label'] + 1;
    submission_df = submission_df.rename(columns={'link_id': 'link'});

    # path_dict['result_path'] = result_path;
    result_path = path_dict['result_path'];

    # /data/prediction_result/result.csv
    submission_df.to_csv(result_path + 'result.csv', index=False);
    del X_subm;
    del y_subm;
    del rescale_y_test_pred;

    # 打印特征重要性
    fi_df = pd.DataFrame({
        'column': model.feature_name_,
        'importance': model.feature_importances_,
    }).sort_values(by='importance', ascending=False).reset_index(drop=True);
    print('fi_df =\n', fi_df);
    print('model is ' + model_prefix + str(model_id) + ', train and prediction is over ...');


def train_by_light_gbm_entry_02_with_second_model_multi_train_datasets():
    """
    模型融合
    :return:
    """
    print(str(
        datetime.datetime.now()) + ' train_by_light_gbm_entry_02_with_second_model_multi_train_datasets is running ...')
    model_prefix = "14_";

    train_file_name_list = ['20190701.txt_hd1_hd4'];
    test_file_name = '20190730.txt_hd1_hd4';
    subm_file_name = '20190801_testdata.txt_hd1_hd4';
    model_id = 2;

    _dir_path = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/20201015181036traffic-fix/hd4/';
    file_name_list = os.listdir(_dir_path);
    file_name_list.sort();
    file_name_list.remove('20190801_testdata.txt_hd1_hd4');
    file_name_list.insert(0, 'fill_index_0_element')

    idx_np = np.array([[3, 10, 17, 24, 25],
                       [5, 12, 19, 26, 25],
                       [6, 13, 20, 27, 25],
                       [7, 14, 21, 28, 25],
                       [1, 8, 15, 22, 25],
                       [2, 9, 16, 23, 25]
                       ]);
    for idx in [0, 1, 2, 3, 4, 5]:
        train_file_name_list = [file_name_list[_id] for _id in idx_np[idx]]
        print('train_file_name_list =', train_file_name_list);
        train_by_light_gbm_with_second_model(train_file_name_list, test_file_name, subm_file_name,
                                             model_prefix, model_id);
        model_id = model_id + 1;

    # train_file_name_list = ['20190701.txt_hd1_hd4', '20190702.txt_hd1_hd4'];
    # 在第1批数据集上训练
    # train_file_name_list = ['20190701.txt_hd1_hd4'];
    # train_file_name_list = ['20190701.txt_hd1_hd4', '20190702.txt_hd1_hd4'];
    # train_file_name_list = ['20190704.txt_hd1_hd4', '20190711.txt_hd1_hd4', '20190718.txt_hd1_hd4',
    #                         '20190724.txt_hd1_hd4', '20190725.txt_hd1_hd4'];

    # # 在第2批数据集上训练
    # train_file_name_list = ['20190703.txt_hd1_hd4'];
    # model_id = model_id + 1;
    # train_by_light_gbm_with_second_model_gkfold(train_file_name_list, test_file_name, subm_file_name,
    #                                      model_prefix, model_id);


def train_by_light_gbm_entry_02_with_second_model_offline_entry():
    pass
    print(str(
        datetime.datetime.now()) + ' train_by_light_gbm_entry_02_with_second_model_offline_entry is running ...')

    model_prefix = "33_";
    test_file_name = '20190730.txt_hd1_hd4';
    subm_file_name = '20190801_testdata.txt_hd1_hd4';
    model_id = 1;

    _dir_path = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/20201015181036traffic-fix/hd4/';
    file_name_list = os.listdir(_dir_path);
    file_name_list.sort();
    file_name_list.remove('20190801_testdata.txt_hd1_hd4');
    file_name_list.insert(0, 'fill_index_0_element')

    if MODE == 'DEV':
        train_file_name_list = ['20190701.txt_hd1_hd4']
    else:
        train_file_name_list = [file_name_list[_id] for _id in [4, 11, 18, 25]]

    print('train_file_name_list =', train_file_name_list);
    train_by_light_gbm_with_second_model(_dir_path, train_file_name_list, test_file_name, subm_file_name,
                                         model_prefix, model_id);


def train_by_light_gbm_entry_02_with_second_model_offline_entry_v02():
    print(str(
        datetime.datetime.now()) + ' train_by_light_gbm_entry_02_with_second_model_offline_entry_v02 is running ...')

    model_prefix = "57_";
    test_file_name = '20190730.txt_hd1_hd4_hd6';
    subm_file_name = '20190801_testdata.txt_hd1_hd4_hd6';
    model_id = 1;

    path_dict = fea.init();
    my_save_dir = path_dict['my_save_dir'];

    # _dir_path = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/20201015181036traffic-fix/hd6/';
    _dir_path = my_save_dir + 'hd6/';

    if MODE == 'DEV':
        train_file_name_list = ['20190704.txt_hd1_hd4_hd6']
    elif MODE == 'PARAMS':
        train_file_name_list = ['20190704.txt_hd1_hd4_hd6']
    else:
        train_file_name_list = [
            '20190704.txt_hd1_hd4_hd6',
            '20190711.txt_hd1_hd4_hd6',
            '20190718.txt_hd1_hd4_hd6',
            '20190725.txt_hd1_hd4_hd6',
            '20190726.txt_hd1_hd4_hd6'
        ]

    print('train_file_name_list =', train_file_name_list);
    train_by_light_gbm_with_second_model(_dir_path, train_file_name_list, test_file_name, subm_file_name,
                                         model_prefix, model_id);


def train_by_light_gbm_entry_02_with_second_model_offline_entry_v03():
    print(str(
        datetime.datetime.now()) + ' train_by_light_gbm_entry_02_with_second_model_offline_entry_v03 is running ...')

    model_prefix = "59_";
    test_file_name = '20190730.txt_hd1_hd4_hd6';
    subm_file_name = '20190801_testdata.txt_hd1_hd4_hd6';
    model_id = 1;

    path_dict = fea.init();
    my_save_dir = path_dict['my_save_dir'];

    # _dir_path = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/20201015181036traffic-fix/hd6/';
    _dir_path = my_save_dir + 'hd6/';

    if MODE == 'DEV':
        train_file_name_list = ['20190704.txt_hd1_hd4_hd6']
    elif MODE == 'PARAMS':
        train_file_name_list = ['20190704.txt_hd1_hd4_hd6']
    else:
        train_file_name_list = [
            '20190704.txt_hd1_hd4_hd6',
            '20190711.txt_hd1_hd4_hd6',
            '20190718.txt_hd1_hd4_hd6',
            '20190725.txt_hd1_hd4_hd6',
            '20190726.txt_hd1_hd4_hd6'
        ]
        # train_file_name_list = [
        #     '20190704.txt_hd1_hd4_hd6'
        # ]

    print('train_file_name_list =', train_file_name_list);
    train_by_light_gbm_with_second_model_with_balanced_distribution(_dir_path, train_file_name_list, test_file_name,
                                                                    subm_file_name,
                                                                    model_prefix, model_id);


def handle_one_day_road_data(data_path, attr_df):
    """
    某一天的路况数据，逐行解析，拆分成一个dataframe，并保存到硬盘中;
    :param data_path: 某一天的路况数据绝对路径
    :param attr_df: 路段的属性集合
    :return:
    """
    print(data_path + ', prehandle is start...');

    # 整体读取成仅有一列的df
    current_history_lukuang = pd.read_csv(data_path, sep='\t', header=None);
    print(current_history_lukuang.head(3));
    print(current_history_lukuang.shape)

    split_2_struct_data_02(current_history_lukuang, attr_df, data_path + '_fp1');
    print(data_path + ', prehandle end ...');


def handle_all_data(attr_df):
    pass;
    data_dir = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/20201015181036traffic-fix'
    file_name_list = os.listdir(data_dir);
    file_name_list.sort()
    for file_name in file_name_list:
        if file_name.startswith('2019'):
            data_path = data_dir + "/" + file_name;
            handle_one_day_road_data(data_path, attr_df)


MODE = 'PRD';
if __name__ == '__main__':
    starttime = datetime.datetime.now()

    # 是开发测试、还是真实全量数据运行
    # MODE = 'DEV';
    MODE = 'PRD';
    # MODE = 'PARAMS';
    print('MODE =', MODE)

    param_num = len(sys.argv)
    print('param_num =', param_num)
    if param_num != 1:
        LEARNNING_RATE = float(sys.argv[1]);
    else:
        LEARNNING_RATE = 0.05;
    print('learnning_rate =', LEARNNING_RATE);

    # 是线下、还是线上的运行版本
    MODE_OFFLINE = True;
    # MODE_OFFLINE = False;

    init_pandas_show();
    init_logger();

    print('hello road speed prediction...., ' + str(starttime))
    # train_by_light_gbm_entry_02_with_second_model_offline_entry_v02()
    train_by_light_gbm_entry_02_with_second_model_offline_entry_v03()

    endtime = datetime.datetime.now();
    print('endtime=' + str(endtime) + ', total running time :' + str((endtime - starttime).seconds))
