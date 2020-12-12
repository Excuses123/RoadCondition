# -*- coding: utf-8 -*-

import pandas as pd
import datetime
import math
import os
import logging


def init():
    # 文件data路径
    if MODE == 'DEV' or MODE == 'PRD':
        data_dir = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/20201015181036traffic-fix/data/'
    else:
        # 相对于code所在路径的上一级; data\code\zhusc
        # data_dir = '../../';
        data_dir = '/data/';

    # 原始文件路径
    raw_dir = data_dir + 'raw_data/'

    # 属性文件路径
    attr_path = raw_dir + 'attr.txt'

    # embedding文件路径; data/user_data/Excuses/output
    # embedding_path = data_dir + 'user_data/Excuses/output/embedding_495.txt'
    embedding_path = data_dir + 'user_data/Excuses/output/embedding_linkid.txt'

    my_save_dir = data_dir + 'user_data/zhusc/'

    result_path = data_dir + 'prediction_result/';
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    handle_file_list = ['20190704.txt', '20190711.txt', '20190718.txt', '20190725.txt', '20190726.txt',
                       '20190730.txt', '20190801_testdata.txt'];
    # handle_file_list = ['20190704.txt'];

    path_dict = {'raw_dir': raw_dir,
                 'attr_path': attr_path,
                 'embedding_path': embedding_path,
                 'my_save_dir': my_save_dir,
                 'handle_file_list': handle_file_list,
                 'result_path': result_path};

    return path_dict;


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


def split_2_struct_data_02(current_history_road, attr_df, save_path):
    """
    之前存在的问题，行数太多，会OOM
    :param current_history_road:
    :param attr_df: 路段的属性集合
    :return:
    """

    row_num = current_history_road.shape[0];
    if BATCH_SIZE is None:
        batch_size = 20000;
    else:
        batch_size = BATCH_SIZE;

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


def split_2_struct_data_02(current_history_road, attr_df, save_path):
    """
    之前存在的问题，行数太多，会OOM
    :param current_history_road:
    :param attr_df: 路段的属性集合
    :return:
    """

    row_num = current_history_road.shape[0];
    if BATCH_SIZE is None:
        batch_size = 20000;
    else:
        batch_size = BATCH_SIZE;
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


# data_path, attr_df, file_name, hd1_path_root
def handle_one_day_road_data(data_path, attr_df, file_name, hd1_path_root):
    """
    某一天的路况数据，逐行解析，拆分成一个dataframe，并保存到硬盘中;
    :param data_path: 某一天的路况数据绝对路径
    :param attr_df: 路段的属性集合
    :return:
    """
    print(data_path + ', prehandle is starting ...');

    # 整体读取成仅有一列的df
    current_history_lukuang = pd.read_csv(data_path, sep='\t', header=None);
    print(current_history_lukuang.head(3));
    print(current_history_lukuang.shape)

    save_path = hd1_path_root + file_name + '_hd1';
    print('save_path =', save_path)
    split_2_struct_data_02(current_history_lukuang, attr_df, save_path);
    print(data_path + ', prehandle is end ...');


def handle_all_data(attr_df):
    pass;
    data_dir = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/20201015181036traffic-fix'
    file_name_list = os.listdir(data_dir);
    file_name_list.sort()
    for file_name in file_name_list:
        if file_name.startswith('2019'):
            data_path = data_dir + "/" + file_name;
            handle_one_day_road_data(data_path, attr_df)


def raw_2_hd1():
    path_dict = init();

    raw_dir = path_dict['raw_dir'];
    attr_path = path_dict['attr_path'];
    my_save_dir = path_dict['my_save_dir'];
    handle_file_list = path_dict['handle_file_list'];

    hd1_path_root = my_save_dir + 'hd1/'
    if not os.path.exists(hd1_path_root):
        os.makedirs(hd1_path_root)

    if MODE == 'DEV':
        handle_file_name_list = [handle_file_list[0]]
    else:
        handle_file_name_list = handle_file_list

    handle_file_name_list.sort()
    print('handle_file_name_list =', handle_file_name_list);

    attr_df = pd.read_csv(attr_path, sep='	');
    attr_df.columns = ['link_id', 'length', 'direction', 'path_class', 'speed_class', 'lane_num', 'speed_limit',
                       'level', 'width']
    print('attr_df.head(5) =\r\n', attr_df.head(3));

    # raw_dir = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/20201015181036traffic-fix/'
    for file_name in handle_file_name_list:
        # data/raw_data/traffic
        data_path = raw_dir + 'traffic/' + file_name;
        print('raw_2_hd1 is handling ', data_path)
        handle_one_day_road_data(data_path, attr_df, file_name, hd1_path_root)


def set_row_data(hist_label_cnt_index_np, hist_label_cnt, label, row_data):
    # 统计结果中label存在情况
    if (hist_label_cnt_index_np == label).sum() != 0:
        row_data[label] = hist_label_cnt[label];
    return row_data;


def count_and_norm_lable_prop(label_list):
    pass
    label_cnt = pd.value_counts(label_list);
    lable_sum = label_cnt.values.sum()
    if lable_sum == 0:
        return [0, 0, 0];
    else:
        label_cnt_index_np = label_cnt.index.values;
        row_data = [0, 0, 0, 0, 0];
        row_data = set_row_data(label_cnt_index_np, label_cnt, 0, row_data)
        row_data = set_row_data(label_cnt_index_np, label_cnt, 1, row_data)
        row_data = set_row_data(label_cnt_index_np, label_cnt, 2, row_data)
        row_data = set_row_data(label_cnt_index_np, label_cnt, 3, row_data)
        row_data = set_row_data(label_cnt_index_np, label_cnt, 4, row_data)

        row_data = [val / lable_sum for val in row_data];

        # 0, 1 --> 属于标签1; 2--> 属于标签2; 3, 4 --> 属于标签3;
        return [row_data[0] + row_data[1], row_data[2], row_data[3] + row_data[4]];


def add_count_label_feature(x_df):
    x_df = x_df.reset_index(drop=True);
    row_label_prop_list = [];
    for index, row in x_df.iterrows():
        hist_label_list = [
            row['label_-28_0'], row['label_-28_1'], row['label_-28_2'], row['label_-28_3'], row['label_-28_4'],
            row['label_-21_0'], row['label_-21_1'], row['label_-21_2'], row['label_-21_3'], row['label_-21_4'],
            row['label_-14_0'], row['label_-14_1'], row['label_-14_2'], row['label_-14_3'], row['label_-14_4'],
            row['label_-7_0'], row['label_-7_1'], row['label_-7_2'], row['label_-7_3'], row['label_-7_4']];
        history_label_cnt_row_data = count_and_norm_lable_prop(hist_label_list);

        current_label_list = [row['label_0_0'], row['label_0_1'], row['label_0_2'], row['label_0_3'], row['label_0_4']]
        current_label_cnt_row_data = count_and_norm_lable_prop(current_label_list);

        # 合并两个list
        current_label_cnt_row_data.extend(history_label_cnt_row_data);

        row_label_prop_list.append(current_label_cnt_row_data);

    x_df[['current_label_1_prop', 'current_label_2_prop', 'current_label_3_prop',
          'history_label_1_prop', 'history_label_2_prop', 'history_label_3_prop']] = \
        pd.DataFrame(data=row_label_prop_list,
                     columns=['current_label_1_prop', 'current_label_2_prop', 'current_label_3_prop',
                              'history_label_1_prop', 'history_label_2_prop', 'history_label_3_prop']);
    return x_df;


def handle_one_day_road_data_hd1_2_hd4(data_path, save_path):
    """
    在hd1基础上，新增3个特征，历史上3个label的占比; 之前存在的问题，行数太多，会OOM
    :param data_path:
    :return:
    """

    print('hd1_2_hd4 is handling ', data_path);
    hd1_df = pd.read_csv(data_path);

    row_num = hd1_df.shape[0];

    if BATCH_SIZE is None:
        batch_size = 20000;
    else:
        batch_size = BATCH_SIZE;
    if MODE == 'DEV':
        batch_size = 100;

    total_batch = math.ceil(row_num / batch_size);

    start_row_id = 0;
    end_row_id = 0;
    for batch_id in range(total_batch + 1):
        print('batch_id = ' + str(batch_id) + ', total_batch = ' + str(total_batch));
        start_row_id = batch_id * batch_size;

        # 右区间远大于总行数，没有问题的。
        end_row_id = start_row_id + batch_size;

        # 只选取n行，进行统计
        hd1_part_df = hd1_df.iloc[start_row_id: end_row_id, :]
        # 说明此时是最末一轮，且查询出来为空的情况，结束循环;
        if len(hd1_part_df) == 0:
            break;

        hd1_part_df = add_count_label_feature(hd1_part_df)

        # 保存结果
        if batch_id == 0:
            hd1_part_df.to_csv(save_path, index=False, mode='a', header=True);
        else:
            hd1_part_df.to_csv(save_path, index=False, mode='a', header=False)

        # 删除临时变量，腾出空间
        del hd1_part_df;


def hd1_2_hd4():
    print('start hd1_2_hd4 ...... ')

    path_dict = init();
    my_save_dir = path_dict['my_save_dir'];
    handle_file_list = path_dict['handle_file_list'];
    hd1_path_root = my_save_dir + 'hd1/'
    hd4_path_root = my_save_dir + 'hd4/'

    if not os.path.exists(hd4_path_root):
        os.makedirs(hd4_path_root)

    if MODE == 'DEV':
        handle_file_name_list = [handle_file_list[0]]
    else:
        handle_file_name_list = handle_file_list

    print('handle_file_name_list =', handle_file_name_list);

    # save_path = hd1_path_root + file_name + '_hd1';
    for file_name in handle_file_name_list:
        hd1_file_path = hd1_path_root + file_name + '_hd1';
        save_path = hd4_path_root + file_name + '_hd1_hd4';
        print('from: hd1_file_path =', hd1_file_path)
        print('to:   hd4_file_path =', save_path)
        handle_one_day_road_data_hd1_2_hd4(hd1_file_path, save_path)


def handle_one_day_road_data_hd4_2_hd6(file_path, embedding_path, save_path):
    # _dir_path = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/20201015181036traffic-fix/hd4/'
    # _save_path = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/20201015181036traffic-fix/hd6/'
    # tp_path = 'F:/DeskRoot/learn/df/didichuxing_lukuangyuce/20201015181036traffic-fix/embedding_495.txt'
    tp_path = embedding_path;

    print('hd4_2_hd6, file_path=', file_path);

    hd4_df = pd.read_csv(file_path);
    tp_df = pd.read_csv(tp_path);
    tp_df.columns = ['link_id', 'tp_em_0', 'tp_em_1', 'tp_em_2', 'tp_em_3', 'tp_em_4', 'tp_em_5', 'tp_em_6', 'tp_em_7',
                     'tp_em_8',
                     'tp_em_9', 'tp_em_10', 'tp_em_11', 'tp_em_12', 'tp_em_13', 'tp_em_14', 'tp_em_15', 'tp_em_16',
                     'tp_em_17',
                     'tp_em_18', 'tp_em_19', 'tp_em_20', 'tp_em_21', 'tp_em_22', 'tp_em_23', 'tp_em_24', 'tp_em_25',
                     'tp_em_26',
                     'tp_em_27', 'tp_em_28', 'tp_em_29', 'tp_em_30', 'tp_em_31', 'tp_em_32', 'tp_em_33', 'tp_em_34',
                     'tp_em_35',
                     'tp_em_36', 'tp_em_37', 'tp_em_38', 'tp_em_39', 'tp_em_40', 'tp_em_41', 'tp_em_42', 'tp_em_43',
                     'tp_em_44',
                     'tp_em_45', 'tp_em_46', 'tp_em_47', 'tp_em_48', 'tp_em_49', 'tp_em_50', 'tp_em_51', 'tp_em_52',
                     'tp_em_53',
                     'tp_em_54', 'tp_em_55', 'tp_em_56', 'tp_em_57', 'tp_em_58', 'tp_em_59', 'tp_em_60', 'tp_em_61',
                     'tp_em_62',
                     'tp_em_63']

    hd6_df = pd.merge(hd4_df, tp_df, on=['link_id'], how='left');
    hd6_df.to_csv(save_path, index=False);


def hd4_2_hd6():
    print('start hd4_2_hd6......')
    path_dict = init();
    embedding_path = path_dict['embedding_path'];
    my_save_dir = path_dict['my_save_dir'];
    handle_file_list = path_dict['handle_file_list'];

    hd4_path_root = my_save_dir + 'hd4/'
    hd6_path_root = my_save_dir + 'hd6/'
    if not os.path.exists(hd6_path_root):
        os.makedirs(hd6_path_root)

    if MODE == 'DEV':
        handle_file_name_list = [handle_file_list[0]]
    else:
        handle_file_name_list = handle_file_list

    print('embedding_path =', embedding_path)
    print('handle_file_name_list =', handle_file_name_list);

    for file_name in handle_file_name_list:
        hd4_file_path = hd4_path_root + file_name + '_hd1_hd4';
        save_path = hd6_path_root + file_name + '_hd1_hd4_hd6';
        print('from: hd4_file_path =', hd4_file_path)
        print('to:   hd6_file_path =', save_path)
        handle_one_day_road_data_hd4_2_hd6(hd4_file_path, embedding_path, save_path)


# MODE = 'DEV';
MODE = 'FX';
if __name__ == '__main__':
    starttime = datetime.datetime.now()

    # 是开发测试、还是真实全量数据运行
    # MODE = 'DEV';
    # MODE = 'PRD';

    # 最终线上复现时，这个打开
    MODE = 'FX';
    print('MODE =', MODE);

    # 预处理过程，一批处理数据的条数
    if MODE == 'FX':
        BATCH_SIZE = 60000;
    else:
        BATCH_SIZE = 20000;

    # 修改init函数中的路径信息 init();

    raw_2_hd1();
    hd1_2_hd4();
    hd4_2_hd6();

    endtime = datetime.datetime.now();
    logging.info('total running time :' + str((endtime - starttime).seconds));