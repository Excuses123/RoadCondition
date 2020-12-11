import os
import datetime
import pandas as pd
import numpy as np
import collections
from tqdm import tqdm
import tensorflow as tf
from argparse import ArgumentParser


def parse_command_params():
    ap = ArgumentParser()
    ap.add_argument('-user_path', default="./data/user_data/Excuses", type=str, help='user out path')
    args = vars(ap.parse_args())

    return args

def Int64List(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

def FloatList(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))

def Byte64List(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def save2TFrecord(read_path, save_path, file):
    datekey = file.split(".")[0].split('_')[0]
    weekday = datetime.datetime.strptime(datekey, "%Y%m%d").weekday()
    workday = 0 if weekday in (5, 6) else 1
    writer = tf.python_io.TFRecordWriter(os.path.join(save_path, '%s.tfrecord') % datekey)
    print("save tfrecord of file %s !" % os.path.join(read_path, file))

    scale = np.array([723, 124, 149, 4, 120])
    with open(os.path.join(read_path, file), "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            rows = line.strip().split(";")
            linkid, label, current_slice_id, future_slice_id = list(map(lambda x: int(x), rows[0].split(" ")))

            if future_slice_id < 0 or future_slice_id > 719:
                continue
            if current_slice_id < 0 or current_slice_id > 719:
                continue

            future_slice = future_slice_id / 719
            current_slice = current_slice_id / 719
            slice_diff = future_slice - current_slice

            features = []
            for row in rows[1:]:
                # 路况速度,eta速度,路况状态,参与路况计算的车辆数
                feat = np.array([i.replace(':', ',').split(",") for i in row.split(" ")]).astype('float32').tolist()
                features.append(feat)
            features = np.array(features) / scale

            features_ = np.concatenate([
                features.max(axis=1, keepdims=True).reshape((5, 5, 1)),
                features.min(axis=1, keepdims=True).reshape((5, 5, 1)),
                features.mean(axis=1, keepdims=True).reshape((5, 5, 1)),
                features.std(axis=1, keepdims=True).reshape((5, 5, 1))
            ], axis=2)

            features = np.concatenate([features, features_], axis=2)  # (5, 5, 9)

            if features.max() > 1:
                continue

            attr_info = attr_cate[linkid]

            # linkid_map = link2ind.get(linkid, 0)  # 测试集中有83条数据为新的id
            links = topo.get(linkid, [0])
            links_len = 0 if (links == [0]) else len(links)

            saveDict = collections.OrderedDict()
            # id
            saveDict['linkid'] = Int64List([linkid])
            saveDict['current_slice_id'] = Int64List([current_slice_id])
            saveDict['future_slice_id'] = Int64List([future_slice_id])
            saveDict['weekday'] = Int64List([weekday])

            saveDict['direction'] = Int64List([attr_info[0]])
            saveDict['pathclass'] = Int64List([attr_info[1]])
            saveDict['speedclass'] = Int64List([attr_info[2]])
            saveDict['LaneNum'] = Int64List([attr_info[3]])
            saveDict['level'] = Int64List([attr_info[4]])
            # num
            saveDict['num_feats'] = FloatList([workday, future_slice, current_slice, slice_diff] + attr_num[linkid])
            saveDict['features'] = FloatList(features.reshape(-1))
            # seq
            saveDict['links'] = Int64List(links)
            saveDict['links_len'] = Int64List([links_len])
            # label
            saveDict['label'] = Int64List([label - 1])

            tf_example = tf.train.Example(features=tf.train.Features(feature=saveDict))
            writer.write(tf_example.SerializeToString())


if __name__ == '__main__':

    args = parse_command_params()

    attr = pd.read_csv('./data/row_data/attr.txt', sep='\t', names=['linkid', 'length', 'direction', 'pathclass', 'speedclass', 'LaneNum', 'speedlimit', 'level', 'width'])

    num_cols = ['length', 'speedlimit', 'width']
    cate_cols = ['direction', 'pathclass', 'speedclass', 'LaneNum', 'level']
    attr[num_cols] = (attr[num_cols] - attr[num_cols].min()) / (attr[num_cols].max() - attr[num_cols].min())

    attr_num = dict(zip(attr.linkid.tolist(), attr[num_cols].values.tolist()))
    attr_cate = dict(zip(attr.linkid.tolist(), attr[cate_cols].values.tolist()))

    topo = pd.read_csv('./data/row_data/topo.txt', sep='\t', names=['linkid', 'links'])
    topo['links'] = topo['links'].map(lambda x: [int(i) for i in x.split(",")])
    topo = dict(topo.values)

    # test_features: 0 - 181.0
    # slice_id:  -30 ~ 719
    # slice_id in feature -34 ~ 723
    savePath = os.path.join(args['user_path'], 'traindata')
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    train_files = sorted(os.listdir("/data/row_data/traffic"))[-13:]
    for file in train_files:
        save2TFrecord("/data/row_data/traffic", savePath, file)
    # save2TFrecord("./data", "20190801_testdata.txt")

