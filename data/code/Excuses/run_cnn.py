import time, os
import pickle
import pandas as pd
import tensorflow as tf
from argparse import ArgumentParser
from DeepCNN import DeepCNN
from config import Args
from numpy.random import seed

seed(41)
tf.set_random_seed(42)

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True

with open("/data/user_data/Excuses/linkids.pkl", "rb") as f:
    linkids = pickle.load(f)

def parser(record):
    read_dict = {
        'linkid': tf.FixedLenFeature([1], dtype=tf.int64),
        'links': tf.VarLenFeature(dtype=tf.int64),
        'links_len': tf.FixedLenFeature([1], dtype=tf.int64),

        'label': tf.FixedLenFeature([1], dtype=tf.int64),
        'current_slice_id': tf.FixedLenFeature([1], dtype=tf.int64),
        'future_slice_id': tf.FixedLenFeature([1], dtype=tf.int64),

        'weekday': tf.FixedLenFeature([1], dtype=tf.int64),
        'direction': tf.FixedLenFeature([1], dtype=tf.int64),
        'pathclass': tf.FixedLenFeature([1], dtype=tf.int64),
        'speedclass': tf.FixedLenFeature([1], dtype=tf.int64),
        'LaneNum': tf.FixedLenFeature([1], dtype=tf.int64),
        'level': tf.FixedLenFeature([1], dtype=tf.int64),

        'num_feats': tf.FixedLenFeature([7], dtype=tf.float32),
        'features': tf.VarLenFeature(dtype=tf.float32)
    }

    parse_example = tf.parse_single_example(record, read_dict)
    parse_example['features'] = tf.reshape(tf.sparse_tensor_to_dense(parse_example['features']), [5, 5, 9])

    return parse_example


def inputs(args, flag):
    filenames = [os.path.join(args.data_path, i) for i in sorted(os.listdir(args.data_path))[1:]]
    print(filenames)
    if flag == 'train':
        dataset = tf.data.TFRecordDataset(filenames[:-1])
        iterator = dataset.map(parser, num_parallel_calls=32).repeat(args.epoch).shuffle(buffer_size=5000).batch(
            args.batch_size).prefetch(buffer_size=1).make_one_shot_iterator()
    else:
        dataset = tf.data.TFRecordDataset(filenames[-1])
        iterator = dataset.map(parser, num_parallel_calls=32).batch(args.batch_size).prefetch(
            buffer_size=1).make_one_shot_iterator()

    batch_x = iterator.get_next()
    batch_x['links'] = tf.sparse_tensor_to_dense(batch_x['links'])

    return batch_x


def train(args):
    print("training.............")
    with tf.Session() as sess:
        batch_x = inputs(args, 'train')
        model = DeepCNN(args, batch_x, keep_prob=0.8, flag='train')
        sess.run(tf.global_variables_initializer())
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(args.save_path, sess.graph)
        total_loss, step = 0, 0
        start_time = time.time()
        while True:
            try:
                loss, summmary, step, _ = sess.run([model.loss, summary_op, model.global_step, model.train_op])
                train_writer.add_summary(summmary, step)
                total_loss += loss
                step += 1
                if step % 100 == 0 or step == 1:
                    print('Global_step:%d\tloss:%.4f' % (step, total_loss / step))
            except tf.errors.OutOfRangeError:
                print("End of dataset")
                break
        end_time = time.time()
        print("train model2 use time: %d sec" % (end_time - start_time))
        model.save(sess, args.save_path + '/model.ckpt', global_step=model.global_step.eval())
        if args.save_emb:
            emb = sess.run(model.embedding_link)
            data = pd.DataFrame(emb).reset_index().rename(columns={'index': 'linkid'})
            print(data.head())
            data.to_csv(os.path.join(args.save_path, "embedding_linkid.txt"), header=None, index=None, sep=",")


def test(args):
    args.fine_tune = False
    print("test................")
    tf.reset_default_graph()
    out_file = args.save_path + "/test_result.txt"
    with tf.Session() as sess:
        batch_x = inputs(args, 'test')
        model = DeepCNN(args, batch_x, keep_prob=1.0, flag='test')
        sess.run(tf.global_variables_initializer())
        start_time = time.time()
        model.load(sess, args.save_path)
        while True:
            try:
                output = sess.run(model.output)
                pd.DataFrame({'linkid': output['linkid'],
                              'current_slice_id': output['current_slice_id'],
                              'future_slice_id': output['future_slice_id'],
                              'label': output['label'],
                              'pred': output['pred'],
                              'pctr': output['pctr']}).to_csv(out_file, index=None, header=None, mode='a', sep="\t")
            except tf.errors.OutOfRangeError:
                print("End of dataset")
                break
        end_time = time.time()
        print("test use time: %d sec" % (end_time - start_time))


def pred(args):
    args.fine_tune = False
    print("pred................")
    tf.reset_default_graph()
    with tf.Session() as sess:
        batch_x = inputs(args, 'test')
        model = DeepCNN(args, batch_x, keep_prob=1.0, flag='test')
        sess.run(tf.global_variables_initializer())
        start_time = time.time()
        model.load(sess, args.save_path)
        result = {'link': [],
                  'current_slice_id': [],
                  'future_slice_id': [],
                  'label': []
                  }
        while True:
            try:
                output = sess.run(model.output)
                result['link'].extend(output['linkid'])
                result['current_slice_id'].extend(output['current_slice_id'])
                result['future_slice_id'].extend(output['future_slice_id'])
                result['label'].extend(output['label'])
            except tf.errors.OutOfRangeError:
                print("End of dataset")
                break
        end_time = time.time()
        print("test use time: %d sec" % (end_time - start_time))
        result = pd.DataFrame(result)
        # result['label'] = result['label'].map(lambda x: 3 if x > 3 else x)  评测会自动处理
        # result['link'] = result['link'].map(lambda x: ind2link.get(x, 0))  # linkids[0]
        result.to_csv(args.save_path + "/result.csv", index=None)

def parse_command_params():
    args = Args()

    ap = ArgumentParser()
    ap.add_argument('-user_path', default="/data/user_data/Excuses", type=str, help='user out path')
    ap.add_argument('-is_train', default=0, type=int, help='is train')
    ap.add_argument('-is_test', default=0, type=int, help='is test')
    ap.add_argument('-is_pred', default=0, type=int, help='is pred')
    ap.add_argument('-save_emb', default=0, type=int, help='is save emb')
    args_ = vars(ap.parse_args())

    args.vocab_size = len(linkids)
    args.data_path = os.path.join(args_['user_path'], 'traindata')
    args.save_path = os.path.join(args_['user_path'], 'output')
    args.emb_path = os.path.join(args_['user_path'], 'w2v/topo_emb.txt')
    args.is_train = args_['is_train']
    args.is_test = args_['is_test']
    args.is_pred = args_['is_pred']
    args.save_emb = args_['save_emb']

    return args

if __name__ == '__main__':

    args = parse_command_params()

    if args.is_train:
        train(args)

    if args.is_test:
        test(args)

    if args.is_pred:
        pred(args)





