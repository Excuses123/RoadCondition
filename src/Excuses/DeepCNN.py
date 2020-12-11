
import pandas as pd
import tensorflow as tf


class DeepCNN(object):

    def __init__(self, args, batch_x, keep_prob, flag):

        self.args = args
        self.linkid = batch_x['linkid'][:, 0]

        self.links = batch_x['links']
        self.links_len = batch_x['links_len'][:, 0]

        self.current_slice_id = batch_x['current_slice_id'][:, 0]
        self.future_slice_id = batch_x['future_slice_id'][:, 0]

        self.weekday = batch_x['weekday'][:, 0]  # 7
        self.direction = batch_x['direction'][:, 0]  # 3
        self.pathclass = batch_x['pathclass'][:, 0]  # 5
        self.speedclass = batch_x['speedclass'][:, 0]  # 7
        self.LaneNum = batch_x['LaneNum'][:, 0]  # 3
        self.level = batch_x['level'][:, 0]  # 5

        self.num_feats = batch_x['num_feats']  # (batch, 7)
        self.features = batch_x['features']  # [batch, 5, 5, 9]
        self.label = batch_x['label']
        self.keep_prob = keep_prob
        self.build_model()
        if flag == 'train':
            self.train()
        else:
            self.test()

    def __load_w2v(self, path, expectDim):
        print("loading embedding!")
        emb = pd.read_csv(path, header=None)
        emb = emb.iloc[:, 1:].values.astype("float32")
        assert emb.shape[1] == expectDim
        return emb

    def build_model(self):

        if self.args.fine_tune:
            self.embedding_link = tf.Variable(self.__load_w2v(self.args.emb_path, self.args.embedding_size),
                                              dtype=tf.float32, name='link_id_w')
        else:
            self.embedding_link = tf.Variable(
                tf.random_uniform([self.args.vocab_size, self.args.embedding_size], seed=42), name='link_id_w')

        self.embedding_slice = tf.Variable(tf.random_uniform([720, 20], seed=42), name='slice_id_w')
        self.embedding_weekday = tf.Variable(tf.random_uniform([7, 5], seed=42), name='weekday_w')

        self.embedding_direction = tf.Variable(tf.random_uniform([5, 4], seed=42), name='direction_w')
        self.embedding_pathclass = tf.Variable(tf.random_uniform([10, 5], seed=42), name='pathclass_w')
        self.embedding_speedclass = tf.Variable(tf.random_uniform([10, 6], seed=42), name='speedclass_w')
        self.embedding_LaneNum = tf.Variable(tf.random_uniform([5, 4], seed=42), name='LaneNum_w')
        self.embedding_level = tf.Variable(tf.random_uniform([10, 5], seed=42), name='level_w')

        links_emb = tf.nn.embedding_lookup(self.embedding_link, self.links)  # (batch, sl, 64)
        mask = tf.sequence_mask(self.links_len, tf.shape(links_emb)[1], dtype=tf.float32)  # shape(batch, sl)
        mask = tf.expand_dims(mask, -1)
        mask = tf.tile(mask, [1, 1, tf.shape(links_emb)[2]])
        links_emb *= mask
        hist = tf.reduce_sum(links_emb, 1)
        hist = tf.div(hist, tf.cast(tf.tile(tf.expand_dims(self.links_len, 1), [1, self.args.embedding_size]),
                                    tf.float32) + 10e-32)

        link_id_emb = tf.nn.embedding_lookup(self.embedding_link, self.linkid)  # (batch, 64)

        current_slice_emb = tf.nn.embedding_lookup(self.embedding_slice, self.current_slice_id)  # (batch, 32)
        future_slice_emb = tf.nn.embedding_lookup(self.embedding_slice, self.future_slice_id)  # (batch, 32)

        weekday_emb = tf.nn.embedding_lookup(self.embedding_weekday, self.weekday)  # (batch, 5)
        direction_emb = tf.nn.embedding_lookup(self.embedding_direction, self.direction)  # (batch, 4)
        pathclass_emb = tf.nn.embedding_lookup(self.embedding_pathclass, self.pathclass)  # (batch, 5)
        speedclass_emb = tf.nn.embedding_lookup(self.embedding_speedclass, self.speedclass)  # (batch, 6)
        LaneNum_emb = tf.nn.embedding_lookup(self.embedding_LaneNum, self.LaneNum)  # (batch, 4)
        level_emb = tf.nn.embedding_lookup(self.embedding_level, self.level)  # (batch, 5)

        num_emb = tf.layers.dense(self.num_feats, 7, activation=None, name='miss_value')

        pooledOutputs = []
        for i, filterSize in enumerate(self.args.filterSizes):
            with tf.name_scope("conv-maxpool-%s" % filterSize):
                filterShape = [filterSize, 5, 9, self.args.numFilters]  # (2, 5, 9, 64)  (3, 5, 9, 64)
                W = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1, seed=42), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.args.numFilters]), name="b")  # (64)
                conv = tf.nn.conv2d(self.features, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")  # (batch, len - filterSize + 1, 1, 128)
                pooled = tf.nn.max_pool(h, ksize=[1, self.args.seq_len - filterSize + 1, 1, 1], strides=[1, 1, 1, 1],
                                        padding='VALID', name="pool")
                pooledOutputs.append(pooled)

        numFiltersTotal = self.args.numFilters * len(self.args.filterSizes)  # 2 * 64
        self.hPoolFlat = tf.reshape(tf.concat(pooledOutputs, 3), [-1, numFiltersTotal])  # (batch, 2 * 64)

        self.hDrop = tf.nn.dropout(self.hPoolFlat, self.keep_prob)  # (batch, 2 * 64)

        embs = tf.concat(
            [self.hDrop, hist, link_id_emb, current_slice_emb, future_slice_emb, weekday_emb, direction_emb,
             pathclass_emb, speedclass_emb, LaneNum_emb, level_emb, num_emb], axis=1)

        inputs = tf.layers.batch_normalization(inputs=embs, name="b1")
        f1 = tf.layers.dense(inputs, 512, activation=tf.nn.relu, name='f1')
        f1 = tf.nn.dropout(f1, self.keep_prob)
        f2 = tf.layers.dense(f1, 256, activation=tf.nn.relu, name='f2')
        f2 = tf.nn.dropout(f2, self.keep_prob)
        self.logits = tf.layers.dense(f2, self.args.num_class, activation=None, name='logits')

        self.prob = tf.nn.softmax(self.logits)

        top_k = tf.nn.top_k(self.prob, k=1)
        self.pred = tf.squeeze(top_k.indices)
        self.pctr = tf.squeeze(top_k.values)

    def train(self):
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.squeeze(self.label)))

        tf.summary.scalar("loss", self.loss)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def test(self):
        self.output = {
            'linkid': tf.squeeze(self.linkid),
            'label': tf.squeeze(self.label) + 1,
            'current_slice_id': self.current_slice_id,
            'future_slice_id': self.future_slice_id,
            'pred': self.pred + 1,
            'pctr': self.pctr,
            'prob': self.prob
        }

    def save(self, sess, path, global_step):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path, global_step=global_step)

    def load(self, sess, path):
        """
        加载模型
        """
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, save_path=ckpt.model_checkpoint_path)
            print("Load models of step %s success" % step)
        else:
            print("No checkpoint!")













