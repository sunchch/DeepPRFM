'''
DeepPRFM
'''
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from time import time
import argparse
import LoadData_PRFM as DATA
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import logging


def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepPRFM.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--topk', nargs='?', default=7,
                        help='Topk recommendation list')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--user_feature_length', type=int, default=4,
                        help='user feature length (eg. frappe=8, lastfm=2, ml=4)')
    parser.add_argument('--item_feature_length', type=int, default=2,
                        help='item feature length (eg. frappe=2, lastfm=2, ml=2)')
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default=0.001,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--keep_prob', nargs='?', default='[0.8]',
                        help='Keep probability (1-dropout_ratio) for the Bi-Interaction layer. 1: no dropout')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=10,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=0,
                        help='Whether to perform batch normalization (0 or 1)')
    parser.add_argument('--neg', type=int, default=1,
                        help='number of negative samples in which to chose the largest score')
    parser.add_argument('--layers', nargs='?', default='[64]',
                        help="Size of each layer.")
    parser.add_argument('--dnn_layers', nargs='?', default='[128, 64]',
                        help="Size of each layer.")
    return parser.parse_args()


class DeepPRFM(BaseEstimator, TransformerMixin):
    def __init__(self, user_field_M, item_field_M, user_feature_length, item_feature_length, pretrain_flag, save_file, hidden_factor, epoch, batch_size, learning_rate,
                 lamda_bilinear, keep, optimizer_type, batch_norm, verbose, layers, dnn_layers, neg, random_seed=2020):
        self.user_field_M=user_field_M
        self.item_field_M=item_field_M
        self.user_feature_length = user_feature_length
        self.item_feature_length = item_feature_length
        self.pretrain_flag = pretrain_flag
        self.save_file = save_file
        self.hidden_factor = hidden_factor
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lamda_bilinear = lamda_bilinear
        self.keep = keep
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.verbose = verbose
        self.layers = layers
        self.dnn_layers = dnn_layers
        self.neg = neg
        self.no_dropout = np.array([1 for i in range(len(keep))])
        self.random_seed = random_seed
        self._init_graph()
     
    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            self.user_features = tf.placeholder(tf.int32, shape=[None, self.user_feature_length])
            self.positive_features= tf.placeholder(tf.int32, shape=[None, self.item_feature_length])
            self.negative_features= tf.placeholder(tf.int32, shape=[None, self.neg, self.item_feature_length])
            self.dropout_keep = tf.placeholder(tf.float32)
            self.train_phase = tf.placeholder(tf.bool)

            self.weights = self._initialize_weights()

            # Build Model
            self.loss = 0
            # --------- sum_square part for positive (u,i) ---------
            self.user_feature_embeddings = tf.nn.embedding_lookup(self.weights['user_feature_embeddings'], self.user_features)
            self.positive_feature_embeddings = tf.nn.embedding_lookup(self.weights['item_feature_embeddings'], self.positive_features)
            self.summed_user_emb = tf.reduce_sum(self.user_feature_embeddings, 1)
            self.summed_item_positive_emb = tf.reduce_sum(self.positive_feature_embeddings, 1)
            self.summed_positive_emb=tf.add(self.summed_user_emb,self.summed_item_positive_emb)
            self.summed_positive_emb_square = tf.square(self.summed_positive_emb)

            self.squared_user_emb=tf.square(self.user_feature_embeddings)
            self.squared_item_positive_emb=tf.square(self.positive_feature_embeddings)
            self.squared_user_emb_sum=tf.reduce_sum(self.squared_user_emb, 1)
            self.squared_item_positive_emb_sum = tf.reduce_sum(self.squared_item_positive_emb, 1)
            self.squared_positive_emb_sum=tf.add(self.squared_user_emb_sum,self.squared_item_positive_emb_sum)

            # --------- FM part for positive (u,i) ---------
            self.FM_positive = 0.5 * tf.subtract(self.summed_positive_emb_square, self.squared_positive_emb_sum)  # (None, k)

            # --------- MLP part for FM_positive ---------
            for i in range(0, len(self.layers)):
                self.FM_positive = tf.add(tf.matmul(self.FM_positive, self.weights['layer_%d' % i]),self.weights['bias_%d' % i])
                if self.batch_norm:
                    self.FM_positive = self.batch_norm_layer(self.FM_positive, train_phase=self.train_phase, scope_bn='bn_%d' % i)
                self.FM_positive = tf.nn.relu(self.FM_positive)
                self.FM_positive = tf.nn.dropout(self.FM_positive, self.dropout_keep[i])

            # --------- DNN part for original user features and positive item features ---------
            self.dnn_positive_input = tf.concat([self.user_feature_embeddings, self.positive_feature_embeddings], axis=1)
            batch_size, feature_size, embedding_size = self.dnn_positive_input.shape
            self.dnn_positive_output = tf.reshape(self.dnn_positive_input, (-1, feature_size * embedding_size))
            for i in range(0, len(self.dnn_layers)):
                self.dnn_positive_output = tf.add(tf.matmul(self.dnn_positive_output, self.weights['dnn_layer_%d' % i]), self.weights['dnn_bias_%d' % i])
                if self.batch_norm:
                    self.dnn_positive_output = self.batch_norm_layer(self.dnn_positive_output, train_phase=self.train_phase, scope_bn='dnn_bn_%d' % i)
                self.dnn_positive_output = tf.nn.relu(self.dnn_positive_output)
                self.dnn_positive_output = tf.nn.dropout(self.dnn_positive_output, self.dropout_keep[0])

            # --------- concat part for positive MLP and DNN ---------
            self.FM_positive = tf.concat([self.FM_positive, self.dnn_positive_output], axis=1)

            # --------- prediction part for positive (u,i) ---------
            self.FM_positive = tf.matmul(self.FM_positive, self.weights['prediction'])

            self.Bilinear_positive = tf.reduce_sum(self.FM_positive, 1, keepdims=True)
            self.user_feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['user_feature_bias'], self.user_features), 1)
            self.item_feature_bias_positive = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['item_feature_bias'], self.positive_features), 1)
            self.positive = tf.add_n([self.Bilinear_positive, self.user_feature_bias, self.item_feature_bias_positive])  # (None, 1)

            # --------- multi negNum for model learning ---------
            for neg_epoch in range(self.neg):
                # --------- sum_square part for negative (u,i) ---------
                self.negative_feature_embeddings = tf.nn.embedding_lookup(self.weights['item_feature_embeddings'], self.negative_features[:,neg_epoch,:])
                self.summed_item_negative_emb = tf.reduce_sum(self.negative_feature_embeddings, 1)
                self.summed_negative_emb = tf.add(self.summed_user_emb, self.summed_item_negative_emb)
                self.summed_negative_emb_square = tf.square(self.summed_negative_emb)

                self.squared_item_negative_emb = tf.square(self.negative_feature_embeddings)
                self.squared_item_negative_emb_sum = tf.reduce_sum(self.squared_item_negative_emb, 1)
                self.squared_negative_emb_sum = tf.add(self.squared_user_emb_sum, self.squared_item_negative_emb_sum)

                # --------- FM part for negative (u,i) ---------
                self.FM_negative = 0.5 * tf.subtract(self.summed_negative_emb_square, self.squared_negative_emb_sum)

                # --------- MLP part for FM_negative ---------
                for i in range(0, len(self.layers)):
                    self.FM_negative = tf.add(tf.matmul(self.FM_negative, self.weights['layer_%d' % i]),self.weights['bias_%d' % i])
                    if self.batch_norm:
                        self.FM_negative= self.batch_norm_layer(self.FM_negative,train_phase=self.train_phase,scope_bn='bn_%d' % i)
                    self.FM_negative = tf.nn.relu(self.FM_negative)
                    self.FM_negative = tf.nn.dropout(self.FM_negative,self.dropout_keep[i])

                # --------- DNN part for original user features and negative item features ---------
                self.dnn_negative_input = tf.concat([self.user_feature_embeddings, self.negative_feature_embeddings], axis=1)
                batch_size, feature_size, embedding_size = self.dnn_negative_input.shape
                self.dnn_negative_output = tf.reshape(self.dnn_negative_input, (-1, feature_size * embedding_size))
                for i in range(0, len(self.dnn_layers)):
                    self.dnn_negative_output = tf.add(tf.matmul(self.dnn_negative_output, self.weights['dnn_layer_%d' % i]), self.weights['dnn_bias_%d' % i])
                    if self.batch_norm:
                        self.dnn_negative_output = self.batch_norm_layer(self.dnn_negative_output, train_phase=self.train_phase, scope_bn='dnn_bn_%d' % i)
                    self.dnn_negative_output = tf.nn.relu(self.dnn_negative_output)
                    self.dnn_negative_output = tf.nn.dropout(self.dnn_negative_output, self.dropout_keep[0])

                # --------- concat part for negative MLP and DNN ---------
                self.FM_negative = tf.concat([self.FM_negative, self.dnn_negative_output], axis=1)

                # --------- prediction part for negative (u,i) ---------
                self.FM_negative = tf.matmul(self.FM_negative, self.weights['prediction'])  # (None, 1)

                self.Bilinear_negative = tf.reduce_sum(self.FM_negative, 1, keepdims=True)
                self.item_feature_bias_negative = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['item_feature_bias'], self.negative_features[:,neg_epoch,:]), 1)
                self.negative = tf.add_n([self.Bilinear_negative, self.user_feature_bias, self.item_feature_bias_negative])

                # --------- compute the loss ---------
                self.loss += tf.reduce_sum(-tf.log(tf.sigmoid(self.positive - self.negative)))

            self.loss += self.lamda_bilinear * (tf.nn.l2_loss(self.user_feature_embeddings)
                        + tf.nn.l2_loss(self.positive_feature_embeddings)
                        + tf.nn.l2_loss(self.negative_feature_embeddings))
            
            for i in range(0, len(self.layers)):
                layer_loss = self.lamda_bilinear*(tf.nn.l2_loss(self.weights['layer_%d' % i]))
            for i in range(0, len(self.dnn_layers)):
                dnn_layer_loss = self.lamda_bilinear*(tf.nn.l2_loss(self.weights['dnn_layer_%d' % i]))
            self.loss = self.loss + layer_loss + dnn_layer_loss

            # --------- choose the optimizer ---------
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)

            # --------- init ---------
            self.saver = tf.train.Saver()
            self.config = tf.ConfigProto()
            init = tf.global_variables_initializer()
            # self.config.gpu_options.per_process_gpu_memory_fraction = 0.1
            self.config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=self.config)
            self.sess.run(init)

            # --------- number of params ---------
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print ("#params: %d" % total_parameters)

    def _initialize_weights(self):
        all_weights = dict()
        if self.pretrain_flag > 0:
            weight_saver = tf.train.import_meta_graph(self.save_file + '.meta')
            pretrain_graph = tf.get_default_graph()
            user_feature_embeddings = pretrain_graph.get_tensor_by_name('user_feature_embeddings:0')
            item_feature_embeddings = pretrain_graph.get_tensor_by_name('item_feature_embeddings:0')
            user_feature_bias = pretrain_graph.get_tensor_by_name('user_feature_bias:0')
            item_feature_bias = pretrain_graph.get_tensor_by_name('item_feature_bias:0')
            with tf.Session() as sess:
                weight_saver.restore(sess, self.save_file)
                ue, ie, ub,ib  = sess.run([user_feature_embeddings, item_feature_embeddings, user_feature_bias, item_feature_bias])
            all_weights['user_feature_embeddings'] = tf.Variable(ue, dtype=tf.float32)
            all_weights['item_feature_embeddings'] = tf.Variable(ie, dtype=tf.float32)
            all_weights['user_feature_bias'] = tf.Variable(ub, dtype=tf.float32)
            all_weights['item_feature_bias'] = tf.Variable(ib, dtype=tf.float32)
        else:
            all_weights['user_feature_embeddings'] = tf.Variable(
                tf.random_normal([self.user_field_M, self.hidden_factor], 0.0, 0.1),
                name='user_feature_embeddings')
            all_weights['item_feature_embeddings'] = tf.Variable(
                tf.random_normal([self.item_field_M, self.hidden_factor], 0.0, 0.1),
                name='item_feature_embeddings')
            all_weights['user_feature_bias'] = tf.Variable(
                tf.random_uniform([self.user_field_M, 1], 0.0, 0.1), name='user_feature_bias')
            all_weights['item_feature_bias'] = tf.Variable(
                tf.random_uniform([self.item_field_M, 1], 0.0, 0.1), name='item_feature_bias')

        # MLP layer
        num_layer = len(self.layers)
        if num_layer > 0:
            glorot = np.sqrt(2.0 / (self.hidden_factor + self.layers[0]))
            all_weights['layer_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.hidden_factor , self.layers[0])), dtype=np.float32)
            all_weights['bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.layers[0])),
                                                dtype=np.float32)
            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.layers[i - 1] + self.layers[i]))
                all_weights['layer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.layers[i - 1], self.layers[i])),
                    dtype=np.float32)
                all_weights['bias_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.layers[i])), dtype=np.float32)
            # prediction layer
            glorot = np.sqrt(2.0 / (self.layers[-1] + self.dnn_layers[-1] + 1))
            all_weights['prediction'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.layers[-1] + self.dnn_layers[-1], 1)),
                                                    dtype=np.float32)
        else:
            all_weights['prediction'] = tf.Variable(
                np.ones((self.hidden_factor, 1), dtype=np.float32))

        # DNN layer
        if len(self.dnn_layers) > 0:
            glorot = np.sqrt(2.0 / (self.hidden_factor + self.dnn_layers[0]))
            all_weights['dnn_layer_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=((self.user_feature_length + self.item_feature_length) * self.hidden_factor, self.dnn_layers[0])), dtype=np.float32)
            all_weights['dnn_bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.dnn_layers[0])), dtype=np.float32)
            for i in range(1, len(self.dnn_layers)):
                glorot = np.sqrt(2.0 / (self.dnn_layers[i - 1] + self.dnn_layers[i]))
                all_weights['dnn_layer_%d' % i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.dnn_layers[i - 1], self.dnn_layers[i])), dtype=np.float32)
                all_weights['dnn_bias_%d' % i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.dnn_layers[i])), dtype=np.float32)

        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        # Note: the decay parameter is tunable
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):
        feed_dict = {self.user_features: data['X_user'], self.positive_features: data['X_positive'],
                     self.negative_features: data['X_negative'], self.dropout_keep: self.keep,
                     self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data_original(self, train_data, batch_size):
        X_user, X_positive, X_negative = [], [], []
        all_items = data.binded_items.values()
        while len(X_user) < batch_size:
            index = np.random.randint(0, len(train_data['X_user']))
            X_user.append(train_data['X_user'][index])
            X_positive.append(train_data['X_item'][index])

            user_features = "-".join([str(item) for item in train_data['X_user'][index][0:]])
            user_id = data.binded_users[user_features]
            pos = data.user_positive_list[user_id]
            candidates = list(set(all_items) - set(pos))

            neg = np.random.choice(candidates, args.neg)
            item_samples = []
            for neg_id in neg:
                negative_feature = data.item_map[neg_id].strip().split('-')
                item_samples.append([int(item) for item in negative_feature[0:]])
            X_negative.append(item_samples)
        return {'X_user': X_user, 'X_positive': X_positive, 'X_negative': X_negative}

    
    def train(self, Train_data):
        for epoch in range(self.epoch):
            total_loss=0
            t1 = time()
            total_batch = int(len(Train_data['X_user']) / self.batch_size)
            for i in range(total_batch):
                batch_xs = self.get_random_block_from_data_original(Train_data, self.batch_size)
                loss=self.partial_fit(batch_xs)
                total_loss = total_loss+loss
            t2 = time()
            print("the total loss in %d th iteration is: %.2f [%.1f s]" %(epoch, total_loss, t2-t1))

            f = open('result/DeepPRFM/ml/DeepPRFM.txt', 'a', encoding='utf-8')
            f.write("the total loss in %d th iteration is: %.2f [%.1f s]\n" % (epoch, total_loss, t2 - t1))
            f.close()

            if self.verbose > 0 and epoch%self.verbose == 0:
                model.evaluate()
            
        if self.pretrain_flag < 0:
            print ("Save model to file as pretrain.")
            self.saver.save(self.sess, self.save_file)

    def evaluate(self):
        self.graph.finalize()
        count = [0, 0, 0]
        rank = [[], [], []]
        topK = [5, 10, 20]
        for index in range(len(data.Test_data['X_user'])):
            user_features = data.Test_data['X_user'][index]
            item_features = data.Test_data['X_item'][index]
            scores = model.get_scores_per_user(user_features)
            # get true item score
            true_item_id = data.binded_items["-".join([str(item) for item in item_features[0:]])]
            true_item_score = scores[true_item_id]
            # delete visited scores
            user_id = data.binded_users["-".join([str(item) for item in user_features[0:]])]
            # logger.info(user_id)
            visited = data.user_positive_list[user_id]
            scores = np.delete(scores, visited)
            # whether hit
            sorted_scores = sorted(scores, reverse=True)

            label = []
            for i in range(len(topK)):
                label.append(sorted_scores[topK[i] - 1])
                if true_item_score >= label[i]:
                    count[i] = count[i] + 1
                    rank[i].append(sorted_scores.index(true_item_score) + 1)

        for i in range(len(topK)):
            mrr = 0
            ndcg = 0
            hit_rate = float(count[i]) / len(data.Test_data['X_user'])
            for item in rank[i]:
                mrr = mrr + float(1.0) / item
                ndcg = ndcg + float(1.0) / np.log2(item + 1)
            mrr = mrr / len(data.Test_data['X_user'])
            ndcg = ndcg / len(data.Test_data['X_user'])
            k = topK[i]
            logger.info("top:%d" % k)
            logger.info("the Hit Rate is: %f" % hit_rate)
            logger.info("the MRR is: %f" % mrr)
            logger.info("the NDCG is: %f" % ndcg)

            f = open('result/DeepPRFM/ml/DeepPRFM.txt', 'a', encoding='utf-8')
            f.write("top:%d\n" % k)
            f.write("the Hit Rate is: %f\n" % hit_rate)
            f.write("the MRR is: %f\n" % mrr)
            f.write("the NDCG is: %f\n" % ndcg)
            f.close()

    def get_scores_per_user(self, user_feature):
        X_user, X_item= [],[]
        all_items = data.binded_items.values()
        for itemID in range(len(all_items)):
            X_user.append(user_feature)
            item_feature=[int(feature) for feature in data.item_map[itemID].strip().split('-')[0:]]
            X_item.append(item_feature)
        feed_dict = {self.user_features: X_user, self.positive_features: X_item, self.train_phase: False, self.dropout_keep: self.no_dropout}
        scores=self.sess.run((self.positive),feed_dict=feed_dict)
        scores=scores.reshape(len(all_items))
        return scores

if __name__ == '__main__':
    
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('fm.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Data loading
    args = parse_args()
    data = DATA.LoadData(args.path, args.dataset)
    if args.verbose > 0:
        print( "DeepPRFM: dataset=%s, factors=%d, layers=%s, dnn_layers=%s, #epoch=%d, batch=%d, topK=%d, lr=%.4f, lambda=%.1e, optimizer=%s, batch_norm=%d, keep=%s"
                % (args.dataset, args.hidden_factor, args.layers, args.dnn_layers, args.epoch, args.batch_size, args.topk, args.lr, args.lamda, args.optimizer, args.batch_norm, args.keep_prob))

    save_file = 'DeepPRFM_pretrain/%s_%d_%d' % (args.dataset, args.hidden_factor, args.topk)
    # Training
    t1 = time()
    model = DeepPRFM(data.user_field_M, data.item_field_M, args.user_feature_length, args.item_feature_length, args.pretrain, save_file, args.hidden_factor, args.epoch,
               args.batch_size, args.lr, args.lamda, eval(args.keep_prob), args.optimizer, args.batch_norm, args.verbose,
               eval(args.layers), eval(args.dnn_layers), args.neg)
  
    model.train(data.Train_data)
    model.evaluate()
