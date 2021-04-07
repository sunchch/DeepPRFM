import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from time import time
import argparse
import LoadData_FM as DATA
import logging

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run FM.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--topk', nargs='?', default=7,
                        help='Topk recommendation list')
    parser.add_argument('--dataset', nargs='?', default='frappe',
                        help='Choose a dataset.')
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
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    return parser.parse_args()


class FM(BaseEstimator, TransformerMixin):
    def __init__(self, user_field_M, item_field_M, pretrain_flag, save_file, hidden_factor, epoch, batch_size, learning_rate,
                 lamda_bilinear, optimizer_type, verbose, random_seed=2020):
    
        self.user_field_M=user_field_M
        self.item_field_M=item_field_M
        self.pretrain_flag = pretrain_flag
        self.save_file = save_file
        self.hidden_factor = hidden_factor
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lamda_bilinear = lamda_bilinear
        self.optimizer_type = optimizer_type
        self.verbose = verbose
        self.random_seed = random_seed
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.user_features = tf.placeholder(tf.int32, shape=[None, None])
            self.item_features= tf.placeholder(tf.int32, shape=[None, None])
            self.y = tf.placeholder(tf.float32, shape=[None, 1])
            self.train_phase = tf.placeholder(tf.bool)

            # Variables.
            self.weights = self._initialize_weights()

            # Model.
            # _________ sum_square part _____________
            self.user_feature_embeddings = tf.nn.embedding_lookup(self.weights['user_feature_embeddings'], self.user_features)
            self.item_feature_embeddings = tf.nn.embedding_lookup(self.weights['item_feature_embeddings'], self.item_features)
            self.summed_user_emb = tf.reduce_sum(self.user_feature_embeddings, 1)
            self.summed_item_emb = tf.reduce_sum(self.item_feature_embeddings, 1)
            self.summed_emb=tf.add(self.summed_user_emb,self.summed_item_emb)
            self.summed_emb_square = tf.square(self.summed_emb)

            self.squared_user_emb=tf.square(self.user_feature_embeddings)
            self.squared_item_emb=tf.square(self.item_feature_embeddings)
            self.squared_user_emb_sum=tf.reduce_sum(self.squared_user_emb, 1)
            self.squared_item_emb_sum = tf.reduce_sum(self.squared_item_emb, 1)
            self.squared_emb_sum=tf.add(self.squared_user_emb_sum,self.squared_item_emb_sum)

            # ________ FM part for positive (u,i)__________
            self.FM= 0.5 * tf.subtract(self.summed_emb_square, self.squared_emb_sum)
            # _________positive_________
            self.Bilinear = tf.reduce_sum(self.FM, 1, keepdims=True)
            self.user_feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['user_feature_bias'], self.user_features), 1)
            self.item_feature_bias= tf.reduce_sum(tf.nn.embedding_lookup(self.weights['item_feature_bias'], self.item_features), 1)
            self.out = tf.add_n([self.Bilinear, self.user_feature_bias, self.item_feature_bias])  # None * 1
            
            self.loss = tf.reduce_sum(tf.square(self.y-self.out))
            self.loss = self.loss+self.lamda_bilinear*(tf.nn.l2_loss(self.user_feature_embeddings)+tf.nn.l2_loss(self.item_feature_embeddings))

            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.2
            self.sess = tf.Session(config=config)
            self.sess.run(init)

            # number of params
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
            #bias = pretrain_graph.get_tensor_by_name('bias:0')
            with tf.Session() as sess:
                weight_saver.restore(sess, self.save_file)
                ue, ie, ub,ib  = sess.run([user_feature_embeddings,item_feature_embeddings,user_feature_bias,item_feature_bias])
            all_weights['user_feature_embeddings'] = tf.Variable(ue, dtype=tf.float32)
            all_weights['item_feature_embeddings'] = tf.Variable(ie, dtype=tf.float32)
            all_weights['user_feature_bias'] = tf.Variable(ub, dtype=tf.float32)
            all_weights['item_feature_bias'] = tf.Variable(ib, dtype=tf.float32)
        else:
            all_weights['user_feature_embeddings'] = tf.Variable(
                tf.random_normal([self.user_field_M, self.hidden_factor], 0.0, 0.01),
                name='user_feature_embeddings')  # user_field_M * K
            all_weights['item_feature_embeddings'] = tf.Variable(
                tf.random_normal([self.item_field_M, self.hidden_factor], 0.0, 0.01),
                name='item_feature_embeddings')  # item_field_M * K
            all_weights['user_feature_bias'] = tf.Variable(
                tf.random_uniform([self.user_field_M, 1], 0.0, 0.01), name='user_feature_bias')  # user_field_M * 1
            all_weights['item_feature_bias'] = tf.Variable(
                tf.random_uniform([self.item_field_M, 1], 0.0, 0.01), name='item_feature_bias')  # item_field_M * 1
        return all_weights


    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.user_features: data['X_user'], self.item_features: data['X_item'], self.y: data['Y'], self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss
   
    def get_random_block_from_data(self, train_data,batch_size):  # generate a random block of training data
        start_index = np.random.randint(0, len(train_data['X_item']) - batch_size)
        X_user, X_item, Y = [], [], []
        # forward get sample
        i = start_index
        while len(X_user) < batch_size and i < len(train_data['X_user']):
            if len(train_data['X_user'][i]) == len(train_data['X_user'][start_index]):
                X_item.append(train_data['X_item'][i])
                X_user.append(train_data['X_user'][i])
                Y.append(train_data['Y'][i])
                i = i + 1
            else:
                break
        # backward get sample
        i = start_index
        while len(X_user) < batch_size and i >= 0:
            if len(train_data['X_user'][i]) == len(train_data['X_user'][start_index]):
                X_item.append(train_data['X_item'][i])
                X_user.append(train_data['X_user'][i])
                Y.append(train_data['Y'][i])
                i = i - 1
            else:
                break
        return {'X_user': X_user,'X_item': X_item, 'Y': Y}

    def shuffle_in_unison_scary(self, a, b,c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
    
    def train(self, Train_data):
        for epoch in range(self.epoch):
            total_loss=0
            t1 = time()
            self.shuffle_in_unison_scary(Train_data['X_user'], Train_data['X_item'], Train_data['Y'])
            total_batch = int(len(Train_data['X_user']) / self.batch_size)
            for i in range(total_batch):
                batch_xs = self.get_random_block_from_data(Train_data,self.batch_size)
                loss=self.partial_fit(batch_xs)
                total_loss= total_loss+loss
            t2 = time()
            print("the total loss in %d th iteration is: %f" %(epoch, total_loss))

            f = open('result/FM/FM.txt', 'a', encoding='utf-8')
            f.write("the total loss in %d th iteration is: %f\n" %(epoch, total_loss))
            f.close()
            
            if self.verbose > 0 and epoch%self.verbose == 0:
                self.evaluate()
            
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
            user_id = data.binded_users["-".join([str(item) for item in user_features[0:]])]  # get userID
            # logger.info(user_id)
            visited = data.user_positive_list[user_id]  # get positive list for the userID
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
            logger.info("top:%f" % k)
            logger.info("the Hit Rate is: %f" % hit_rate)
            logger.info("the MRR is: %f" % mrr)
            logger.info("the NDCG is: %f" % ndcg)

            f = open('result/FM/FM.txt', 'a', encoding='utf-8')
            f.write("top:%f\n" % k)
            f.write("the Hit Rate is: %f\n" % hit_rate)
            f.write("the MRR is: %f\n" % mrr)
            f.write("the NDCG is: %f\n" % ndcg)
            f.close()

    def get_scores_per_user(self, user_feature):  # evaluate the results for an user context, return scorelist
       
        X_user, X_item= [],[]
        all_items = data.binded_items.values()
        
        for itemID in range(len(all_items)):
            X_user.append(user_feature)
            item_feature=[int(feature) for feature in data.item_map[itemID].strip().split('-')[0:]]
            X_item.append(item_feature)

        feed_dict = {self.user_features: X_user, self.item_features: X_item, self.train_phase: False}
        scores=self.sess.run((self.out),feed_dict=feed_dict)
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
    args = parse_args()
    data = DATA.LoadData(args.path, args.dataset)
    
    if args.verbose > 0:
        print( "FM: dataset=%s, factors=%d,  #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e,optimizer=%s"
                % (args.dataset, args.hidden_factor, args.epoch, args.batch_size, args.lr, args.lamda, args.optimizer))

    save_file = '../pretrain/%s_%d' % (args.dataset, args.hidden_factor)
    # Training
    t1 = time()
    model = FM(data.user_field_M, data.item_field_M, args.pretrain, save_file, args.hidden_factor, args.epoch,
               args.batch_size, args.lr, args.lamda, args.optimizer, args.verbose)
    model.train(data.Train_data)
    model.evaluate()
