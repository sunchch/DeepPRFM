'''
PRFM
'''
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from time import time
import argparse
import LoadData_PRFM as DATA
import logging

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run PRFM.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--topk', nargs='?', default=5,
                        help='Topk recommendation list')
    parser.add_argument('--dataset', nargs='?', default='frappe',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of epochs.')
    parser.add_argument('--PRFMpretrain', type=int, default=0,
                        help='flag for PRFMpretrain. 1: initialize from PRFMpretrain; 0: randomly initialize; -1: save the model to PRFMpretrain file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default=0.01,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=10,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--neg', type=int, default=1,
                        help='number of negative samples in which to chose the largest score')

    return parser.parse_args()


class PRFM(BaseEstimator, TransformerMixin):
    def __init__(self, user_field_M, item_field_M, PRFMpretrain_flag, save_file, hidden_factor, epoch, batch_size, learning_rate,
                 lamda_bilinear, optimizer_type, verbose, random_seed=2020):
    
        self.user_field_M=user_field_M
        self.item_field_M=item_field_M
        self.PRFMpretrain_flag = PRFMpretrain_flag
        self.save_file = save_file
        self.hidden_factor = hidden_factor
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lamda_bilinear = lamda_bilinear
        self.optimizer_type = optimizer_type
        self.verbose = verbose
        self.random_seed = random_seed

        # init all variables in a tensorflow graph
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
            self.positive_features= tf.placeholder(tf.int32, shape=[None, None])
            self.negative_features= tf.placeholder(tf.int32, shape=[None, None])
            self.train_phase = tf.placeholder(tf.bool)

            # Variables.
            self.weights = self._initialize_weights()

            # Model.
            # _________ sum_square part for positive (u,i)_____________
            self.user_feature_embeddings = tf.nn.embedding_lookup(self.weights['user_feature_embeddings'], self.user_features)
            self.positive_feature_embeddings = tf.nn.embedding_lookup(self.weights['item_feature_embeddings'], self.positive_features)
            self.summed_user_emb = tf.reduce_sum(self.user_feature_embeddings, 1)
            self.summed_item_positive_emb = tf.reduce_sum(self.positive_feature_embeddings, 1)
            self.summed_positive_emb=tf.add(self.summed_user_emb,self.summed_item_positive_emb)
            self.summed_positive_emb_square = tf.square(self.summed_positive_emb)

            self.squared_user_emb=tf.square(self.user_feature_embeddings)
            self.squared_item_positiv_emb=tf.square(self.positive_feature_embeddings)
            self.squared_user_emb_sum=tf.reduce_sum(self.squared_user_emb, 1)
            self.squared_item_positive_emb_sum = tf.reduce_sum(self.squared_item_positiv_emb, 1)
            self.squared_positive_emb_sum=tf.add(self.squared_user_emb_sum,self.squared_item_positive_emb_sum)

            # ________ FM part for positive (u,i)__________
            self.FM_positive = 0.5 * tf.subtract(self.summed_positive_emb_square, self.squared_positive_emb_sum)
            # _________positive_________
            self.Bilinear_positive = tf.reduce_sum(self.FM_positive, 1, keepdims=True)  # None * 1
            self.user_feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['user_feature_bias'], self.user_features), 1)
            self.item_feature_bias_positive = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['item_feature_bias'], self.positive_features), 1)
            self.positive = tf.add_n([self.Bilinear_positive, self.user_feature_bias, self.item_feature_bias_positive])


            # _________ sum_square part for negative (u,j)_____________
            self.negative_feature_embeddings = tf.nn.embedding_lookup(self.weights['item_feature_embeddings'],self.negative_features)
            self.summed_item_negative_emb = tf.reduce_sum(self.negative_feature_embeddings, 1)
            self.summed_negative_emb = tf.add(self.summed_user_emb, self.summed_item_negative_emb)
            self.summed_negative_emb_square = tf.square(self.summed_negative_emb)

            self.squared_item_negative_emb = tf.square(self.negative_feature_embeddings)
            self.squared_item_negative_emb_sum = tf.reduce_sum(self.squared_item_negative_emb, 1)
            self.squared_negative_emb_sum = tf.add(self.squared_user_emb_sum, self.squared_item_negative_emb_sum)

            # ________ FM part for negative (u,j)__________
            self.FM_negative = 0.5 * tf.subtract(self.summed_negative_emb_square, self.squared_negative_emb_sum)
            # _________negative_________
            self.Bilinear_negative = tf.reduce_sum(self.FM_negative, 1, keepdims=True)
            self.item_feature_bias_negative = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['item_feature_bias'], self.negative_features), 1)
            self.negative = tf.add_n([self.Bilinear_negative, self.user_feature_bias, self.item_feature_bias_negative])

            # Compute the loss.
            self.loss=-tf.log(tf.sigmoid(self.positive-self.negative))
            self.loss=tf.reduce_sum(self.loss)+self.lamda_bilinear*(
                    tf.nn.l2_loss(self.user_feature_embeddings)
                    + tf.nn.l2_loss(self.positive_feature_embeddings)
                    + tf.nn.l2_loss(self.negative_feature_embeddings))
            
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
            self.config = tf.ConfigProto()
            init = tf.global_variables_initializer()
            # self.config.gpu_options.per_process_gpu_memory_fraction = 0.1
            self.config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=self.config)
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()  # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print ("#params: %d" % total_parameters)

    def _initialize_weights(self):
        all_weights = dict()
        if self.PRFMpretrain_flag > 0:
            weight_saver = tf.train.import_meta_graph(self.save_file + '.meta')
            PRFMpretrain_graph = tf.get_default_graph()
            user_feature_embeddings = PRFMpretrain_graph.get_tensor_by_name('user_feature_embeddings:0')
            item_feature_embeddings = PRFMpretrain_graph.get_tensor_by_name('item_feature_embeddings:0')
            user_feature_bias = PRFMpretrain_graph.get_tensor_by_name('user_feature_bias:0')
            item_feature_bias = PRFMpretrain_graph.get_tensor_by_name('item_feature_bias:0')
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
                tf.random_normal([self.user_field_M, self.hidden_factor], 0.0, 0.1),
                name='user_feature_embeddings')  # user_field_M * K
            all_weights['item_feature_embeddings'] = tf.Variable(
                tf.random_normal([self.item_field_M, self.hidden_factor], 0.0, 0.1),
                name='item_feature_embeddings')  # item_field_M * K
            all_weights['user_feature_bias'] = tf.Variable(
                tf.random_uniform([self.user_field_M, 1], 0.0, 0.1), name='user_feature_bias')  # user_field_M * 1
            all_weights['item_feature_bias'] = tf.Variable(
                tf.random_uniform([self.item_field_M, 1], 0.0, 0.1), name='item_feature_bias')  # item_field_M * 1
            #all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1
        return all_weights


    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.user_features: data['X_user'], self.positive_features: data['X_positive'],self.negative_features: data['X_negative'],self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss
   
    def get_random_block_from_data(self, train_data,batch_size):  # generate a random block of training data
        X_user, X_positive, X_negative = [], [], []
        all_items = data.binded_items.values()
        #get sample
        while len(X_user) < batch_size:
            index = np.random.randint(0, len(train_data['X_user']))
            X_user.append(train_data['X_user'][index])
            X_positive.append(train_data['X_item'][index])
            
            #uniform sampler
            user_features="-".join([str(item) for item in train_data['X_user'][index][0:]])
            user_id=data.binded_users[user_features] # get userID
            pos=data.user_positive_list[user_id]   #get positive list for the userID
            candidates = list(set(all_items) -set(pos))  #get negative set
            
            #negative sampler
            neg = np.random.choice(candidates,args.neg)
            item_samples=[]
            for neg_id in neg:
                negative_feature = data.item_map[neg_id].strip().split('-')  # get negative item feature
                item_samples.append([int(item) for item in negative_feature[0:]])
                for item_sample in item_samples:
                    X_negative.append(item_sample)   
        return {'X_user': X_user,'X_positive': X_positive,'X_negative': X_negative}

   
    def train(self, Train_data):
        for epoch in range(self.epoch):
            total_loss=0
            t1 = time()
            total_batch = int(len(Train_data['X_user']) / self.batch_size)
            for i in range(total_batch):
                # generate a batch
                batch_xs = self.get_random_block_from_data(Train_data,self.batch_size)
                # Fit training
                loss=self.partial_fit(batch_xs)
                total_loss = total_loss+loss
            t2 = time()
            print("the total loss in %d th iteration is: %.2f [%.1f s]" %(epoch, total_loss, t2-t1))
            f = open('result/PRFM/frappe/PRFM.txt', 'a', encoding='utf-8')
            f.write("the total loss in %d th iteration is: %.2f [%.1f s]\n" % (epoch, total_loss, t2 - t1))
            f.close()
            
            if self.verbose > 0 and epoch%self.verbose == 0:
                self.evaluate()       
            
        if self.PRFMpretrain_flag < 0:
            print ("Save model to file as PRFMpretrain.")
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
            logger.info("top:%d" % k)
            logger.info("the Hit Rate is: %f" % hit_rate)
            logger.info("the MRR is: %f" % mrr)
            logger.info("the NDCG is: %f" % ndcg)

            f = open('result/PRFM/frappe/PRFM.txt', 'a', encoding='utf-8')
            f.write("top:%d\n" % k)
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

        feed_dict = {self.user_features: X_user, self.positive_features: X_item,self.train_phase: False}
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
        print("FM: dataset=%s, factors=%d,  #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e,optimizer=%s"
                % (args.dataset, args.hidden_factor, args.epoch, args.batch_size, args.lr, args.lamda, args.optimizer))
    save_file = '../PRFMpretrain/%s_%d' % (args.dataset, args.hidden_factor)
    # Training
    t1 = time()
    model = PRFM(data.user_field_M, data.item_field_M, args.PRFMpretrain, save_file, args.hidden_factor, args.epoch,
               args.batch_size, args.lr, args.lamda, args.optimizer, args.verbose)
    model.train(data.Train_data)
    model.evaluate()

