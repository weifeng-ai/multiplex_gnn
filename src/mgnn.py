import numpy as np
import tensorflow as tf
import pandas as pd
import sys
import os
from tensorflow.contrib.layers import dense_to_sparse
from tensorflow import sparse_tensor_dense_matmul
from collections import defaultdict

from layer import GCN

MODEL_DIR = "../models/"
OUTPUT_DIR = "../results/"

class MGNN(object):
    
    def __init__(self, n_user, n_item, n_behavior, DATA_DIR, DATA_NAME):
        self.n_user = n_user
        self.n_item = n_item
        self.n_behavior = n_behavior
        self.DATA_DIR = DATA_DIR
        self.DATA_NAME = DATA_NAME

    def config_global(self, MODEL_CLASS, HIDDEN_DIM, LAMBDA, LEARNING_RATE, BATCH_SIZE):        
        self.HIDDEN_DIM = HIDDEN_DIM
        self.LAMBDA = LAMBDA
        self.LEARNING_RATE = LEARNING_RATE
        self.BATCH_SIZE = BATCH_SIZE
        self.MODEL_CLASS = MODEL_CLASS

        DATA_NAME = self.DATA_NAME
        self.MODEL_NAME = DATA_NAME+"."+MODEL_CLASS+".dim."+str(HIDDEN_DIM)+".lambda."+str(LAMBDA)+".lr."+str(LEARNING_RATE)

    def load_data(self, data_train, data_valid, data_test, user_history):
        self.data_train = data_train
        self.data_valid = data_valid
        self.data_test = data_test
        self.user_history = user_history

    def load_matrix(self, supra_adj, merged_adj):
        self.supra_adj = self._get_sparse_tensor(supra_adj)
        self.merged_adj = self._get_sparse_tensor(merged_adj)

    def _get_sparse_tensor(self, sparse_mat):
        indices = np.mat([sparse_mat.row, sparse_mat.col]).transpose()
        return  tf.cast(tf.SparseTensor(indices, sparse_mat.data, sparse_mat.shape),tf.float32)

    def next_training_batch(self, BATCH_SIZE, N_MAX=1000000):
        # training_tuple: [user_id, pos_item_id, behavior, neg_item_id * n_neg]
        n_training = min(self.data_train.shape[0], N_MAX)
        N_BATCH = n_training//BATCH_SIZE
        index_selected = np.random.permutation(self.data_train.shape[0])[:n_training]

        n_neg = self.data_train.shape[1] - 3
        for i in range(0, N_BATCH*BATCH_SIZE, BATCH_SIZE):
            current_index = index_selected[i:(i+BATCH_SIZE)]
            neg_index = np.random.choice(n_neg, size=current_index.shape[0])+3
            xu = self.data_train[current_index, 0]
            xi = self.data_train[current_index, 1]
            xb = self.data_train[current_index, 2]
            xj = self.data_train[current_index, neg_index]
            yield xu, xi, xb, xj
            
    def next_evaluation_batch(self, data, BATCH_SIZE):
        batch = []
        for i in range(0, data.shape[0], BATCH_SIZE):
            xu = data[i:(i+BATCH_SIZE), 0]
            xi = data[i:(i+BATCH_SIZE), 1]
            xb = data[i:(i+BATCH_SIZE), 2]
            yield xu, xi, xb

    def model_constructor(self):
        supra_adj = self.supra_adj
        merged_adj = self.merged_adj
        n_behavior = self.n_behavior
        n_user = self.n_user
        n_item = self.n_item
        HIDDEN_DIM = self.HIDDEN_DIM
        LAMBDA = self.LAMBDA
        LEARNING_RATE = self.LEARNING_RATE
        n_node = n_user + n_item

        u = tf.placeholder(tf.int32, [None]) # user
        i = tf.placeholder(tf.int32, [None]) # pos_item
        j = tf.placeholder(tf.int32, [None]) # neg_item
        b = tf.placeholder(tf.int32, [None]) # behavior
        dropout = tf.placeholder_with_default(0., shape=())
        
        with tf.variable_scope(self.MODEL_NAME+'_vars'):
            specific_features = tf.get_variable("specific_feature", [n_node*n_behavior, HIDDEN_DIM], 
                                         initializer=tf.random_uniform_initializer(-0.01, 0.01))
            shared_features = tf.get_variable("shared_feature", [n_node, HIDDEN_DIM], 
                                         initializer=tf.random_uniform_initializer(-0.01, 0.01))
            # weights of different behaviors
            behavior_weights = tf.get_variable("behavior_W", initializer=tf.constant([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]), trainable=False)

        shared_embeddings = GCN(adj=merged_adj, feature_dim=HIDDEN_DIM, hidden_dim=HIDDEN_DIM, name=self.MODEL_NAME+'gcn_1', dropout=dropout)(shared_features)
        specific_init = tf.concat([tf.tile(shared_embeddings, [n_behavior, 1]), specific_features], axis=1)
        specific_embeddings = GCN(adj=supra_adj, feature_dim=HIDDEN_DIM*2, hidden_dim=HIDDEN_DIM, name=self.MODEL_NAME+'gcn_2', dropout=dropout)(specific_init)        
        
        # embeddings: [n_node*n_behavior, emb_size]
        embeddings = tf.concat([specific_init, specific_embeddings], axis=1)

        u_emb = tf.nn.embedding_lookup(embeddings, u + n_node * b)
        i_emb = tf.nn.embedding_lookup(embeddings, i + n_node * b + n_user)
        j_emb = tf.nn.embedding_lookup(embeddings, j + n_node * b + n_user)

        x_pos = tf.reduce_sum(tf.multiply(u_emb, i_emb), 1, keep_dims=True)
        x_neg = tf.reduce_sum(tf.multiply(u_emb, j_emb), 1, keep_dims=True)

        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weight' in v.name])
        batch_weights = tf.nn.embedding_lookup(behavior_weights, b)  
        logloss = - tf.reduce_sum(tf.log_sigmoid(x_pos - x_neg) * batch_weights)
        logloss0 = LAMBDA*l2_loss + logloss
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(logloss0)  
        
        model={}
        model['u'] = u; model['i'] = i; model['j'] = j; model['b'] = b; model['embeddings'] = embeddings;
        model['dropout'] = dropout; model['loss'] = logloss; model['optimizer'] = optimizer
        return model
            
    def train(self, save_result=False):
        BATCH_SIZE = self.BATCH_SIZE
        EPOCHS = 5000
        MAX_NO_PROGRESS = 5
        topK = 20 if self.DATA_NAME != 'db_book' else 50

        print("start training "+self.MODEL_NAME+" ...")
        sys.stdout.flush()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True

        with tf.Session(config=config) as session:
            model = self.model_constructor()
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
                
            loss_train_min = 1e10
            ndcg_vali_max = 0
            n_noprogress = 0
            
            for epoch in range(1,EPOCHS):
                count = 0
                count_sample = 0
                loss_train = 0

                print("\n=== current batch: ", end="")
                for xu, xi, xb, xj in self.next_training_batch(BATCH_SIZE):
                    loss_train_batch, _= session.run([model['loss'], model['optimizer']], 
                                            feed_dict={model['u']:xu, model['i']:xi, model['j']:xj, model['b']:xb, model['dropout']:0.2})
                    loss_train += loss_train_batch
                    count += 1.0
                    count_sample += len(xu)
                    if count % 500 == 0:
                        print(int(count), end=", ")
                        sys.stdout.flush()
                print("complete!")

                loss_train /= count_sample
                if loss_train < loss_train_min:
                    loss_train_min = loss_train
                print("epoch: ", epoch, "  train_loss: {:.4f}, min_loss: {:.4f}".format(loss_train, loss_train_min))

                result = self.evaluation(session, model, self.data_valid, self.user_history, topK)
                ndcg_vali = 0

                print('behavior\tAUC\tNDCG\tHR')
                n_behavior = len(result)
                for behavior in range(n_behavior):
                    print(behavior, '\t', np.array(result[behavior]).mean(axis=0))

                for behavior in range(n_behavior):
                    ndcg_vali += np.array(result[behavior]).mean(axis=0)[1]
                ndcg_vali /= n_behavior
                
                if ndcg_vali >= ndcg_vali_max:
                    ndcg_vali_max = ndcg_vali
                    n_noprogress = 0
                    saver.save(session, MODEL_DIR + self.MODEL_NAME + ".model.ckpt")
                else:
                    n_noprogress += 1

                print("valid_sample_ndcg: {:.4f}, max_sample_ndcg: {:.4f}".format(ndcg_vali, ndcg_vali_max), 
                      "  #no progress: ", n_noprogress)
                sys.stdout.flush()
                if n_noprogress >= MAX_NO_PROGRESS:
                    break

                # saver.restore(session, MODEL_DIR + self.MODEL_NAME + ".model.ckpt")

                print('evaluating model ...')
                test_result = self.evaluation(session, model, self.data_test, self.user_history, topK)
                test_final_result = np.zeros((n_behavior, 3))
                print('test result:')
                print('behavior\tAUC\tNDCG\tHR')
                for behavior in range(len(test_result)):
                    test_final_result[behavior] = np.array(test_result[behavior]).mean(axis=0)
                    print(behavior, '\t', test_final_result[behavior])

                if save_result:
                    OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, self.MODEL_NAME+'.result.csv')
                    df_result = pd.DataFrame(np.concatenate([valid_final_result,test_final_result], axis=0),columns=['AUC', 'NDCG', 'HR'])
                    df_result['behavior'] = list(range(n_behavior)) * 2
                    df_result.to_csv(OUTPUT_FILE_PATH, index=False)

        print("done!")
        sys.stdout.flush()
             
    def evaluation(self, session, model, data_eval, user_history, topK):
        n_item = self.n_item
        n_user = self.n_user
        n_node = n_user + n_item
        result = defaultdict(list)

        embeddings = session.run(model['embeddings'], feed_dict={model['dropout']:0.})
        behavior_scores = []
        user_embs = []
        item_embs = []
        for behavior in range(self.n_behavior):
            user_embs.append(embeddings[behavior*n_node:behavior*n_node + n_user, ])
            item_embs.append(embeddings[behavior*n_node+n_user : (behavior+1)*n_node, :])

        for xu, xi, xb in self.next_evaluation_batch(data_eval, self.BATCH_SIZE):
            behavior_scores = []
            for behavior in range(self.n_behavior):
                score = np.dot(user_embs[behavior][xu], item_embs[behavior].T)
                behavior_scores.append(score)

            for i, (user, item, behavior) in enumerate(zip(xu, xi, xb)):
                scores = behavior_scores[behavior][i]
                history_items = user_history[user][behavior]
                mask = np.ones(n_item)
                mask[history_items] = 0
                n = np.sum(mask)
                mask_scores = scores * mask
                rank = np.sum(mask_scores > mask_scores[item])
                auc = (n - rank)/n
                ndcg = 1.0/np.log2(2+rank) if rank<topK else 0
                hr = int(rank<topK)
                result[behavior].append([auc, ndcg, hr])
        return result