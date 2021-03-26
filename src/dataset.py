import numpy as np
import pandas as pd
import sys
import os
import bisect
from scipy import sparse
from tqdm import tqdm
from collections import defaultdict


class Dataset(object):
    
    def __init__(self, DATA_DIR='../data', DATA_NAME='steam'):
        self.DATA_DIR = DATA_DIR
        self.DATA_NAME = DATA_NAME
        
        print("Initializing dataset:", DATA_NAME)
        sys.stdout.flush()
        
        DATA_PATH = os.path.join(DATA_DIR, DATA_NAME, "user_item_behavior.csv")
        n_user = 0
        n_item = 0
        n_behavior = 0
        n_interaction = 0
        n_interaction_each_behavior = np.zeros(6)

        data = defaultdict(lambda: defaultdict(set))
        try:
            with open(DATA_PATH) as fin:
                for line in fin:
                    user_id, item_id, behavior = map(int, line.strip().split(','))
                    if item_id not in data[user_id]:
                        n_interaction += 1
                    data[user_id][item_id].add(behavior)
                    n_user = max(user_id+1, n_user)
                    n_item = max(item_id+1, n_item)
                    n_behavior = max(behavior+1, n_behavior)
                    n_interaction_each_behavior[behavior] += 1
        except:
            print("Fail to load", DATA_PATH, ". Please check if the file exists and is in a correct format!")
            return
        
        self.data = data       # data indexed by (user_id -> item_id -> behavior_id)
        self.data_ubi = None   # data indexed by (user_id -> behavior_id -> item_id), which is used for negative sampling 
        self.n_user = n_user
        self.n_item = n_item
        self.n_behavior = n_behavior
        self.n_interaction = n_interaction
        
        print("Successfully initialized!")
        print(n_interaction, "interactions about", n_user, "users,", n_item, "items,", n_behavior, "behaviors are loaded!")
        print("#interactions of each behavior are as follows:")
        print(n_interaction_each_behavior[:n_behavior])
        sys.stdout.flush()
    
    
    def split_train_test(self, seed=None, dump_splits=False):
        data = self.data
        data_valid = []
        data_test = []
        
        print("Spliting data ...")
        sys.stdout.flush()
        np.random.seed(seed)

        for user_id in data:
            items = list(data[user_id].keys())
            if len(items) >= 3:
                valid_test_items = np.random.choice(items, size=2, replace=False)
                valid_item = valid_test_items[0]
                test_item = valid_test_items[1]

                for behavior in data[user_id][valid_item]:
                    data_valid.append([user_id, valid_item, behavior])
                data[user_id].pop(valid_item)

                for behavior in data[user_id][test_item]:
                    data_test.append([user_id, test_item, behavior])
                data[user_id].pop(test_item)

        self.data_valid = np.array(data_valid)
        self.data_test = np.array(data_test)

        print("Successfully splited data in to train/validation/test!")
        sys.stdout.flush()

        if dump_splits:
            print('Dumping data...')
            sys.stdout.flush()

            TRAIN_FILE_PATH = os.path.join(self.DATA_DIR, self.DATA_NAME, 'train.csv')
            with open(TRAIN_FILE_PATH, 'w') as fout:
                for user_id in data:
                    items = data[user_id]
                    for item_id in items:
                        for behavior in data[user_id][item_id]:
                            fout.write(str(user_id)+','+str(item_id)+','+str(behavior)+'\n')

            VALID_FILE_PATH = os.path.join(self.DATA_DIR, self.DATA_NAME, 'valid.csv')
            TEST_FILE_PATH = os.path.join(self.DATA_DIR, self.DATA_NAME, 'test.csv')

            np.savetxt(VALID_FILE_PATH, self.data_valid, fmt="%s", delimiter=",")
            np.savetxt(TEST_FILE_PATH,  self.data_test,  fmt="%s", delimiter=",")
            
    def load_split_data(self):
        print("loading train/valid/test data ...")
        sys.stdout.flush()

        TRAIN_FILE_PATH = os.path.join(self.DATA_DIR, self.DATA_NAME, 'train.csv')
        VALID_FILE_PATH = os.path.join(self.DATA_DIR, self.DATA_NAME, 'valid.csv')
        TEST_FILE_PATH = os.path.join(self.DATA_DIR, self.DATA_NAME, 'test.csv')

        data = defaultdict(lambda: defaultdict(set))
        with open(TRAIN_FILE_PATH) as fin:
            for line in fin:
                user_id, item_id, behavior = map(int, line.strip().split(','))
                data[user_id][item_id].add(behavior)

        self.data = data
        self.data_valid = np.loadtxt(VALID_FILE_PATH, dtype=int, delimiter=",")
        self.data_test = np.loadtxt(TEST_FILE_PATH, dtype=int, delimiter=",")

    def sampling_training(self, N_NEG=5, dump_samples=False):        
        if self.data_ubi is None:
            self.data_ubi = self.get_user_behavior_item_data()
        print("Start sampling training data ...")
        sys.stdout.flush()
        
        training_samples = []
        for user_id in tqdm(self.data_ubi):
            behaviors = self.data_ubi[user_id]
            for behavior in behaviors:
                items = self.data_ubi[user_id][behavior]
                n_pos = len(items)
                neg_samples = self.negative_sampling(items, self.n_item, N_NEG*n_pos)

                # training_tuple: [user_id, pos_item_id, behavior, neg_item_id * n_neg]
                training_tuple = np.concatenate([np.ones((n_pos, 1)) * user_id,
                                                 np.array(items).reshape(n_pos, 1),
                                                 np.ones((n_pos, 1)) * behavior,
                                                 neg_samples.reshape(n_pos, N_NEG)
                                                 ], axis = 1)                    
                training_samples += list(training_tuple)
        training_samples = np.array(training_samples, dtype=int)
        print("done!")
        sys.stdout.flush()
        if dump_samples:
            FILE_PATH = os.path.join(self.DATA_DIR, self.DATA_NAME, 'training_samples.csv')
            np.savetxt(FILE_PATH, training_samples, fmt="%s", delimiter=",")
            
        return training_samples

    def get_user_behavior_item_data(self):
        user_behavior_item_list = defaultdict(lambda: defaultdict(list))
        for user_id in self.data:
            for item_id, behaviors in self.data[user_id].items():
                for behavior in behaviors:
                    bisect.insort(user_behavior_item_list[user_id][behavior], item_id)
        return user_behavior_item_list

        
    def negative_sampling(self, pos_inds, n_items, n_samp=1):
        """ Pre-verified with binary search
        `pos_inds` is assumed to be ordered
        """
        raw_samp = np.random.randint(0, n_items - len(pos_inds), size=n_samp)
        pos_inds_adj = pos_inds - np.arange(len(pos_inds))
        neg_inds = raw_samp + np.searchsorted(pos_inds_adj, raw_samp, side='right')
        return neg_inds

    def get_adjacency_matrices(self):
        n_user = self.n_user
        n_item = self.n_item
        n_behavior = self.n_behavior
        data = self.data

        print("Constructing supra-adjacency matrix...")
        sys.stdout.flush()
        N = (n_user + n_item) * n_behavior
        n_node = n_user + n_item

        intra_adj_row_ind = []
        intra_adj_col_ind = []

        node_partition_row_ind = []
        node_partition_col_ind = []

        for user_id in data:
            for item_id in data[user_id]:
                for behavior in data[user_id][item_id]:
                    user_index_n = user_id
                    user_index_N = n_node * behavior + user_index_n
                    item_index_n = n_user + item_id
                    item_index_N = n_node * behavior + item_index_n
                    
                    intra_adj_row_ind.append(user_index_N)
                    intra_adj_col_ind.append(item_index_N)
                    node_partition_row_ind.append(user_index_N)
                    node_partition_col_ind.append(user_index_n)
                    node_partition_row_ind.append(item_index_N)
                    node_partition_col_ind.append(item_index_n)
                    
        # construct adjacency matrix which represents connections within each layer
        diag_adj = sparse.coo_matrix((np.ones(len(intra_adj_row_ind)), 
                                (intra_adj_row_ind, intra_adj_col_ind)), 
                                shape=(N, N)).tocsc()
        diag_adj += diag_adj.T
        diag_adj = diag_adj + sparse.diags(np.ones(diag_adj.shape[0]))
        
        # construct node partition matrix, which is matrix S in our paper
        node_partition_mat = sparse.coo_matrix((np.ones(len(node_partition_row_ind)), 
                                        (node_partition_row_ind, node_partition_col_ind)),
                                        shape=(N, n_node)).tocsc()
        node_partition_mat.data = np.ones(node_partition_mat.nnz)

        # compute the adjacency matrix of the quotient graph
        merged_adj = diag_adj[:n_node, :n_node]
        for i in range(1, n_behavior):
            merged_adj += diag_adj[i*n_node : (i+1)*n_node, i*n_node : (i+1)*n_node]

        normalizer_gamma = np.array(node_partition_mat.sum(0).flatten()).reshape([-1])+np.ones(n_node)  # normalization matrix, which is matrix Gamma in our paper
        normalizer_gamma_inv_sqrt  = sparse.diags(np.power(normalizer_gamma, -0.5))
        normalized_merged_adj = merged_adj.dot(normalizer_gamma_inv_sqrt).transpose().dot(normalizer_gamma_inv_sqrt)

        # construct adjacency matrix which represents connections across each layer
        coupling_mat_row_ind = []
        coupling_mat_col_ind = []

        for node in range(n_node):
            for b_row in range(n_behavior):
                for b_col in range(n_behavior):
                    row = node + b_row * n_node
                    col = node + b_col * n_node
                    if row != col:
                        coupling_mat_row_ind.append(row)
                        coupling_mat_col_ind.append(col)

        coupling_mat = sparse.coo_matrix((np.ones(len(coupling_mat_row_ind)), 
                                 (coupling_mat_row_ind, coupling_mat_col_ind)),
                                 shape=(N, N)).tocsc()

        # supra adjacency matrix of the multiplex graph
        supra_adj = diag_adj + coupling_mat
        normalizer_supra_adj = np.array(supra_adj.sum(1)).flatten()
        normalizer_supra_adj_inv_sqrt  = sparse.diags(np.power(normalizer_supra_adj, -0.5).flatten())
        supra_adj_normalized = supra_adj.dot(normalizer_supra_adj_inv_sqrt).transpose().dot(normalizer_supra_adj_inv_sqrt)
        
        print("done!")
        sys.stdout.flush()
        
        # return supra_adj_normalized.tocoo(), normalized_merged_adj.tocoo()
        # It is interesting that using the original merged_adj can achieve better results. Further research is needed.
        return supra_adj_normalized.tocoo(), merged_adj.tocoo()