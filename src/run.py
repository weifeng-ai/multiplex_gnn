import os
import sys
sys.path.append('./src/')
from dataset import Dataset
from mgnn import MGNN
import argparse


DATA_DIR = "../data/"
MODEL_DIR = "../models/"
OUTPUT_DIR = "../results/"

for DIR in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    if not os.path.exists(DIR):
        os.makedirs(DIR)


def train(DATA_NAME, method, embedding_size, lbda, learning_rate, batch_size, save_result):
    dataset = Dataset(DATA_DIR, DATA_NAME)
    # dataset.split_train_test(seed=3, dump_splits=True)
    dataset.load_split_data()
    training_samples = dataset.sampling_training()

    mgnn = MGNN(dataset.n_user, dataset.n_item, dataset.n_behavior, DATA_DIR, DATA_NAME)
    mgnn.config_global(MODEL_CLASS="MGNN", HIDDEN_DIM=embedding_size, 
                              LAMBDA=lbda, LEARNING_RATE=learning_rate, BATCH_SIZE=batch_size)

    mgnn.load_data(training_samples, dataset.data_valid, dataset.data_test, dataset.data_ubi)
    supra_adj, merged_adj = dataset.get_adjacency_matrices()
    mgnn.load_matrix(supra_adj, merged_adj)

    mgnn.train(save_result)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='steam', help="choose dataset from [yoochoose, steam, db_book]")
    parser.add_argument('--method', default='MGNN', help="which method to use")
    parser.add_argument('--embedding_size', default=32,help="specify embedding size", type=int)
    parser.add_argument('--l2_weight', default=1e-5,help="specify the hyper-parameter to control l2 penalty", type=float)
    parser.add_argument('--learning_rate', default=0.00003, help="learning rate", type=float)
    parser.add_argument('--batch_size', default=1024, help='batch size', type=int)
    parser.add_argument('--save_result',default=False, help='save results or not?', type=bool)
   
    args = parser.parse_args()
    train(args.dataset, args.method, args.embedding_size, args.l2_weight, args.learning_rate, args.batch_size, args.save_result)