import sys
import os
import numpy as np
import pandas as pd
import ast
from collections import Counter, defaultdict

DATA_DIR = './data/'

def preprocess_db_book():
    DATASET = 'db_book/'
    behavior_map = {'uwish':0, 'ureading':1, 'uread':2, 'utag':3, 'ucomment':4, 'urating':5}
    data = []
    print('loading douban book dataset ...')
    sys.stdout.flush()

    n_interaction = 0
    user_dict = {}
    item_dict = {}
    for filename, behavior_type in behavior_map.items():
        with open(DATA_DIR + DATASET + filename + '.txt') as fin:
            for line in fin:
                u, i = map(int, line.strip().split('\t'))
                data.append([u, i, behavior_type])
                n_interaction += 1
                if u not in user_dict:
                    user_dict.add(u)
                if i not in item_dict:
                    item_dict.add(i)
    df_data = pd.DataFrame(np.array(data, dtype=int), columns=['user_id', 'item_id', 'behavior'])
    df_data.sort_values(['user_id', 'behavior'],inplace=True)

    print('dumping data ...')
    df_data.to_csv(DATA_DIR + DATASET + 'user_item_behavior.csv', header=False, index=False)
    np.savetxt(DATA_DIR + DATASET + 'behavior_map.csv', np.array(list(behavior_map.items())), fmt='%s', delimiter=',')
    print(n_interaction, "interactions about", len(user_dict), "users,", len(item_dict), "items.")
    print('done!')

def preprocess_steam():
    DATASET = 'steam/'
    behavior_map = {'purchase':0, 'play':1, 'review':2, 'recommend':3}
    user_candidate = set()

    print('loading steam review dataset ...')
    sys.stdout.flush()

    with open(DATA_DIR+DATASET+'australian_user_reviews.json') as fin:
        for line in fin:
            parsed_line = ast.literal_eval(line)
            user_id = parsed_line['user_id']
            if user_id not in user_candidate:
                for review in parsed_line['reviews']:
                    if review['recommend']:
                        user_candidate.add(user_id)
                        break

    data = defaultdict(lambda: defaultdict(set))
    with open(DATA_DIR+DATASET+'australian_user_reviews.json') as fin:
        for line in fin:
            parsed_line = ast.literal_eval(line)
            user_id = parsed_line['user_id']
            if user_id in user_candidate:
                for review in parsed_line['reviews']:
                    item_id = review['item_id']
                    data[user_id][item_id].add(behavior_map['review'])
                    if review['recommend']:
                         data[user_id][item_id].add(behavior_map['recommend'])
    
    print('loading steam_purchase dataset ...')
    sys.stdout.flush()

    with open(DATA_DIR+DATASET+'australian_users_items.json') as fin:
        for line in fin:
            parsed_line = ast.literal_eval(line)
            user_id = parsed_line['user_id']
            if user_id in user_candidate:
                for item in parsed_line['items']:
                    item_id = item['item_id']
                    data[user_id][item_id].add(behavior_map['purchase'])
                    if int(item['playtime_forever']) > 0:
                        data[user_id][item_id].add(behavior_map['play'])
    
    item_count = np.array(list(Counter([i for u in data for i in data[u]]).items()))
    item_set = set(list(item_count[item_count[:,1].astype(int)>=5, 0]))
    
    user_map = {}
    item_map = {}
    n_interaction = 0
    print('dumping data ...')
    sys.stdout.flush()

    with open(DATA_DIR+DATASET+'user_item_behavior.csv', "w") as fout:
        for user_id in data:
            items = data[user_id]
            for item_id in items:
                if item_id in item_set:
                    if item_id not in item_map:
                        item_map[item_id] = len(item_map)
                    iid = item_map[item_id]
                    if user_id not in user_map:
                        user_map[user_id] = len(user_map)
                    uid = user_map[user_id]
                    n_interaction += 1
                    for behavior in data[user_id][item_id]:
                        fout.write(str(uid)+','+str(iid)+','+str(behavior)+'\n')
                        
    np.savetxt(DATA_DIR+DATASET+'user_map.csv', np.array(list(user_map.items())), fmt="%s", delimiter=",")
    np.savetxt(DATA_DIR+DATASET+'item_map.csv', np.array(list(item_map.items())), fmt="%s", delimiter=",")
    np.savetxt(DATA_DIR+DATASET+'behavior_map.csv', np.array(list(behavior_map.items())), fmt='%s', delimiter=',')
    print(n_interaction, "interactions about", len(user_map), "users,", len(item_map), "items.")
    print('done!')
    
def preprocess_yoochoose():
    DATASET = 'yoochoose/'
    data = defaultdict(lambda: defaultdict(set))
    behavior_map = {'click':0, 'purchase':1}

    print('loading yoochoose dataset ...')
    sys.stdout.flush()

    with open(DATA_DIR+DATASET+'yoochoose-buys.dat') as fin:
        for line in fin:
            data_line = line.strip().split(",")
            user_id = int(data_line[0])
            item_id = int(data_line[2])
            data[user_id][item_id].add(behavior_map['purchase'])

    with open(DATA_DIR+DATASET+"yoochoose-clicks.dat") as fin:
        for line in fin:
            data_line = line.strip().split(",")
            user_id = int(data_line[0])
            item_id = int(data_line[2])
            if user_id in data:
                data[user_id][item_id].add(behavior_map['click'])
        
    item_count = np.array(list(Counter([i for u in data for i in data[u]]).items()))
    item_set = set(list(item_count[item_count[:,1].astype(float)>=5,0]))

    user_map = {}
    item_map = {}
    n_interaction = 0

    print('dumping data ...')
    sys.stdout.flush()

    with open(DATA_DIR+DATASET+"user_item_behavior.csv", "w") as fout:
        for user_id in data:
            items = data[user_id]
            for item_id in items:
                if item_id in item_set:
                    if item_id not in item_map:
                        item_map[item_id] = len(item_map)
                    iid = item_map[item_id]
                    if user_id not in user_map:
                        user_map[user_id] = len(user_map)
                    uid = user_map[user_id]
                    n_interaction += 1
                    for behavior in data[user_id][item_id]:
                        fout.write(str(uid)+','+str(iid)+','+str(behavior)+'\n')

    np.savetxt(DATA_DIR+DATASET+'user_map.csv', np.array(list(user_map.items())), fmt="%s", delimiter=",")
    np.savetxt(DATA_DIR+DATASET+'item_map.csv', np.array(list(item_map.items())), fmt="%s", delimiter=",")
    np.savetxt(DATA_DIR+DATASET+'behavior_map.csv', np.array(list(behavior_map.items())), fmt='%s', delimiter=',')
    print(n_interaction, "interactions about", len(user_map), "users,", len(item_map), "items.")
    print('done!')
    sys.stdout.flush()


if __name__ == '__main__':
    preprocess_yoochoose()