import os
import pickle
import pandas as pd
import multiprocessing
from tqdm import tqdm
from argparse import ArgumentParser
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

def parse_command_params():
    ap = ArgumentParser()
    ap.add_argument('-user_path', default="./data/user_data/Excuses", type=str, help='user out path')
    args = vars(ap.parse_args())

    return args

if __name__ == '__main__':

    args = parse_command_params()

    if not os.path.exists(args['user_path']):
        os.makedirs(args['user_path'])
        os.makedirs(os.path.join(args['user_path'], 'w2v'))

    attr = pd.read_csv('./data/row_data/attr.txt', sep='\t', names=['linkid', 'length', 'direction', 'pathclass', 'speedclass', 'LaneNum', 'speedlimit', 'level', 'width'])
    linkids = sorted(list(attr.linkid))

    with open(os.path.join(args['user_path'], 'linkids.pkl'), 'wb') as fw:
        pickle.dump(linkids, fw)

    #训练word2vec
    fw = open(os.path.join(args['user_path'], 'topo_w2v_train.txt'), 'w')
    with open("./data/row_data/topo.txt", 'r') as fr:
        lines = fr.readlines()
        for line in tqdm(lines):
            line = line.replace('\t', ' ').replace(',', ' ')
            fw.write(line)
        print(line)
    fr.close()
    fw.close()

    model = Word2Vec(LineSentence(os.path.join(args['user_path'], 'topo_w2v_train.txt')), alpha=0.03, size=64, window=3, min_count=2, sg=1, hs=1, workers=multiprocessing.cpu_count())
    model.save(os.path.join(args['user_path'], 'w2v/topo.model'))

    data = pd.concat([pd.DataFrame(model.wv.index2word, columns=['linkid'], dtype=int), pd.DataFrame(model.wv.vectors)], axis=1)
    ind = pd.DataFrame(linkids, columns=['linkid'])
    data = ind.merge(data, on='linkid', how='left').sort_values(by = ['linkid'], ascending=True).fillna(0.001)
    print(data.head())

    data.to_csv(os.path.join(args['user_path'], 'w2v/topo_emb.txt'), index=None, header=None)



