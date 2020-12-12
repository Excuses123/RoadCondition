
class Args:
    data_path = "/data/user_data/Excuses/traindata/"
    save_path = "/data/user_data/Excuses/output/"
    emb_path = '/data/user_data/Excuses/w2v/topo_emb.txt'

    fine_tune = True

    epoch = 10
    num_class = 4
    batch_size = 1024
    learning_rate = 0.0002
    embedding_size = 64
    vocab_size = 1

    filterSizes = [1, 2, 3]
    numFilters = 96
    seq_len = 5

    save_emb = False
    is_train = False
    is_test = False
    is_pred = False

