from __future__ import print_function
import os
import numpy as np
import pickle
import bcolz

from config import DATA_PATH


def prepare_word_embedding(root="glove.840B", embed_dim=300):
    glove_path = os.path.join(DATA_PATH, root)

    words = []
    idx = 0
    word2idx = {}
    # vectors = bcolz.carray(np.zeros(1), rootdir='{}/6B.50.dat'.format(glove_path), mode='w')
    vectors = bcolz.carray(np.zeros(1),
                           rootdir='{}/processed'.format(glove_path), mode='w')

    # with open('{}/glove.6B.50d.txt'.format(glove_path), 'rb') as f:
    with open('{}/glove.840B.300d.txt'.format(glove_path), 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[-embed_dim:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((-1, embed_dim)),
                           rootdir='{}/processed'.format(glove_path), mode='w')
    print(vectors.shape)
    vectors.flush()
    # pickle.dump(words, open('{}/6B.50_words.pkl'.format(glove_path), 'wb'))
    # pickle.dump(word2idx, open('{}/6B.50_idx.pkl'.format(glove_path), 'wb'))
    pickle.dump(words, open('{}/840B.300_words.pkl'.format(glove_path), 'wb'))
    pickle.dump(word2idx, open('{}/840B.300_idx.pkl'.format(glove_path), 'wb'))


def prepare_vocab_embedding(target_vocab, root="glove.840B", embed_dim=300):
    glove_path = os.path.join(DATA_PATH, root)

    # TODO format filenames
    vectors = bcolz.open(f'{glove_path}/processed')[:]
    # words = pickle.load(open(f'{glove_path}/840B.300_words.pkl', 'rb'))
    word2idx_all = pickle.load(open(f'{glove_path}/840B.300_idx.pkl', 'rb'))

    word2idx = {}
    idx = 0

    # glove = {w: vectors[word2idx[w]] for w in words}

    matrix_len = len(target_vocab)
    weights_matrix = np.zeros((matrix_len, embed_dim))
    words_found = 0

    for i, word in enumerate(target_vocab):
        try:
            weights_matrix[i] = vectors[word2idx_all[word]]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(embed_dim,))
        word2idx[word] = idx
        idx += 1

    return weights_matrix, word2idx


def prepare_vocab(df, columns, custom_transforms=None, uncased=True):
    vocab = []

    if custom_transforms is None:
        custom_transforms = {}

    for column in columns:
        for entry in df[column]:

            try:
                entry = custom_transforms[entry]
            except KeyError:
                pass

            if uncased:
                entry = entry.lower()

            for word in entry.split(" "):
                if word not in vocab:
                    vocab.append(word)

    return vocab


def prepare_class_embedding(classes, vocab_embedding, word2idx,
                            pooling=np.mean, uncased=True):
    weights_matrix = np.zeros((len(classes), vocab_embedding.shape[1]))

    for c, class_ in enumerate(classes):
        if uncased:
            class_ = class_.lower()
        class_words = class_.split(" ")
        class_words_embeddings = []
        for word in class_words:
            class_words_embeddings.append(vocab_embedding[word2idx[word]])
        weights_matrix[c] = pooling(class_words_embeddings, axis=0)

    return weights_matrix