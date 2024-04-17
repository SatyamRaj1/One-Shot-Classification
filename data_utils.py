import pandas as pd
import json
import pickle
import numpy as np
from utils import ROOT_DIR

def load_toxicBias():
    train_df = pd.read_csv('./data/ToxicBias/Train.csv')
    train_df['bias'] = train_df['bias'].apply(lambda x: 1 if x=='bias' else 0)
    # train_sentences = list()
    # for i in range(train_df.shape[0]):
    #     d = {'sentence': train_df['comment_text'], 'label': train_df['bias'], 'idx': i}
    #     train_sentences.append(d)
    train_sentences = list(train_df['comment_text'])
    train_labels = list(train_df['bias'])

    test_df = pd.read_csv('./data/ToxicBias/Test.csv')
    test_df['bias'] = test_df['bias'].apply(lambda x: 1 if x=='bias' else 0)
    # test_sentences = list()
    # for i in range(test_df.shape[0]):
    #     d = {'sentence': test_df['comment_text'], 'label': test_df['bias'], 'idx': i}
    #     test_sentences.append(d)
    test_sentences = list(test_df['comment_text'])
    test_labels = list(test_df['bias'])
        
    return train_sentences, train_labels, test_sentences, test_labels

def load_dataset(params):
    """
    Load train and test data
    :param params: experiment parameter, which contains dataset spec
    :return: train_x, train_y, test_x, test_y
    """
    if params['dataset'] == 'toxicbias':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_toxicBias()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Review: "
        params["a_prefix"] = "Sentiment: "
        params['label_dict'] = {0: ['Negative'], 1: ['Positive']}
        params['inv_label_dict'] = {'Negative': 0, 'Positive': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    else:
        print(params['dataset'])
        raise NotImplementedError
    return orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels