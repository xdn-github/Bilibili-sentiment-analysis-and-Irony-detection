# coding=utf-8
# @Time : 2024/8/18 14:21
# @Author : 萧迪楠
# @File : sentiment_analysis.py
# @Software : PyCharm

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from nn_class import bert_lstm_class, sentiment_analysis_class


# CONFIG
class ModelConfig:
    batch_size = 32
    output_size = 2
    hidden_dim = 384  # 768/2
    n_layers = 2
    lr = 2e-5
    bidirectional = True  # 这里为True，为双向LSTM
    # training params
    k_fold = 10
    epochs = 5
    # batch_size=50
    print_every = 10
    clip = 5  # gradient clipping
    bert_path = 'model/bert-base-chinese/'  # 预训练bert路径
    save_path = 'model/sentiment_analysis_bert_bilstm.pth'  # 模型保存路径
    disable_warnings = True
    random_seed = 0
    use_cuda = torch.cuda.is_available()
    USE_CUDA = torch.cuda.is_available()
    # use_cuda = False
    data_path = 'datasets/17分区评论合集.csv'
    pre_token_name = 'bert-base-chinese'


model_config = ModelConfig()

# 显示计算平台是cpu还是gpu
USE_CUDA = torch.cuda.is_available()
# USE_CUDA = False
print(f'[USE CUDA {USE_CUDA}]')
# 关闭警报warnings
if model_config.disable_warnings:
    import warnings
    warnings.filterwarnings("ignore")

# 设计随机数
np.random.seed(model_config.random_seed)
torch.manual_seed(model_config.random_seed)
if USE_CUDA:
    torch.cuda.manual_seed(model_config.random_seed)

# 使用十折交叉验证对模型进行训练
k_fold = model_config.k_fold
column_ls = ['precision', 'recall', 'f1', 'accuracy']
result_df = pd.DataFrame(data=None, columns=column_ls)
for i_fold_round in range(k_fold):
    result_df.loc[i_fold_round, 'name'] = f'{i_fold_round + 1}/{k_fold} k-fold'
    model_config.save_path = f'model/sentiment_analysis_bert_bilstm[{i_fold_round + 1}-{k_fold}K_fold].pth'
    train_loader = sentiment_analysis_class.SentimentDataLoader_Func(model_config, 'train', i_fold_round, k_fold)
    test_loader = sentiment_analysis_class.SentimentDataLoader_Func(model_config, 'test', i_fold_round, k_fold)
    bert_lstm_class.train_model(model_config, train_loader)
    score_ls = bert_lstm_class.test_model(model_config, test_loader)
    for score_index in range(len(score_ls)):
        result_df.loc[i_fold_round, column_ls[score_index]] = float("{:.3f}".format(float(score_ls[score_index])))

# 为结果添加mean行和std行
mean_ls = result_df.drop(columns=['name']).mean()
std_ls = result_df.drop(columns=['name']).std()
result_df.loc[k_fold] = 'mean'
result_df.loc[k_fold + 1] = 'std'
for col in column_ls:
    result_df.loc[k_fold, col] = mean_ls[col]
    result_df.loc[k_fold + 1, col] = std_ls[col]
# 调换列的顺序, 让name列在最前面
result_df = result_df[list(result_df.columns)[::-1]]
result_df.to_csv(f'result/sentiment_analysis_{k_fold}_fold_result.csv', index=False)
