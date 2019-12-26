# -*- coding: utf-8 -*-
"""
@author: huangqiao
@file: expert-last
@time: 2019/12/16 16:02
"""
# %%
# 特征分析
import warnings

warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import gc
import pickle

# -------------------------------------------------------train---------------------------#
with open('pkl/train.pkl', 'rb') as file:
    invite_info = pickle.load(file)
invite_info.head()
# -------------------------------------------------------test---------------------------#
with open('pkl/test.pkl', 'rb') as file:
    test = pickle.load(file)
test.head()
# -------------------------------------------------------single_word---------------------------#
with open('pkl/single_word.pkl', 'rb') as file:
    single_word = pickle.load(file)
single_word.head()
# -------------------------------------------------------word---------------------------#
with open('pkl/word.pkl', 'rb') as file:
    word = pickle.load(file)
word.head()
# -------------------------------------------------------topic---------------------------#
with open('pkl/topic.pkl', 'rb') as file:
    topic = pickle.load(file)
topic.head()
# -------------------------------------------------------member_info---------------------------#

with open('pkl/user.pkl', 'rb') as file:
    member_info = pickle.load(file)
member_info.head()
# -------------------------------------------------------question_info---------------------------#

with open('pkl/question_info.pkl', 'rb') as file:
    question_info = pickle.load(file)
question_info.head()
# -------------------------------------------------------answer_info---------------------------#

with open('pkl/answer_info.pkl', 'rb') as file:
    answer_info = pickle.load(file)
answer_info.head()
# -------------------------------------------------------member_info---------------------------#

with open('pkl/user_feat.pkl', 'rb') as file:
    user_feat = pickle.load(file)
user_feat.head()
# %%
# 特征工程
import warnings

warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import pickle
import gc
import os
import time
import multiprocessing as mp
import logging
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm, tqdm_notebook, _tqdm_notebook, tqdm_pandas

tic = time.time()
SAVE_PATH = 'data/feats'
if not os.path.exists(SAVE_PATH):
    print('create dir: %s' % SAVE_PATH)
    os.mkdir(SAVE_PATH)
#####################################################################################################################################
######################################################user###########################################################################
with open('pkl/user.pkl', 'rb') as file:
    user = pickle.load(file)
logging.info("user %s", user.shape)


def parse_str(d):
    return np.array(list(map(float, d.split())))


with open('pkl/topic.pkl', 'rb') as file:
    topicmap = pickle.load(file)
topicmap.shape
topic_vector_dict = dict(zip(np.array(topicmap['id']), np.array(topicmap['embed'])))

type(topic_vector_dict.keys())


# 求话题向量平均值
def topic2v(x):
    try:
        tmp = topic_vector_dict[x[0]]
    except:
        tmp = np.zeros(64)
    for i in x[1:]:
        tmp = tmp + topic_vector_dict[i]
    if len(tmp) == 0:
        return np.zeros(64)
    return (tmp / len(x))


user.head()
tqdm.pandas(desc="topic2v...")
user['follow_topic_vector'] = user['follow_topic'].progress_apply(lambda x: topic2v(x))
print('finished!')


def topic_interest2v(x):
    if len(x) == 0:
        return np.zeros(64)
    else:
        tmp = np.zeros(64)
        for i in x:
            tmp = tmp + topic_vector_dict[i] * x[i]
        return (tmp / len(x))


tqdm.pandas(desc="topic_interest2v...")
user['inter_topic_vector'] = user['inter_topic'].progress_apply(lambda x: topic_interest2v(x))
print('finished!')
user.head()
user.shape


def listi(x, i):
    return x[i]


for i in range(64):
    col_name = 'topic_vector_{}'.format(str(i))
    tqdm.pandas(desc="topic_interest2v...")
    user[col_name] = user['follow_topic_vector'].apply(lambda x: listi(x, i))
for i in range(64):
    col_name = 'topic_interestvector_{}'.format(str(i))
    tqdm.pandas(desc="topic_interest2v...")
    user[col_name] = user['inter_topic_vector'].apply(lambda x: listi(x, i))
user.head()
user.shape
with open('pkl/user_feat.pkl', 'wb') as file:
    pickle.dump(user, file)
#######################################################添加
PATH = 'data'

user = pd.read_csv(os.path.join(PATH, 'member_info_0926.txt'),
                   names=['uid', 'gender', 'creat_keyword', 'level', 'hot', 'reg_type', 'reg_plat', 'freq', 'uf_b1',
                          'uf_b2',
                          'uf_b3', 'uf_b4', 'uf_b5', 'uf_c1', 'uf_c2', 'uf_c3', 'uf_c4', 'uf_c5', 'score',
                          'follow_topic',
                          'inter_topic'], sep='\t')
user.head()


def parse_list_1(d):
    if d == '-1':
        return [0]
    return list(map(lambda x: int(x[1:]), str(d).split(',')))


user['creat_keyword'] = user['creat_keyword'].apply(parse_list_1)
user.head()
with open('pkl/word.pkl', 'rb') as file:
    word = pickle.load(file)
word.shape
word_vector_dict = dict(zip(np.array(word['id']), np.array(word['embed'])))


def w2v(x):
    try:
        tmp = word_vector_dict[x[0]]
    except:
        tmp = np.zeros(64)
    for i in x[1:]:
        tmp = tmp + word_vector_dict[i]
    if len(tmp) == 0:
        return np.zeros(64)
    return (tmp / len(x))


tqdm.pandas(desc="w2v...")
user['keyword_vector'] = user['creat_keyword'].progress_apply(lambda x: w2v(x))
user.head()


def listi(x, i):
    return x[i]


for i in range(64):
    col_name = 'keyword_vector_{}'.format(str(i))
    tqdm.pandas(desc="w2v...")
    user[col_name] = user['keyword_vector'].apply(lambda x: listi(x, i))
user.shape
columns = ['uid']
for i in range(64):
    columns.append('keyword_vector_{}'.format(i))
keyword_vector = user[columns]
keyword_vector.head()
with open('pkl/user_keyword_feat.pkl', 'wb') as file:
    pickle.dump(keyword_vector, file)
######################################################################################################################################
###############################################question############################################################################
###################################################################加上topic的64维
with open('pkl/question_info.pkl', 'rb') as file:
    ques = pickle.load(file)
ques.shape
ques.head()
tqdm.pandas(desc="topic2v...")
ques['topic_vector'] = ques['topic'].progress_apply(lambda x: topic2v(x))
print('finished!')


def listi(x, i):
    return x[i]


for i in range(64):
    col_name = 'questopic_vector_{}'.format(str(i))
    tqdm.pandas(desc="topic_interest2v...")
    ques[col_name] = ques['topic_vector'].apply(lambda x: listi(x, i))
ques.head()
ques.shape
with open('pkl/ques_feat.pkl', 'wb') as file:
    pickle.dump(ques, file)
##############################################加上title_word切词的64维  title_t2
with open('pkl/ques_feat.pkl', 'rb') as file:
    ques = pickle.load(file)
ques.head()
ques.shape
with open('pkl/word.pkl', 'rb') as file:
    word = pickle.load(file)
word.shape
word_vector_dict = dict(zip(np.array(word['id']), np.array(word['embed'])))


def w2v(x):
    try:
        tmp = word_vector_dict[x[0]]
    except:
        tmp = np.zeros(64)
    for i in x[1:]:
        tmp = tmp + word_vector_dict[i]
    if len(tmp) == 0:
        return np.zeros(64)
    return (tmp / len(x))


tqdm.pandas(desc="w2v...")
ques['title_w_vector'] = ques['title_t2'].progress_apply(lambda x: w2v(x))
ques.head()


def parse_str(d):
    return np.array(list(map(float, d.split())))


def listi(x, i):
    return x[i]


from tqdm import tqdm, tqdm_notebook, _tqdm_notebook, tqdm_pandas

for i in range(64):
    col_name = 'title_w_vector_{}'.format(str(i))
    tqdm.pandas(desc="topic_interest2v...")
    ques[col_name] = ques['title_w_vector'].apply(lambda x: listi(x, i))
ques.head()
###################################################################加上内容切词  desc_t2的64维
tqdm.pandas(desc="w2v...")
ques['desc_w_vector'] = ques['desc_t2'].progress_apply(lambda x: w2v(x))
for i in range(64):
    col_name = 'desc_w_vector_{}'.format(str(i))
    tqdm.pandas(desc="topic_interest2v...")
    ques[col_name] = ques['desc_w_vector'].apply(lambda x: listi(x, i))
ques.head()
with open('pkl/ques_feat.pkl', 'wb') as file:
    pickle.dump(ques, file)
############################################################加上内容单字切词desc_t1的64维
with open('pkl/ques_feat.pkl', 'rb') as file:
    ques = pickle.load(file)
ques.head()
with open('pkl/single_word.pkl', 'rb') as file:
    single_word = pickle.load(file)
single_word.shape
single_word_vector_dict = dict(zip(np.array(single_word['id']), np.array(single_word['embed'])))


def sw2v(x):
    try:
        tmp = single_word_vector_dict[x[0]]
    except:
        tmp = np.zeros(64)
    for i in x[1:]:
        tmp = tmp + single_word_vector_dict[i]
    if len(tmp) == 0:
        return np.zeros(64)
    return (tmp / len(x))


tqdm.pandas(desc="sw2v...")
ques['desc_sw_vector'] = ques['desc_t1'].progress_apply(lambda x: sw2v(x))
ques.head()


def parse_str(d):
    return np.array(list(map(float, d.split())))


def listi(x, i):
    return x[i]


from tqdm import tqdm, tqdm_notebook, _tqdm_notebook, tqdm_pandas

for i in range(64):
    col_name = 'desc_sw_vector_{}'.format(str(i))
    tqdm.pandas(desc="sw2v...")
    ques[col_name] = ques['desc_sw_vector'].apply(lambda x: listi(x, i))
ques.shape
############################################################加上内容单字切词desc_t1的64维
tqdm.pandas(desc="sw2v...")
ques['title_sw_vector'] = ques['title_t1'].progress_apply(lambda x: sw2v(x))
for i in range(64):
    col_name = 'title_sw_vector_{}'.format(str(i))
    tqdm.pandas(desc="sw2v...")
    ques[col_name] = ques['title_sw_vector'].apply(lambda x: listi(x, i))
ques.shape
with open('pkl/ques_feat.pkl', 'wb') as file:
    pickle.dump(ques, file)
######################################################################################################################################
###############################################user_question_similary############################################################################
with open('pkl/ques_feat.pkl', 'rb') as file:
    ques = pickle.load(file)
ques.shape

columns = ['qid']

columns.append('topic_vector')
ques_topic = ques[columns]
ques_topic.head()
with open('pkl/user_feat.pkl', 'rb') as file:
    user = pickle.load(file)
user.shape
columns = ['uid']

columns.append('inter_topic_vector')
user_topic = user[columns]
user_topic.head()
uqid_sim = pd.merge(ques_topic, user_topic, on='qid')
uqid_sim.head(100)
#################################################################################################################################
#################################################merge(ans,ques)的topic#####################################################################
# 加载问题
with open('pkl/question_info.pkl', 'rb') as file:
    ques = pickle.load(file)

columns = ['qid']
columns.append('topic')
ques_topic = ques[columns]
ques_topic.head()
# 加载回答
with open('pkl/answer_info.pkl', 'rb') as file:
    ans = pickle.load(file)

columns = ['qid']
columns.append('uid')
ans_topic = ans[columns]
ans_topic.head()
# 将回答和问题信息按照qid进行合并
ans_topic_vector = pd.merge(ans_topic, ques_topic, on='qid')
del ques

# ans对于文本信息只留了topic
ans_topic_vector.head()
print(ans_topic_vector.shape)


def parse_str(d):
    return np.array(list(map(float, d.split())))


with open('pkl/topic.pkl', 'rb') as file:
    topicmap = pickle.load(file)
topicmap.shape
topic_vector_dict = dict(zip(np.array(topicmap['id']), np.array(topicmap['embed'])))

type(topic_vector_dict.keys())


# 求话题向量平均值
def topic2v(x):
    try:
        tmp = topic_vector_dict[x[0]]
    except:
        tmp = np.zeros(64)
    for i in x[1:]:
        tmp = tmp + topic_vector_dict[i]
    if len(tmp) == 0:
        return np.zeros(64)
    return (tmp / len(x))


tqdm.pandas(desc="topic2v...")
ans_topic_vector['topic_vector'] = ans_topic_vector['topic'].progress_apply(lambda x: topic2v(x))
print('finished!')
ans_topic_vector.head()
with open('pkl/user_feat.pkl', 'rb') as file:
    user = pickle.load(file)
user.shape
columns = ['uid']

columns.append('inter_topic_vector')
user_topic = user[columns]
user_topic.head()
answer_q_topic_vector = pd.merge(user_topic, ans_topic_vector, on='uid')
answer_q_topic_vector.head()
for i in range(64):
    col_name = 'answer_q_topic_vector_{}'.format(str(i))
    tqdm.pandas(desc="topic_interest2v...")
    ans_topic_vector[col_name] = ans_topic_vector['topic_vector'].apply(lambda x: listi(x, i))
# %%
# 模型训练
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import logging
import pickle

with open('pkl/data_vecor.pkl', 'rb') as file:
    data_a = pickle.load(file)
data_a.head()
feature_cols = [x for x in data_a.columns if x not in ('label', 'uid', 'qid', 'dt', 'day')]
# target编码
# train_label = train[(train['day'] > train_label_feature_end)]
# print(len(train_label))
train_label = 2593669
X_train_all = data_a.iloc[:train_label][feature_cols]
y_train_all = data_a.iloc[:train_label]['label']
test = data_a.iloc[train_label:]

logging.info("train shape %s, test shape %s", X_train_all.shape, test.shape)
print(X_train_all.shape)
print(test.shape)
model_lgb = LGBMClassifier(boosting_type='gbdt', num_leaves=64, learning_rate=0.01, n_estimators=2000,
                           max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                           min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                           colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=-1, silent=True)
fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for index, (train_idx, val_idx) in enumerate(fold.split(X=X_train_all, y=y_train_all)):
    X_train, X_val, y_train, y_val = X_train_all.iloc[train_idx][feature_cols], X_train_all.iloc[val_idx][feature_cols], \
                                     y_train_all.iloc[train_idx], \
                                     y_train_all.iloc[val_idx]
    model_lgb.fit(X_train, y_train,
                  eval_metric=['logloss', 'auc'],
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=10)
sub = pd.read_csv(f'data/invite_info_evaluate_1_0926.txt', sep='\t', header=None)
sub.columns = ['qid', 'uid', 'dt']
sub['label'] = model_lgb.predict_proba(test[feature_cols])[:, 1]
sub.to_csv('result.txt', index=None, header=None, sep='\t')
# %%
