# -*- coding: utf-8 -*-
"""
@author: huangqiao
@file: expert
@time: 2019/12/16 15:21
"""

# 一、特征预处理
# 数据预处理，包括解析列表，重编码id，pickle保存。运行时间 1388s，内存占用峰值 125G * 30%
import pandas as pd
import numpy as np
import pickle
import gc
from tqdm import tqdm_notebook
import os
import time
import logging
from sklearn.preprocessing import LabelEncoder

tic = time.time()


# 减少内存占用
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# 解析列表， 重编码id
def parse_str(d):
    return np.array(list(map(float, d.split())))


def parse_list_1(d):
    if d == '-1':
        return [0]
    return list(map(lambda x: int(x[1:]), str(d).split(',')))


def parse_list_2(d):
    if d == '-1':
        return [0]
    return list(map(lambda x: int(x[2:]), str(d).split(',')))


def parse_map(d):
    if d == '-1':
        return {}
    return dict([int(z.split(':')[0][1:]), float(z.split(':')[1])] for z in d.split(','))


PATH = 'data'
SAVE_PATH = 'pkl'
if not os.path.exists(SAVE_PATH):
    print('create dir: %s' % SAVE_PATH)
    os.mkdir(SAVE_PATH)
# single word
single_word = pd.read_csv(os.path.join(PATH, 'single_word_vectors_64d.txt'),
                          names=['id', 'embed'], sep='\t')
single_word.head()
# 把embed变成列表  id变成int
single_word['embed'] = single_word['embed'].apply(parse_str)
single_word['id'] = single_word['id'].apply(lambda x: int(x[2:]))
single_word.head()
with open('pkl/single_word.pkl', 'wb') as file:
    pickle.dump(single_word, file)

del single_word
gc.collect()
# ---------------------word-----------------------------#

word = pd.read_csv(os.path.join(PATH, 'word_vectors_64d.txt'),
                   names=['id', 'embed'], sep='\t')
word.head()
# 把embed变成列表  id变成int
word['embed'] = word['embed'].apply(parse_str)
word['id'] = word['id'].apply(lambda x: int(x[1:]))
word.head()
with open('pkl/word.pkl', 'wb') as file:
    pickle.dump(word, file)

del word
gc.collect()
# ---------------------topic-----------------------------#
topic = pd.read_csv(os.path.join(PATH, 'topic_vectors_64d.txt'),
                    names=['id', 'embed'], sep='\t')
topic.head()
# 把embed变成列表  id变成int
topic['embed'] = topic['embed'].apply(parse_str)
topic['id'] = topic['id'].apply(lambda x: int(x[1:]))
topic.head()
with open('pkl/topic.pkl', 'wb') as file:
    pickle.dump(topic, file)

del topic
gc.collect()
# ---------------------invite-----------------------------#
train = pd.read_csv(os.path.join(PATH, 'invite_info_0926.txt'),
                    names=['qid', 'uid', 'dt', 'label'], sep='\t')
test = pd.read_csv(os.path.join(PATH, 'invite_info_evaluate_1_0926.txt'),
                   names=['qid', 'uid', 'dt'], sep='\t')
train.head()


# train['invite_day'] = train['invite_time'].apply(lambda x: int(x.split('-')[0][1:])).astype(np.int16)
# train['invite_hour'] = train['invite_time'].apply(lambda x: int(x.split('-')[1][1:])).astype(np.int8)
# test['invite_day'] = test['invite_time'].apply(lambda x: int(x.split('-')[0][1:])).astype(np.int16)
# test['invite_hour'] = test['invite_time'].apply(lambda x: int(x.split('-')[1][1:])).astype(np.int8)
# train = reduce_mem_usage(train)
# train.head()


def extract_day(s):
    return s.apply(lambda x: int(x.split('-')[0][1:]))


def extract_hour(s):
    return s.apply(lambda x: int(x.split('-')[1][1:]))


logging.info("test %s", test.shape)

sub = test.copy()

sub_size = len(sub)

train['day'] = extract_day(train['dt'])
train['hour'] = extract_hour(train['dt'])

test['day'] = extract_day(test['dt'])
test['hour'] = extract_hour(test['dt'])

del train['dt'], test['dt']
train.head()
test.head()
with open('pkl/train.pkl', 'wb') as file:
    pickle.dump(train, file)

with open('pkl/test.pkl', 'wb') as file:
    pickle.dump(test, file)

del train, test
gc.collect()
# ---------------------member-----------------------------#
user = pd.read_csv(os.path.join(PATH, 'member_info_0926.txt'),
                   names=['uid', 'gender', 'creat_keyword', 'level', 'hot', 'reg_type', 'reg_plat', 'freq', 'uf_b1',
                          'uf_b2',
                          'uf_b3', 'uf_b4', 'uf_b5', 'uf_c1', 'uf_c2', 'uf_c3', 'uf_c4', 'uf_c5', 'score',
                          'follow_topic',
                          'inter_topic'], sep='\t')
user.head()
logging.info("user %s", user.shape)

unq = user.nunique()
logging.info("user unq %s", unq)

for x in unq[unq == 1].index:
    del user[x]
    logging.info('del unq==1 %s', x)

t = user.dtypes
cats = [x for x in t[t == 'object'].index if x not in ['follow_topic', 'inter_topic', 'uid']]
logging.info("user cat %s", cats)

for d in cats:
    lb = LabelEncoder()
    user[d] = lb.fit_transform(user[d])
    logging.info('encode %s', d)

user.columns
# 删除了'creat_keyword', 'level', 'hot', 'reg_type', 'reg_plat',
user.head()
user['follow_topic'] = user['follow_topic'].apply(parse_list_1)
user['inter_topic'] = user['inter_topic'].apply(parse_map)
user = reduce_mem_usage(user)
user.head()
with open('pkl/user.pkl', 'wb') as file:
    pickle.dump(user, file)

del user
gc.collect()
# ---------------------question-----------------------------#
question_info = pd.read_csv(os.path.join(PATH, 'question_info_0926.txt'),
                            names=['qid', 'q_dt', 'title_t1', 'title_t2', 'desc_t1', 'desc_t2', 'topic'], sep='\t')
question_info.head()
logging.info("ques %s", question_info.shape)

question_info['q_day'] = extract_day(question_info['q_dt'])
question_info['q_hour'] = extract_hour(question_info['q_dt'])
del question_info['q_dt']
question_info['title_t1'] = question_info['title_t1'].apply(parse_list_2)  # .apply(sw_lbl_enc.transform).apply(list)
question_info['title_t2'] = question_info['title_t2'].apply(parse_list_1)  # .apply(w_lbl_enc.transform).apply(list)
question_info['desc_t1'] = question_info['desc_t1'].apply(parse_list_2)  # .apply(sw_lbl_enc.transform).apply(list)
question_info['desc_t2'] = question_info['desc_t2'].apply(parse_list_1)  # .apply(w_lbl_enc.transform).apply(list)
question_info['topic'] = question_info['topic'].apply(parse_list_1)  # .apply(topic_lbl_enc.transform).apply(list)

gc.collect()
question_info = reduce_mem_usage(question_info)
question_info.head()
with open('pkl/question_info.pkl', 'wb') as file:
    pickle.dump(question_info, file)

del question_info
gc.collect()
# %%time
# ---------------------answer-----------------------------#


answer_info = pd.read_csv(os.path.join(PATH, 'answer_info_0926.txt'),
                          names=['aid', 'qid', 'uid', 'ans_dt', 'ans_t1', 'ans_t2', 'is_good', 'is_rec', 'is_dest',
                                 'has_img',
                                 'has_video', 'word_count', 'reci_cheer', 'reci_uncheer', 'reci_comment', 'reci_mark',
                                 'reci_tks',
                                 'reci_xxx', 'reci_no_help', 'reci_dis'], sep='\t')
answer_info.head()


def extract_day(s):
    return s.apply(lambda x: int(x.split('-')[0][1:]))


def extract_hour(s):
    return s.apply(lambda x: int(x.split('-')[1][1:]))


answer_info['ans_t1'] = answer_info['ans_t1'].apply(parse_list_2)
answer_info['ans_t2'] = answer_info['ans_t2'].apply(parse_list_1)

# logging.info("ans %s", answer_info.shape)

answer_info['a_day'] = extract_day(answer_info['ans_dt'])
answer_info['a_hour'] = extract_hour(answer_info['ans_dt'])
del answer_info['ans_dt']
# 回答距提问的天数
# answer_info['diff_qa_days'] = answer_info['a_day'] - answer_info['q_day']

gc.collect()
answer_info = reduce_mem_usage(answer_info)
answer_info.head()
with open('pkl/answer_info.pkl', 'wb') as file:
    pickle.dump(answer_info, file)

del answer_info
gc.collect()
toc = time.time()
print('Used time: %d' % int(toc - tic))



# %%
# 特征抽取与合并
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import logging
import pickle

log_fmt = "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
logging.basicConfig(format=log_fmt, level=logging.INFO)
import warnings

warnings.filterwarnings('ignore')


def extract_day(s):
    return s.apply(lambda x: int(x.split('-')[0][1:]))


def extract_hour(s):
    return s.apply(lambda x: int(x.split('-')[1][1:]))


# 加载训练、测试数据
with open('pkl/train.pkl', 'rb') as file:
    train = pickle.load(file)
logging.info("invite %s", train.shape)

with open('pkl/test.pkl', 'rb') as file:
    test = pickle.load(file)
logging.info("test %s", test.shape)
# 加载问题
with open('pkl/question_info.pkl', 'rb') as file:
    ques = pickle.load(file)

del ques['title_t1'], ques['title_t2'], ques['desc_t1'], ques['desc_t2']

logging.info("ques %s", ques.shape)
# 加载回答
with open('pkl/answer_info.pkl', 'rb') as file:
    ans = pickle.load(file)

del ans['ans_t1'], ans['ans_t2']
logging.info("ans %s", ans.shape)
# 将回答和问题信息按照qid进行合并
ans = pd.merge(ans, ques, on='qid')
del ques
# ans对于文本信息只留了topic
logging.info("ans %s", ans.shape)
# 回答距提问的天数
ans['diff_qa_days'] = ans['a_day'] - ans['q_day']

# 时间窗口划分
# train
# val
train_start = 3838
train_end = 3867

val_start = 3868
val_end = 3874

label_end = 3867
label_start = label_end - 6

train_label_feature_end = label_end - 7
train_label_feature_start = train_label_feature_end - 22

train_ans_feature_end = label_end - 7
train_ans_feature_start = train_ans_feature_end - 50

val_label_feature_end = val_start - 1
val_label_feature_start = val_label_feature_end - 22

val_ans_feature_end = val_start - 1
val_ans_feature_start = val_ans_feature_end - 50
train_label_feature = train[(train['day'] >= train_label_feature_start) & (train['day'] <= train_label_feature_end)]
logging.info("train_label_feature %s", train_label_feature.shape)

val_label_feature = train[(train['day'] >= val_label_feature_start) & (train['day'] <= val_label_feature_end)]
logging.info("val_label_feature %s", val_label_feature.shape)

train_label = train[(train['day'] > train_label_feature_end)]

logging.info("train feature start %s end %s, label start %s end %s", train_label_feature['day'].min(),
             train_label_feature['day'].max(), train_label['day'].min(), train_label['day'].max())

logging.info("test feature start %s end %s, label start %s end %s", val_label_feature['day'].min(),
             val_label_feature['day'].max(), test['day'].min(), test['day'].max())
# 确定ans的时间范围
# 3807~3874
train_ans_feature = ans[(ans['a_day'] >= train_ans_feature_start) & (ans['a_day'] <= train_ans_feature_end)]

val_ans_feature = ans[(ans['a_day'] >= val_ans_feature_start) & (ans['a_day'] <= val_ans_feature_end)]

logging.info("train ans feature %s, start %s end %s", train_ans_feature.shape, train_ans_feature['a_day'].min(),
             train_ans_feature['a_day'].max())

logging.info("val ans feature %s, start %s end %s", val_ans_feature.shape, val_ans_feature['a_day'].min(),
             val_ans_feature['a_day'].max())

fea_cols = ['is_good', 'is_rec', 'is_dest', 'has_img', 'has_video', 'word_count',
            'reci_cheer', 'reci_uncheer', 'reci_comment', 'reci_mark', 'reci_tks',
            'reci_xxx', 'reci_no_help', 'reci_dis', 'diff_qa_days']
logging.info("train_label %s", train_label.shape)
logging.info("ans %s", ans.shape)


def extract_feature1(target, label_feature, ans_feature):
    # 问题特征
    t1 = label_feature.groupby('qid')['label'].agg(['mean', 'sum', 'std', 'count']).reset_index()
    t1.columns = ['qid', 'q_inv_mean', 'q_inv_sum', 'q_inv_std', 'q_inv_count']
    target = pd.merge(target, t1, on='qid', how='left')

    # 用户特征
    t1 = label_feature.groupby('uid')['label'].agg(['mean', 'sum', 'std', 'count']).reset_index()
    t1.columns = ['uid', 'u_inv_mean', 'u_inv_sum', 'u_inv_std', 'u_inv_count']
    target = pd.merge(target, t1, on='uid', how='left')
    #
    # train_size = len(train)
    # data = pd.concat((train, test), sort=True)

    # 回答部分特征

    t1 = ans_feature.groupby('qid')['aid'].count().reset_index()
    t1.columns = ['qid', 'q_ans_count']
    target = pd.merge(target, t1, on='qid', how='left')

    t1 = ans_feature.groupby('uid')['aid'].count().reset_index()
    t1.columns = ['uid', 'u_ans_count']
    target = pd.merge(target, t1, on='uid', how='left')

    for col in fea_cols:
        t1 = ans_feature.groupby('uid')[col].agg(['sum', 'max', 'mean']).reset_index()
        t1.columns = ['uid', f'u_{col}_sum', f'u_{col}_max', f'u_{col}_mean']
        target = pd.merge(target, t1, on='uid', how='left')

        t1 = ans_feature.groupby('qid')[col].agg(['sum', 'max', 'mean']).reset_index()
        t1.columns = ['qid', f'q_{col}_sum', f'q_{col}_max', f'q_{col}_mean']
        target = pd.merge(target, t1, on='qid', how='left')
        logging.info("extract %s", col)
    return target


# 共加了100个特征
train_label = extract_feature1(train_label, train_label_feature, train_ans_feature)
pd.options.display.max_columns = None
train_label.head()
test = extract_feature1(test, val_label_feature, val_ans_feature)
test.head()
logging.info("train_label %s", train_label.shape)
logging.info("test %s", test.shape)
logging.info("ans %s", ans.shape)
ans.head()
# 加载用户
with open('pkl/user.pkl', 'rb') as file:
    user = pickle.load(file)

del user['follow_topic'], user['inter_topic']
logging.info("user %s", user.shape)
# 删除用户特征中的常量
unq = user.nunique()
logging.info("user unq %s", unq)

for x in unq[unq == 1].index:
    del user[x]
    logging.info('del unq==1 %s', x)
logging.info("user %s", user.shape)
# 对多特征值进行编码
t = user.dtypes
cats = [x for x in t[t == 'object'].index if x not in ['follow_topic', 'inter_topic', 'uid']]
logging.info("user cat %s", cats)

for d in cats:
    lb = LabelEncoder()
    user[d] = lb.fit_transform(user[d])
    logging.info('encode %s', d)

logging.info("user %s", user.shape)
# 对uid和qid进行编码
q_lb = LabelEncoder()
q_lb.fit(list(train_label['qid'].astype(str).values) + list(test['qid'].astype(str).values))
train_label['qid_enc'] = q_lb.transform(train_label['qid'])
test['qid_enc'] = q_lb.transform(test['qid'])

u_lb = LabelEncoder()
u_lb.fit(user['uid'])
train_label['uid_enc'] = u_lb.transform(train_label['uid'])
test['uid_enc'] = u_lb.transform(test['uid'])
logging.info("user %s", user.shape)
# merge user之前
logging.info("train_label %s", train_label.shape)
logging.info("test %s", test.shape)
# merge user
train_label = pd.merge(train_label, user, on='uid', how='left')
test = pd.merge(test, user, on='uid', how='left')
logging.info("train shape %s, test shape %s", train_label.shape, test.shape)

data = pd.concat((train_label, test), axis=0, sort=True)
del train_label, test
logging.info("data %s", data.shape)
# count编码
count_fea = ['uid_enc', 'qid_enc', 'gender', 'freq', 'uf_c1', 'uf_c2', 'uf_c3', 'uf_c4', 'uf_c5']
for feat in count_fea:
    col_name = '{}_count'.format(feat)
    data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
    data.loc[data[col_name] < 2, feat] = -1
    data[feat] += 1
    data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
    data[col_name] = (data[col_name] - data[col_name].min()) / (data[col_name].max() - data[col_name].min())
data.head()
logging.info("data %s", data.shape)
# 压缩数据
t = data.dtypes
for x in t[t == 'int64'].index:
    data[x] = data[x].astype('int32')

for x in t[t == 'float64'].index:
    data[x] = data[x].astype('float32')

data['wk'] = data['day'] % 7
feature_cols = [x for x in data.columns if x not in ('label', 'uid', 'qid', 'dt', 'day')]
# target编码
logging.info("feature size %s", len(feature_cols))
logging.info("data %s", data.shape)
# 保存处理好的data
with open('pkl/data.pkl', 'wb') as file:
    pickle.dump(data, file)
######################################################################################################################################
#######################################特征处理#######################################################################################
import gc
import os
import time
import multiprocessing as mp
import logging
import pickle
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm, tqdm_notebook, _tqdm_notebook, tqdm_pandas

tic = time.time()
#############################user#################################
with open('pkl/user_feat.pkl', 'rb') as file:
    user = pickle.load(file)
user.head()
user.shape
columns = ['uid']
for i in range(64):
    columns.append('topic_vector_{}'.format(i))
user_topic = user[columns]
user_topic_vector = user[columns]
user_topic_vector.head()
user_topic_vector.shape
columns = ['uid']
for i in range(64):
    columns.append('topic_interestvector_{}'.format(i))
user_intertopic_vector = user[columns]
user_intertopic_vector = user[columns]
user_intertopic_vector.head()
###########################  添加 keyword_vector_64
with open('pkl/user_keyword_feat.pkl', 'rb') as file:
    user_keyword = pickle.load(file)
user_keyword.head()
#############################ques#################################
with open('pkl/ques_feat.pkl', 'rb') as file:
    ques = pickle.load(file)
ques.shape
columns = ['qid']
for i in range(64):
    columns.append('questopic_vector_{}'.format(i))
ques_topic = ques[columns]
ques_topic.head()
ques_topic.shape
with open('pkl/ques_feat.pkl', 'rb') as file:
    ques = pickle.load(file)
ques.shape
columns = ['qid']
for i in range(64):
    columns.append('title_w_vector_{}'.format(i))
ques_w_topic = ques[columns]
with open('pkl/ques_feat.pkl', 'rb') as file:
    ques = pickle.load(file)
ques.shape
columns = ['qid']
for i in range(64):
    columns.append('desc_w_vector_{}'.format(i))
ques_w_desc = ques[columns]
ques_w_desc.head()
ques.shape
columns = ['qid']
for i in range(64):
    columns.append('desc_sw_vector_{}'.format(i))
ques_sw_desc = ques[columns]
columns = ['qid']
for i in range(64):
    columns.append('title_sw_vector_{}'.format(i))
ques_sw_title = ques[columns]
#############################合并################################# user_topic_vector    ques_topic
with open('pkl/data.pkl', 'rb') as file:
    data_a = pickle.load(file)
data_a.shape
import pandas as pd
import numpy as np

data_a = pd.merge(data_a, user_topic_vector, how='left', left_on='uid', right_on='uid')
data_a.head()
data_a.shape
data_a = pd.merge(data_a, ques_topic, how='left', left_on='qid', right_on='qid')
data_a.head()
data_a.shape
# 保存处理好的data_a
with open('pkl/data_vecor.pkl', 'wb') as file:
    pickle.dump(data_a, file)
#######################################################添加 user_intertopic_vector
with open('pkl/data_vecor.pkl', 'rb') as file:
    data_a = pickle.load(file)
data_a.head()
data_a = pd.merge(data_a, user_intertopic_vector, how='left', left_on='uid', right_on='uid')
data_a.head()
# 保存处理好的data_a
with open('pkl/data_vecor.pkl', 'wb') as file:
    pickle.dump(data_a, file)
#######################################################################添加 ques_w_desc
with open('pkl/data_vecor.pkl', 'rb') as file:
    data_a = pickle.load(file)
data_a.head()
data_a = pd.merge(data_a, ques_w_desc, how='left', left_on='qid', right_on='qid')
data_a.head()
# 保存处理好的data_a
with open('pkl/data_vecor.pkl', 'wb') as file:
    pickle.dump(data_a, file)
#######################################################################添加 ques_w_topic
with open('pkl/data_vecor.pkl', 'rb') as file:
    data_a = pickle.load(file)
data_a.head()
data_a = pd.merge(data_a, ques_w_topic, how='left', left_on='qid', right_on='qid')
data_a.head()
# 保存处理好的data_a
with open('pkl/data_vecor.pkl', 'wb') as file:
    pickle.dump(data_a, file)
#######################################################################添加 desc_sw_vector
with open('pkl/data_vecor.pkl', 'rb') as file:
    data_a = pickle.load(file)
data_a.shape
data_a = pd.merge(data_a, ques_sw_desc, how='left', left_on='qid', right_on='qid')
data_a.shape
#######################################################################添加 title_sw_vector
data_a = pd.merge(data_a, ques_sw_title, how='left', left_on='qid', right_on='qid')
data_a.shape
# 保存处理好的data_a
with open('pkl/data_vecor.pkl', 'wb') as file:
    pickle.dump(data_a, file)
#######################################################################添加 user_keyword
with open('pkl/data_vecor.pkl', 'rb') as file:
    data_a = pickle.load(file)
data_a = pd.merge(data_a, user_keyword, how='left', left_on='uid', right_on='uid')
data_a.shape
# 保存处理好的data_a
with open('pkl/data_vecor.pkl', 'wb') as file:
    pickle.dump(data_a, file)
########################################################################################################################################
#############################################模型######################################################################################
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
#######################################################################
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
sub.to_csv('result31.txt', index=None, header=None, sep='\t')
