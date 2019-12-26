import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import logging

# logging打印信息函数
log_fmt = "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
logging.basicConfig(format=log_fmt, level=logging.INFO)

import warnings
warnings.filterwarnings('ignore')


# 获取邀请时间函数，形式为Day-Hour，如D3870-H9
def extract_day(s):
    return s.apply(lambda x: int(x.split('-')[0][1:]))


def extract_hour(s):
    return s.apply(lambda x: int(x.split('-')[1][1:]))


# 加载邀请回答数据,train为9489162*4矩阵,四列数据分别命名为['qid', 'uid', 'dt', 'label']
'''
数据集7，invite_info_0926.txt(9489162行*4列):
包含用户最近1个月的邀请数据，每一行代表一个问题邀请的相关信息,每一行有4列,列之间采用／tab分隔符分割
数据格式如下：[Qxxx   Mxxx   D3-H4   label]
1.邀请的问题ID, 格式为 Qxxx。
2.被邀请用户ID, 格式为 Mxxx。
3.邀请创建时间, 格式为 D3-H4。
4.邀请是否被回答, 如果值为1表示邀请被回答, 值为0表示邀请没有被回答。
'''
train = pd.read_csv(f'train/invite_info_0926.txt', sep='\t', header=None)
train.columns = ['qid', 'uid', 'dt', 'label']
logging.info("invite %s", train.shape)
# [2019-12-15 19:38:02,989] INFO in baseline: invite (9489162, 4)


# 加载邀请数据验证集,test为1141683*3矩阵,三列数据分别命名为['qid', 'uid', 'dt']
'''
验证集8，invite_info_evaluate_0926.txt(1141682行*3列):
未来7天的问题邀请数据,每一行代表一个问题邀请相关信息,每一行有3列,列之间采用／tab分隔符分割。
数据格式如下：[Qxxx  Mxxx  D3-H4]
1.邀请的问题ID,格式为 Qxxx。
2.被邀请用户ID,格式为 Mxxx。
3.邀请创建时间,格式为 D3-H4。
'''
test = pd.read_csv(f'test/invite_info_evaluate_1_0926.txt', sep='\t', header=None)
test.columns = ['qid', 'uid', 'dt']
logging.info("test %s", test.shape)
# [2019-12-15 19:38:06,911] INFO in baseline: test (1141683, 3)


# 获取训练数据的邀请时间day，hour
train['day'] = extract_day(train['dt'])
train['hour'] = extract_hour(train['dt'])

# 获取测试数据的邀请时间day，hour
test['day'] = extract_day(test['dt'])
test['hour'] = extract_hour(test['dt'])

del train['dt'], test['dt']


# 加载问题数据,ques为1829900行*3列矩阵，['qid', 'q_dt', 'topic']
'''
数据集question_info_0926.txt (1829900行*3列）：
包含邀请数据集(数据集7和8)及回答数据集(数据集5)表中涉及到的所有问题列表，
每一行代表一个问题的相关信息, 每一行有7列, 列之间采用/tab分隔符分割。
数据格式如下：
[问题ID  问题创建时间  问题标题的单字编码序列  问题标题的切词编码序列  问题描述的单字编码序列  问题描述的词编码序列  问题绑定的话题ID]
1.问题ID, 格式为 Qxxx。
2.问题创建时间, 格式为 D3-H4。
3.问题标题的单字编码序列, 格式为 SW1,SW2,SW3,...,SWn , 表示问题标题的单字编码序号。
4.问题标题的切词编码序列, 格式为 W1,W2,W3,...,Wn , 表示问题标题的切词编码序号, 如果问题标题切词后为空, 则用 -1 进行占位。
5.问题描述的单字编码序列, 格式为 SW1,SW2,SW3,...,SWn , 表示问题描述的单字编码序号, 如果问题没有描述, 则用 -1 进行占位。
6.问题描述的切词编码序列, 格式为 W1,W2,W3,...,Wn , 表示问题描述的切词编码序号, 如果问题没有描述或者描述切词后为空, 则用 -1 进行占位。
7.问题绑定的话题 ID, 格式为 T1,T2,T3,...,Tn , 表示问题绑定的话题 ID 的编码序号， 如果问题没有绑定的话题，则用 -1 进行占位。
'''
ques = pd.read_csv(f'train/question_info_0926.txt', header=None, sep='\t')
ques.columns = ['qid', 'q_dt', 'title_t1', 'title_t2', 'desc_t1', 'desc_t2', 'topic']
# 取消baseline中对特征的删除
# 删除了'title_t1','title_t2'问题标题和'desc_t1','desc_t2'问题描述的四个特征，只留下了topic信息
# del ques['title_t1'], ques['title_t2'], ques['desc_t1'], ques['desc_t2']
logging.info("ques %s", ques.shape)

# 获取问题的创建时间day，hour
ques['q_day'] = extract_day(ques['q_dt'])
ques['q_hour'] = extract_hour(ques['q_dt'])
del ques['q_dt']


# 加载回答
'''
数据集5 answer_info_0926.txt（4,513,735）：
为邀请数据集(数据集7和8)中用户最近2个月内的所有回答，每一行代表一个回答的相关信息,每一行有20列
数据格式如下：
[回答ID 问题ID 用户ID 回答创建时间 回答内容的单字编码序列 回答内容的切词编码序列 回答是否被标优 回答是否被推荐 
回答是否被收入圆桌 是否包含图片 是否包含视频 回答字数 点赞数 取赞数 评论数 收藏数 感谢数 举报数 没有帮助数 反对数]
1.回答ID, 格式为 Axxx。
2.问题ID, 格式为 Qxxx。
3.作者ID, 格式为 Mxxx。
4.回答创建时间, 格式为 D3-H4。
5.回答内容的单字编码序列, 格式为 SW1,SW2,SW3,...,SWn , 表示回答内容的单字编码序号, 如果回答内容为空, 则用 -1 进行占位。
6.回答内容的切词编码序列, 格式为 W1,W2,W3,...,Wn , 表示回答内容的切词编码序号, 如果回答内容为空或者回答内容切词后为空, 则用 -1 进行占位。
7.回答是否被标为优秀回答。
8.回答是否被推荐。
9.回答是否被收入圆桌。
10.回答是否包含图片。
11.回答是否包含视频。
12.回答的内容字数。
13.回答收到的点赞数。
14.回答收到的取赞数。
15.回答收到的评论数。
16.回答收藏数。
17.回答收到的感谢数。
18.回答收到的被举报数。
19.回答收到的没有帮助数。
20.回答收到的反对数。

'''
ans = pd.read_csv(f'train/answer_info_0926.txt', header=None, sep='\t')
ans.columns = ['aid', 'qid', 'uid', 'ans_dt', 'ans_t1', 'ans_t2', 'is_good', 'is_rec', 'is_dest', 'has_img',
               'has_video', 'word_count', 'reci_cheer', 'reci_uncheer', 'reci_comment', 'reci_mark', 'reci_tks',
               'reci_xxx', 'reci_no_help', 'reci_dis']
# 取消baseline中对特征的删除
# 删除了'ans_t1', 'ans_t2'5.回答内容的单字编码序列和切词编码序列的两个特征
# del ans['ans_t1'], ans['ans_t2']
logging.info("ans %s", ans.shape)

ans['a_day'] = extract_day(ans['ans_dt'])
ans['a_hour'] = extract_hour(ans['ans_dt'])
del ans['ans_dt']


# In[10]:


#将回答和问题信息按照qid进行合并
ans = pd.merge(ans, ques, on='qid')
del ques


# In[11]:


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


# In[12]:


train_label_feature = train[(train['day'] >= train_label_feature_start) & (train['day'] <= train_label_feature_end)]
logging.info("train_label_feature %s", train_label_feature.shape)

val_label_feature = train[(train['day'] >= val_label_feature_start) & (train['day'] <= val_label_feature_end)]
logging.info("val_label_feature %s", val_label_feature.shape)

train_label = train[(train['day'] > train_label_feature_end)]

logging.info("train feature start %s end %s, label start %s end %s", train_label_feature['day'].min(),
             train_label_feature['day'].max(), train_label['day'].min(), train_label['day'].max())

logging.info("test feature start %s end %s, label start %s end %s", val_label_feature['day'].min(),
             val_label_feature['day'].max(), test['day'].min(), test['day'].max())


# In[13]:


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


# In[14]:


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


# In[15]:


train_label = extract_feature1(train_label, train_label_feature, train_ans_feature)


# In[12]:


pd.options.display.max_columns = None


# In[17]:


print(train_label.head())


# In[18]:


test = extract_feature1(test, val_label_feature, val_ans_feature)


# In[19]:


# 加载用户
user = pd.read_csv(f'train/member_info_0926.txt', header=None, sep='\t')
user.columns = ['uid', 'gender', 'creat_keyword', 'level', 'hot', 'reg_type', 'reg_plat', 'freq', 'uf_b1', 'uf_b2',
                'uf_b3', 'uf_b4', 'uf_b5', 'uf_c1', 'uf_c2', 'uf_c3', 'uf_c4', 'uf_c5', 'score', 'follow_topic',
                'inter_topic']
del user['follow_topic'], user['inter_topic']
logging.info("user %s", user.shape)


# In[20]:


#删除用户特征中的常量
unq = user.nunique()
logging.info("user unq %s", unq)

for x in unq[unq == 1].index:
    del user[x]
    logging.info('del unq==1 %s', x)


# In[21]:


#对多特征值进行编码
t = user.dtypes
cats = [x for x in t[t == 'object'].index if x not in ['follow_topic', 'inter_topic', 'uid']]
logging.info("user cat %s", cats)

for d in cats:
    lb = LabelEncoder()
    user[d] = lb.fit_transform(user[d])
    logging.info('encode %s', d)


# In[22]:


#对uid和qid进行编码
q_lb = LabelEncoder()
q_lb.fit(list(train_label['qid'].astype(str).values) + list(test['qid'].astype(str).values))
train_label['qid_enc'] = q_lb.transform(train_label['qid'])
test['qid_enc'] = q_lb.transform(test['qid'])

u_lb = LabelEncoder()
u_lb.fit(user['uid'])
train_label['uid_enc'] = u_lb.transform(train_label['uid'])
test['uid_enc'] = u_lb.transform(test['uid'])


# In[23]:


# merge user
train_label = pd.merge(train_label, user, on='uid', how='left')
test = pd.merge(test, user, on='uid', how='left')
logging.info("train shape %s, test shape %s", train_label.shape, test.shape)

data = pd.concat((train_label, test), axis=0, sort=True)
# del train_label, test


# In[24]:


# count编码，不太明白作用
count_fea = ['uid_enc', 'qid_enc', 'gender', 'freq', 'uf_c1', 'uf_c2', 'uf_c3', 'uf_c4', 'uf_c5']
for feat in count_fea:
    col_name = '{}_count'.format(feat)
    data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
    data.loc[data[col_name] < 2, feat] = -1
    data[feat] += 1
    data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
    data[col_name] = (data[col_name] - data[col_name].min()) / (data[col_name].max() - data[col_name].min())


# In[25]:


print(data.head())


# In[26]:


# 压缩数据
t = data.dtypes
for x in t[t == 'int64'].index:
    data[x] = data[x].astype('int32')

for x in t[t == 'float64'].index:
    data[x] = data[x].astype('float32')

data['wk'] = data['day'] % 7


# In[27]:


feature_cols = [x for x in data.columns if x not in ('label', 'uid', 'qid', 'dt', 'day')]
# target编码
logging.info("feature size %s", len(feature_cols))


# In[29]:
# 文件修改352   可以输出一下看看是否一样的结果
# train_label=2593669
train_label = train[(train['day'] > train_label_feature_end)]
print("train_label =",train_label)

# In[31]:


X_train_all = data.iloc[:len(train_label)][feature_cols]
y_train_all = data.iloc[:len(train_label)]['label']
test = data.iloc[len(train_label):]

logging.info("train shape %s, test shape %s", train_label.shape, test.shape)


# In[33]:


import pickle


# In[34]:


with open('train/data.pkl','wb') as file:
    pickle.dump(data,file)


# In[35]:


logging.info("train shape %s, test shape %s", X_train_all.shape, test.shape)


# In[36]:


train_label.shape


# In[37]:


fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for index, (train_idx, val_idx) in enumerate(fold.split(X=X_train_all, y=y_train_all)):
    break

X_train, X_val, y_train, y_val = X_train_all.iloc[train_idx][feature_cols], X_train_all.iloc[val_idx][feature_cols], \
                                 y_train_all.iloc[train_idx],y_train_all.iloc[val_idx]

# In[45]:


model_lgb = LGBMClassifier(boosting_type='gbdt', num_leaves=64, learning_rate=0.01, n_estimators=2000,
                           max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                           min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                           colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=-1, silent=True)
model_lgb.fit(X_train, y_train,
              eval_metric=['logloss', 'auc'],
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=10)


# In[46]:


sub = pd.read_csv(f'test/invite_info_evaluate_1_0926.txt', sep='\t', header=None)
sub.columns = ['qid', 'uid', 'dt']
logging.info("test %s", sub.shape)


# In[47]:


sub['label'] = model_lgb.predict_proba(test[feature_cols])[:, 1]


# In[48]:


print(sub.head())


# In[49]:


sub.to_csv('result/result11.txt', index=None, header=None, sep='\t')


# In[50]:
'''


data.head()


def getsimilarity(uid,qid):
    s=uid+qid
    if s not in uqid_sim:
        return None
    else:
        return uqid_sim[uid+qid]


# In[16]:


from tqdm import tqdm, tqdm_notebook, _tqdm_notebook, tqdm_pandas


# In[17]:


tqdm.pandas(desc="topic_interest2v...")
data_a['member_question_similarity']=data_a.progress_apply(lambda x:getsimilarity(x['uid'],x['qid']),axis=1)


# In[13]:


data_a.head()


fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for index, (train_idx, val_idx) in enumerate(fold.split(X=X_train_all, y=y_train_all)):
    X_train, X_val, y_train, y_val = X_trainStratifiedKFold_all.iloc[train_idx][feature_cols], X_train_all.iloc[val_idx][feature_cols],y_train_all.iloc[train_idx],y_train_all.iloc[val_idx]
    model_lgb.fit(X_train, y_train,
              eval_metric=['logloss', 'auc'],
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=10)


# In[17]:



# get_ipython().magic('pinfo model_lgb.fit')
# model_lgb.fit?



import pickle
with open('train/data.pkl','rb') as file:
    data_a=pickle.load(file)
data_a.shape


# In[5]:


with open('feature/member_feat_149.pkl','rb') as file:
    member_info=pickle.load(file)
member_info.shape


# In[6]:


pd.options.display.max_columns=None
member_info.head()


# In[7]:


columns=['author_id']
for i in range(64):
    columns.append('topic_vector_{}'.format(i))
member_topic=member_info[columns]


# In[8]:


member_topic.head()


# In[9]:


with open('feature/question_feat_134.pkl','rb') as file:
    question_info=pickle.load(file)
question_info.shape


# In[10]:


question_info.head()


# In[11]:


columns=['question_id']
for i in range(64):
    columns.append('q_topic_vector_{}'.format(i))
question_topic=question_info[columns]


# In[12]:


question_topic.shape


# In[13]:


data_a=pd.merge(data_a,member_topic,how='left',left_on='uid',right_on='author_id')


# In[14]:


data_a.shape


# In[15]:


data_a.head()


# In[16]:


data_a=data_a.drop(['author_id'],axis=1)


# In[17]:


data_a.shape


# In[18]:


data_a=pd.merge(data_a,question_topic,how='left',left_on='qid',right_on='question_id')


# In[19]:


data_a=data_a.drop(['question_id'],axis=1)


# In[20]:


data_a.shape


# In[59]:


with open('train/answer_author_question_vector.pkl','rb') as file:
    answer_author_question_vector=pickle.load(file)
answer_author_question_vector.head()


# In[60]:


answer_author_question_vector.shape


# In[61]:


answer_author_question_vector=answer_author_question_vector.drop(['q_topic_vector'],axis=1)
answer_author_question_vector.shape


# In[62]:


data_a=pd.merge(data_a,answer_author_question_vector,how='left',on='uid')


# In[63]:


data_a.shape


# In[65]:


data_a[data_a['uid']=='M1000000382']


# In[21]:


feature_cols = [x for x in data_a.columns if x not in ('label', 'uid', 'qid', 'dt', 'day')]
# target编码


# In[22]:


train_label=2593669
X_train_all = data_a.iloc[:train_label][feature_cols]
y_train_all = data_a.iloc[:train_label]['label']
test = data_a.iloc[train_label:]

logging.info("train shape %s, test shape %s", X_train_all.shape, test.shape)
print(X_train_all.shape,test.shape)


# In[23]:


test[feature_cols].shape


# In[24]:


data_a.head()


# In[25]:


model_lgb = LGBMClassifier(boosting_type='gbdt', num_leaves=64, learning_rate=0.01, n_estimators=2000,
                           max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                           min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                           colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=-1, silent=True)


# In[26]:


fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for index, (train_idx, val_idx) in enumerate(fold.split(X=X_train_all, y=y_train_all)):
    X_train, X_val, y_train, y_val = X_train_all.iloc[train_idx][feature_cols], X_train_all.iloc[val_idx][feature_cols],y_train_all.iloc[train_idx],y_train_all.iloc[val_idx]
    model_lgb.fit(X_train, y_train,
                  eval_metric=['logloss', 'auc'],
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=10)
sub = pd.read_csv(f'test/invite_info_evaluate_1_0926.txt', sep='\t', header=None)
sub.columns = ['qid', 'uid', 'dt']
sub['label'] = model_lgb.predict_proba(test[feature_cols])[:, 1]


sub.to_csv('result/result15.txt', index=None, header=None, sep='\t')


'''