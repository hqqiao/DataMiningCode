# -*- coding: utf-8 -*-
"""
@author: huangqiao
@time: 2019/10/7 10:56
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from time import time
import numpy as np

# #############################################################################
print("Loading 20 newsgroups dataset for categories:")

'''
fetch_20newsgroups(data_home=None, # 文件下载的路径
                   subset='train', # 加载那一部分数据集 train/test
                   categories=None, # 选取哪一类数据集[类别列表]，默认20类
                   shuffle=True,    # 将数据集随机排序
                   random_state=42, # 随机数生成器
                   remove=(), # ('headers','footers','quotes') 去除部分文本
                   download_if_missing=True # 如果没有下载过，重新下载
                   )
'''
# 加载数据集
dataset = fetch_20newsgroups(subset='all',      # 加载哪一部分数据集 train/test
                             categories=None,   # 选取哪一类数据集[类别列表]，默认20类
                             shuffle=True,      # 将数据集随机排序
                             random_state=42,   # 随机数生成器
                             )
print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
print()

labels = dataset.target              # 原始数据标签类别
true_k = np.unique(labels).shape[0]  # 种类数

'''

机器学习中，我们总是要先将源数据处理成符合模型算法输入的形式，比如将文字、声音、图像转化成矩阵。
对于文本数据首先要进行分词（tokenization），移除停止词（stop words），然后将词语转化成矩阵形式，然后再输入机器学习模型中，
这个过程称为特征提取（feature extraction）或者向量化（vectorization）

在scikit-learn中，对文本数据进行特征提取，其实就是将文本数据转换为计算机能够处理的数字形式。
Scikit-learning提供了三种向量化的方法，分别是：

CountVectorizer：用于将文本转换为词项数量的向量
    CountVectorizer简单地对文档分分词，形成词汇表，然后对新文档就可以使用这个词汇表进行编码，
    最终将会返回一个长度等于词汇表长度的向量，每个数字表示单词在文档中出现的次数。
    由于这个向量中会有很多0，python中可使用scipy.sparse包提供的稀疏向量表示
HashingVectorizer：用于将文本转换为Hash值构成的向量
    上面介绍的基于次数和频率的方法比较耿直，其限制性在于生成的词汇表可能会非常庞大，
    导致最后生成的向量会很长，对内存需求会很大，最后就会降低算法效率。
    一个巧妙的方法是使用哈希方法，将文本转化为数字。这种方法不需要词汇表，你可以使用任意长度的向量来表示，
    但这种方法不可逆，不能再转化成对应的单词，不过很多监督学习任务并不care。
TfidfVectorizer：用于将文本转换为TF-IDF值构成的向量,将原始文档集合转换为TF-IDF特性矩阵。
    在CountVectorizer方法中，我们仅仅是统计了词汇的出现次数，比如单词the会出现比较多的次数，
    但实际上，这个单词并不是很有实际意义。因此使用TF-IDF方法来进行向量化操作。
    TF-IDF原本是一种统计方法，用以评估字词对于一个文件集或一个语料库中的其中一份文件的重要程度。
    这个方法认为，字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降，
    其实也就相当于在CountVectorizer的基础上结合整个语料库考虑单词的权重，并不是说这个单词出现次数越多它就越重要。
    使用TF-IDF并标准化以后，我们就可以使用各个文本的词特征向量作为文本的特征，进行分类或者聚类分析。

'''
print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
# 文本向量化与TF-IDF预处理
# Extracting features from the dataset, using a sparse vectorizer
# Vectorizer results are normalized
vectorizer = TfidfVectorizer(max_df=0.5,            # 词汇表中过滤掉df>(0.5*doc_num)的单词
                             max_features=3000,     # 构建词汇表仅考虑max_features(按语料词频排序)
                             min_df=2,              # 词汇表中过滤掉df<2的单词
                             stop_words='english',  # 词汇表中过滤掉英文停用词
                             use_idf=True)          # 启动inverse-document-frequency重新计算权重
X = vectorizer.fit_transform(dataset.data)          # 稀疏矩阵

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

# #############################################################################
# Do the actual clustering
# Evaluation—Homogeneity、Completeness、Normalized Mutual Information(NMI)
# 使用Homogeneity、completeness、NMI指标进行聚类分析评估

print("Clustering sparse data with KMeans")
km = KMeans(n_clusters=true_k,  # 形成的簇数以及生成的中心数
             init='k-means++',   # 用智能的方式选择初始聚类中心以加速收敛
             max_iter=100,       # 一次单独运行的最大迭代次数
             n_init=1            # 随机初始化的次数
             )
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("Normalized Mutual Information (NMI): %0.3f" % metrics.normalized_mutual_info_score(labels, km.labels_))
print()


print("Clustering sparse data with MiniBatchKMeans")
km2 = MiniBatchKMeans(n_clusters=true_k,  # 形成的簇数以及生成的中心数
                      init='k-means++',   # 用智能的方式选择初始聚类中心以加速收敛
                      n_init=1,           # 随机初始化的次数
                      init_size=1000,
                      batch_size=1000     # Size of the mini batches
                      )
t0 = time()
km2.fit(X)
print("done in %0.3fs" % (time() - t0))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km2.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km2.labels_))
print("Normalized Mutual Information (NMI): %0.3f" % metrics.normalized_mutual_info_score(labels, km2.labels_))
print()