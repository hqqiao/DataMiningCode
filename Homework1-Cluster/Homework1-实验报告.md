# Homework1 实验报告 #

----------

## 实验目的
1. 理解聚类算法的原理，学会应用。
2. 调用sklearn模块中的聚类函数以及评估函数对数据集进行聚类分析和效果评价。
## 实验环境
 windows10，python3.5，scikit-learn0.21.2
## 实验内容
1. 测试sklearn中以下聚类算法在sklearn.datasets.load_digits，sklearn.datasets.fetch_20newsgroups两个数据集上的聚类效果。
![](https://i.imgur.com/VvMWpvb.png)
2. 使用NMI(Normalized Mutual Information),Homogeneity,Completeness作为评价指标，评估以上聚类算法的聚类效果。
## 实验过程
1. 20 newsgroups数据集包括18846篇新闻文章，一共涉及到20种话题。sklearn提供了该数据的接口：sklearn.datasets.fetch_20newsgroups，首先加载数据集，使用TfidfVectorizer方法进行文本向量化与TF-IDF预处理。
```
vectorizer = TfidfVectorizer(max_df=0.5,            # 词汇表中过滤掉df>(0.5*doc_num)的单词
                             max_features=3000,     # 构建词汇表仅考虑max_features(按语料词频排序)
                             min_df=2,              # 词汇表中过滤掉df<2的单词
                             stop_words='english',  # 词汇表中过滤掉英文停用词
                             use_idf=True)          # 启动inverse-document-frequency重新计算权重

X = vectorizer.fit_transform(dataset.data)          # 稀疏矩阵
```


2. 