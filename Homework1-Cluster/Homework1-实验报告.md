# Homework1 实验报告-Clustering with sklearn #
姓名：黄巧 

学号：201934715

----------

## 实验目的
1、理解聚类算法的原理，学会应用。

2、调用sklearn模块中的聚类函数以及评估函数对数据集进行聚类分析和效果评价。
## 实验环境
 windows10，python3.5，scikit-learn0.21.2
## 实验内容
1、测试sklearn中以下聚类算法在sklearn.datasets.load_digits，sklearn.datasets.fetch_20newsgroups两个数据集上的聚类效果。

![](https://i.imgur.com/VvMWpvb.png)

2、使用Homogeneity,Completeness，NMI(Normalized Mutual Information)作为评价指标，评估以上聚类算法的聚类效果。
## 实验过程
### （一）数据预处理
1、fetch_20newsgroups数据集包括18846篇新闻文章，一共涉及到20种话题。sklearn提供了该数据的接口：sklearn.datasets.fetch_20newsgroups，首先加载数据集，选取categories=['alt.atheism','talk.religion.misc','comp.graphics','sci.space','rec.autos']的类别进行聚类分析，使用TfidfVectorizer方法进行文本向量化与TF-IDF预处理，得到一个[n_samples,n_features]维度的稀疏矩阵。

```
vectorizer = TfidfVectorizer(max_df=0.5,            # 词汇表中过滤掉df>(0.5*doc_num)的单词
                             max_features=3000,     # 构建词汇表仅考虑max_features(按语料词频排序)
                             min_df=2,              # 词汇表中过滤掉df<2的单词
                             stop_words='english',  # 词汇表中过滤掉英文停用词
                             use_idf=True)          # 启动inverse-document-frequency重新计算权重
X = vectorizer.fit_transform(dataset.data)          # 稀疏矩阵
```

执行结果如下图所示：

![](https://i.imgur.com/FBH5ma9.png)

### （二）聚类算法调用与评估
1、分别调用 sklearn 中的八种聚类算法对 fetch_20newsgroups 数据进行聚类分析，算法的关键函数如下：

- KMeans

`km = KMeans(n_clusters=true_k,init='k-means++',max_iter=100,n_init=1).fit(X)`

- Affinity propagation

`ap = AffinityPropagation(damping=0.5, preference=None).fit(X)`

- Mean-shift

```
ms = MeanShift(bandwidth=0.9, bin_seeding=True)
X_array = X.toarray()
ms.fit(X_array)
```

- Spectral clustering

`sc = SpectralClustering(n_clusters=true_k).fit(X)`

- Ward hierarchical clustering

`ward = AgglomerativeClustering(n_clusters=true_k, linkage='ward').fit(X_array)`

- Aggiomerative clustering

`ac = AgglomerativeClustering(n_clusters=true_k, linkage='complete').fit(X_array)`

- DBSCAN

`dbscan = DBSCAN(min_samples=1,metric='cosine').fit(X)`

- Gaussian mixtures

`gm = GaussianMixture(n_components=50).fit(X_array)`

2、利用sklearn 自带的 Homogeneity、completeness、NMI 评估函数对每个聚类算法进行效果评价，得到评估分数。以KMeans为例，评估函数调用如下所示，分别得到运行时间、同质性、完整性、归一化的评估分数：
```
print("done in %0.3fs" % (time() - t0))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("Normalized Mutual Information (NMI): %0.3f" % metrics.normalized_mutual_info_score(labels, km.labels_))
```
## 实验结果
1、在fetch20newsgroups数据集上八种聚类算法的效果评估如下表所示：

![运行结果数据对比](https://i.imgur.com/FBdNIQ2.png)
运行结果截图为：
![运行结果1](https://i.imgur.com/iwZvLY7.png)
![运行结果2](https://i.imgur.com/WDHBSK2.png)
![运行结果3](https://i.imgur.com/2AwDa6U.png)

