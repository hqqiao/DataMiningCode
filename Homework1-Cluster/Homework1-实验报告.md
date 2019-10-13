# Homework1 实验报告-Clustering with sklearn #
姓名：黄巧 

学号：201934715

----------

## 实验目的
1、理解聚类算法的原理，学会应用。

2、调用sklearn模块中的聚类函数以及评估函数对数据集进行聚类分析和效果评价。
## 实验环境
 windows10，python3.5，scikit-learn0.19.0
## 实验内容
1、测试sklearn中以下聚类算法在sklearn.datasets.load_digits，sklearn.datasets.fetch_20newsgroups两个数据集上的聚类效果。

![](https://i.imgur.com/VvMWpvb.png)

2、使用Homogeneity,Completeness，NMI(Normalized Mutual Information)作为评价指标，评估以上聚类算法的聚类效果。
## 实验过程
### （一）数据预处理
1、fetch_20newsgroups数据集包括18846篇新闻文章，一共涉及到20种话题，这些文档在20 个不同的新闻组中几乎均匀分布。sklearn提供了该数据的接口：sklearn.datasets.fetch_20newsgroups。首先加载数据集，选取categories=['alt.atheism','talk.religion.misc','comp.graphics','sci.space','rec.autos']的类别进行聚类分析，使用TfidfVectorizer方法进行文本向量化与TF-IDF预处理，得到一个[n_samples,n_features]维度的稀疏矩阵。

```
vectorizer = TfidfVectorizer(max_df=0.5,            # 词汇表中过滤掉df>(0.5*doc_num)的单词
                             max_features=3000,     # 构建词汇表仅考虑max_features(按语料词频排序)
                             min_df=2,              # 词汇表中过滤掉df<2的单词
                             stop_words='english',  # 词汇表中过滤掉英文停用词
                             use_idf=True)          # 启动inverse-document-frequency重新计算权重
X = vectorizer.fit_transform(dataset.data)          # 稀疏矩阵
```

执行结果如下图所示：

![加载数据集](https://i.imgur.com/lZHnWC7.png)

2、load_digits数据集包括有1797个样本，每个样本包括8*8像素的图像和一个[0,9]整数的标签。

![digits数据集](https://i.imgur.com/a8x42FZ.png)

images是一个1797*8*8的三维矩阵，即有1797张8*8的数字图片组成。

![digits手写字图片](https://i.imgur.com/vU1ZyDt.png)

### （二）聚类算法调用与评估
1、分别调用 sklearn 中的八种聚类算法对 fetch_20newsgroups 数据进行聚类分析，算法的关键函数如下：

(1) KMeans

K均值聚类是以样本间距离为基础，将所有的观测样本划分到K个群体，使得群体和群体之间的距离尽量大，同时群体内部的样本之间的“距离和”最小。K均值是期望最大化算法的特殊情况，K均值是在每次迭代中只计算聚类分布的质心。

优点：

- KMeans算法具有实现简单，收敛速度快的优点。

缺点：

- 必须预先指定聚类的数目，在聚类数目未知时效果较差，而且有时候不稳定，陷入局部收敛。

`km = KMeans(n_clusters=true_k,init='k-means++',max_iter=100,n_init=1).fit(X)`

(2) Affinity propagation

AP聚类是通过在样本对之间发送消息直到收敛来创建聚类。然后使用少量示例样本作为聚类中心来描述数据集，聚类中心是数据集中最能代表一类数据的样本。在样本对之间发送的消息表示一个样本作为另一个样本的示例样本的适合程度，适合程度值在根据通信的反馈不断更新。更新迭代直到收敛，完成聚类中心的选取，因此也给出了最终聚类。

优点：

- AP聚类不需要指定K(经典的K-Means)或者是其他描述聚类个数(SOM中的网络结构和规模)的参数。

- 一个聚类中最具代表性的点在AP算法中叫做Examplar，与其他算法中的聚类中心不同，examplar是原始数据中确切存在的一个数据点，而不是由多个数据点求平均而得到的聚类中心(K-Means)。

- 多次执行AP聚类算法，得到的结果是完全一样的，即不需要进行随机选取初值步骤。

缺点：

- 算法复杂度较高，为O(N*N*logN)，而K-Means只是O(N*K)的复杂度。因此当N比较大时(N>3000)，AP聚类算法往往需要算很久。。

- 若以误差平方和来衡量算法间的优劣，AP聚类比其他方法的误差平方和都要低。(无论k-center clustering重复多少次，都达不到AP那么低的误差平方和)

`ap = AffinityPropagation(damping=0.5, preference=None).fit(X)`

(3) Mean-shift

MeanShift 算法旨在于发现一个样本密度平滑的blobs。均值漂移算法是基于质心的算法，通过更新质心的候选位置为所选定区域的偏移均值。然后，这些候选者在后处理阶段被过滤以消除近似重复，从而形成最终质心集合。即聚类中心是通过在给定区域中的样本均值确定的，通过不断更新聚类中心，直到聚类中心不再改变为止

优点：

- 不同于K-Means算法，均值漂移聚类算法不需要我们知道有多少类/组。

- 基于密度的算法相比于K-Means受均值影响较小。

缺点：

- 窗口半径r的选择可能是不重要的。

```
ms = MeanShift(bandwidth=0.9, bin_seeding=True)
X_array = X.toarray()
ms.fit(X_array)
```

(4) Spectral clustering

SpectralClustering 是在样本之间进行亲和力矩阵的低维度嵌入，其实是低维空间中的KMeans。如果亲和度矩阵稀疏，则这是非常有效的并且 SpectralClustering 需要指定聚类数。这个算法适用于聚类数少时，在聚类数多时不建议使用。它使用最近邻图来计算数据的高维表示，然后用k-means算法分配标签。

`sc = SpectralClustering(n_clusters=true_k).fit(X)`

(5) Ward hierarchical clustering

Hierarchical clustering 是一个常用的聚类算法，它通过不断的合并或者分割来构建聚类。聚类的层次被表示成树（或者 dendrogram（树形图））。树根是拥有所有样本的唯一聚类，叶子是仅有一个样本的聚类。

Ward 方法是一种质心算法。质心方法通过计算集群的质心之间的距离来计算两个簇的接近度。对于 Ward 方法来说，两个簇的接近度指的是当两个簇合并时产生的平方误差的增量。

优点：

- 能够展现数据层次结构，易于理解

- 可以基于层次事后再选择类的个数（根据数据选择类，但是数据量大，速度慢）

缺点：

- 计算量比较大，不适合样本量大的情形，较多用于宏观综合评价。

`ward = AgglomerativeClustering(n_clusters=true_k, linkage='ward').fit(X_array)`

(6) Agglomerative clustering

凝聚聚类对象使用自底向上的方法执行层次聚类:每个观察从其自己的集群中开始，集群依次合并在一起。链接标准决定了用于合并策略的度量:

`ac = AgglomerativeClustering(n_clusters=true_k, linkage='complete').fit(X_array)`

(7) DBSCAN

DBSCAN 算法将聚类视为被低密度区域分隔的高密度区域。DBSCAN 发现的聚类可以是任何形状的，与假设聚类是凸区域的 K-means 相反。 DBSCAN 的核心概念是 core samples,是指位于高密度区域的样本。因此一个聚类是一组核心样本，每个核心样本彼此靠近（通过一定距离度量测量）和一组接近核心样本的非核心样本（但本身不是核心样本）。算法中的两个参数, min_samples 和 eps ,正式的定义了 dense （稠密）。较高的 min_samples 或者较低的eps表示形成聚类所需的较高密度。

优点：不需要知道簇的数量；对噪声不敏感；能发现任意形状的聚类。

缺点：

- 需要确定距离r和minPoints；

- 聚类的结果与参数有很大的关系；DBSCAN用固定参数识别聚类，但当聚类的稀疏程度不同时，相同的判定标准可能会破坏聚类的自然结构，即较稀的聚类会被划分为多个类或密度较大且离得较近的类会被合并成一个聚类。

`dbscan = DBSCAN(min_samples=1,metric='cosine').fit(X)`

(8) Gaussian mixtures

高斯混合模型是一种概率模型，它假定所有的数据点都是由有限个未知参数的高斯分布的混合产生的。可以把混合模型看作是对KMeans的一般化，它包含了关于数据的协方差结构以及潜在高斯分布中心的信息。

`gm = GaussianMixture(n_components=50).fit(X_array)`

2、利用sklearn 自带的 Homogeneity、Completeness、NMI 评估函数对每个聚类算法进行效果评价，得到评估分数。以KMeans为例，评估函数调用如下所示，分别得到运行时间、同质性Homogeneity、完整性Completeness、归一化NMI的评估分数：
```
print("done in %0.3fs" % (time() - t0))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("Normalized Mutual Information (NMI): %0.3f" % metrics.normalized_mutual_info_score(labels, km.labels_))
```
(1)同质性Homogeneity、完整性Completeness

同质性Homogeneity：每个群集只包含单个类的成员；

完整性Completeness：给定类的所有成员都分配给同一个群集；

数学公式如下图所示：
![同质性和完整性数学公式](https://i.imgur.com/5Prqud1.png)

优点：

- 分数明确：从0到1反应出最差到最优的表现；

- 解释直观：具有不良 v-measure 的聚类可以在同质性和完整性方面进行定性分析，以更好地感知到聚类的错误类型；

- 对簇结构不作假设：可以比较两种聚类算法如k均值算法和谱聚类算法的结果。

缺点：

- 随着样本数量、簇的数量以及标定过的真实标签的不同，完全随机的标签并不总是产生相同数值的同质性、完整性和 v-measure（随机标记不会产生零分，特别是当簇的数量大时）；
当样本数量超过 1000，簇的数量小于 10 时，可以忽略上述缺点，但对于较小的样本数量或者较大数量的簇，还是建议使用调整过的度量标准，比如 Adjusted Rand Index (ARI)；
- 基于互信息的度量方式，由于需要正确聚类标签，在实践中几乎不可用；

(2)NMI(Normalized Mutual Information)
数学公式如下图所示：

![归一化互信息](https://i.imgur.com/VsCLBkz.png)

优点：

- 对于随机的标签分配，AMI 趋近于 0（而 MI 就不能保证获得接近 0 的值）；

- MI和NMI的取值范围是[0,1]，值越大意味着聚类结果与真实情况越吻合，接近0的值代表两列聚类标签相对独立，接近1的值代表两列聚类标签具有一致性，0代表两列聚类标签完全独立，1代表两列聚类标签完全相同（AMI 的取值范围为[-1,1]）；

- 基于互信息的度量方式对于簇的结构没有作出任何假设，例如，可以用于比较K-Means与谱聚类的结果；

缺点：

- 基于互信息的度量方式，由于需要正确聚类标签，在实践中几乎不可用，但是可以用来在无监督的环境下，比较各种聚类算法结果的一致性（基于互信息的三种度量方式是对称的）；

## 实验结果与分析
1、在fetch20newsgroups数据集上八种聚类算法的效果评估如下图所示：

![新闻数据集运行结果及数据对比](https://i.imgur.com/Sy376B8.png)

效果评估图中用红色标注了效果最好的指标值，橙色次之；用蓝色标注了效果最差的指标值，绿色次之。

- 从运行时间指标看，DBSCAN算法运行时间仅1.37s最快，K-means算法2.57s以及Spectral clustering算法5.53s略慢一些，而 Gaussian mixtures 高达465.87s非常慢，效率较低。
- 从Homogeneity指标看，DBSCAN算法0.855表现最好，AffinityPropagation算法0.829也较好；而Agglomerative clustering算法0.014、Spectral clustering算法0.089均未超过0.02，表现最差；其他几个算法集中在0.5左右较为一般。
- 从Completeness指标看，K-means算法0.621表现最好，其次是Ward hierarchical clustering算法0.470次之；而Agglomerative clustering算法0.016表现最差；其他几个算法集中在0.2/0.3左右，较为一般。
- 从NMI指标看，K-means算法0.585表现最好，DBSCAN算法0.437次之；Agglomerative clustering算法0.015表现最差，Spectral clustering算法也相对较差；其他几个算法集中在0.32~0.41之间，表现较为一般。

三项指标综合来看：

Aggiomerative clustering 算法三项评估指标均低于0.02，运行时间也在70s以上，整体效果最差；

Gaussian mixtures的运行时间高达465s，各个指标值也相对一般，整体运行效果相对较差；

Affinity propagation 和 DBSCAN 算法运行效果相对较好，其中DBSCAN算法运行时间也相对较少，表现最好。

2、在digits数据集上八种聚类算法的效果评估如下图所示：

![digits数据集运行结果以及数据对比](https://i.imgur.com/vclAXxb.png)

其中K-means算法使用PCA降维后绘制的聚类图示如下：

![digits数据集K-means降维聚类图示](https://i.imgur.com/hwtuDvr.png)

效果评估图中用红色标注了效果最好的指标值，橙色次之；用蓝色标注了效果最差的指标值，绿色次之。

- 从运行时间指标看，DBSCAN算法运行时间仅0.05s最快，Spectral clustering算法418s最慢，AffinityPropagation算法4.71s相对较慢，其他算法都在0.1-1s之间，从运行时间角度相对较好。
- 从Homogeneity指标看，MeanShift算法可以达到1.0表现最好，AffinityPropagation算法0.932也较好；而DBSCAN算法仅有0.001表现最差，Agglomerative clustering算法0.017、Spectral clustering算法0.019均未超过0.02，表现也较差；其他几个算法集中在0.7左右相对较好。
- 从Completeness指标看，Ward hierarchical clustering算法0.836表现最好，K-means算法0.650表现次之；而Agglomerative clustering算法0.249表现最差，DBSCAN以及Spectral clustering算法也都处于0.3以下整体较差；其他几个算法集中在0.5左右，相对一般。
- 从NMI指标看，Ward hierarchical clustering算法0.797表现最好，GaussianMixture算法0.689次之；DBSCAN算法仅有0.017表现最差，Agglomerative clustering算法0.065以及Spectral clustering算法0.076也相对较差；其他几个算法集中在0.6左右，相对较好。

三项指标综合来看：

SpectralClustering算法三项评估指标均较差且运行时间极长，整体效果最差；

DBSCAN算法和Aggiomerative clustering 算法三项评估指标也均较差，效果也很不好；

Gaussian mixtures算法、Ward hierarchical clustering算法以及AffinityPropagation算法在三项指标的综合表现最好；

MeanShift算法在Homogeneity表现极佳但在Completeness以及NMI指标上不是很出色；

K-means算法三项指标都在0.6左右比较平均。

3、

综合两个数据集运行结果来看：

在Homogeneity方面：

AffinityPropagation算法在两个数据集上表现均较好，都能达到0.9左右；而Agglomerative clustering算法以及Spectral clustering算法在两个数据集上表现都比较差，均未超过0.02，这两个算法在同质性方面效果较差。

在Completeness方面：

K-means算法以及Ward hierarchical clustering算法在两个数据集上表现均较好；而Agglomerative clustering算法一直表现较差。

在NMI方面：

Agglomerative clustering算法一直表现较差，其他算法各有波动。

