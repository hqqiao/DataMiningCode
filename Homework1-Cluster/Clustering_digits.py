# -*- coding: utf-8 -*-
"""
@author: huangqiao
@file: Clustering_digits
@time: 2019/10/8 19:21
"""
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import *
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.mixture import GaussianMixture

# numpy.random.seed()函数可使得随机数具有预见性，即当参数相同时使得每次生成的随机数相同；
# 当参数不同或者无参数时，作用与numpy.random.rand()函数相同，即多次生成随机数且每次生成的随机数都不同。
np.random.seed(42)

digits = load_digits()
data = scale(digits.data)
# 获取digits数据集的属性
print("digits.keys=", digits.keys())
# 获取digits数据集的target
print('digits.target = ',digits.target)

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target
sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))

#imgaes 是一个三维矩阵1797 张 8 * 8的图片
print('digits.images.shape = ',digits.images.shape)
print('digits.images = ',digits.images)
plt.gray()
plt.matshow(digits.images[0])
plt.show()



# 输出分隔符以及列表表头
print(82 * '_')
print('init\t\t\t\t\t\t\t\ttime\t\tHomogeneity\t\tCompleteness\tNMI')

def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-30s\t\t%.2fs\t\t%.3f\t\t\t%.3f\t\t\t%.3f'
          % (name,
             (time() - t0),
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.normalized_mutual_info_score(labels, estimator.labels_)))

def bench_GaussianMixture(name, data):
    t0 = time()
    gm = GaussianMixture(n_components=50).fit(data)
    print('%-30s\t\t%.2fs\t\t%.3f\t\t\t%.3f\t\t\t%.3f'
          % (name,
             (time() - t0),
             metrics.homogeneity_score(labels, gm.predict(data)),
             metrics.completeness_score(labels, gm.predict(data)),
             metrics.normalized_mutual_info_score(labels, gm.predict(data))))

bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)

bench_k_means(AffinityPropagation(damping=0.5, preference=None),
              name="AffinityPropagation",data=data)

bench_k_means(MeanShift(bandwidth=0.9, bin_seeding=True),
              name="MeanShift",data=data)

bench_k_means(SpectralClustering(n_clusters=n_digits-4),
              name="SpectralClustering",data=data)

bench_k_means(AgglomerativeClustering(n_clusters=n_digits, linkage='ward'),
              name="Ward hierachical clustering",data=data)

bench_k_means(AgglomerativeClustering(n_clusters=n_digits, linkage='complete'),
              name="AgglomerativeClustering",data=data)

bench_k_means(DBSCAN(min_samples=1,metric='cosine'),
              name="DBSCAN",data=data)

bench_GaussianMixture(name="GaussianMixture",data=data)




# in this case the seeding of the centers is deterministic, hence we run the kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
              name="PCA-based",
              data=data)
print(82 * '_')

# #############################################################################
# Visualize the results on PCA-reduced data 在pca减少的数据上可视化结果

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
# 步长网格。降低以提高VQ的质量
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
# 绘制决策边界。为此，我们将为每个对象分配一个颜色
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
# 获取网格中每个点的标签。使用最后一次训练的模型
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot 将结果放入颜色图中
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X 把中心体画成白色的X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


