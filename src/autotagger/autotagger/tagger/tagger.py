from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
try:
    from ..exceptions import ModelNotTrainedError
except:
    ParamatersError = ValueError

class AutoTagger(object):
    """
    根据用户指定的算法对数据进行聚类。
 
    参数:
    - data : array-like, shape (n_samples, n_features)
        要聚类的数据。
    - algorithm : str or class, optional (default='kmeans')
        聚类算法的名称或sklearn聚类器类。
    - kwargs : dict
        传递给聚类算法的其他参数。

    本类型在实例化时就会输出原数据聚类结果
 
    """
    def __init__(self, data, algorithm='kmeans', n_clusters=None, **kwargs):
        if algorithm == 'kmeans':
            if n_clusters is None:
                raise ParamatersError('参数n_clusters必须被指定。')
            model = KMeans(n_clusters=n_clusters, **kwargs)
        else:
            raise ParamatersError('没有指定算法。')
        
        self.data = pd.DataFrame(data)
        labels = model.fit_predict(data)
        self.data['Cluster_Label'] = labels
        self.n_tags = n_clusters
        examples = model.cluster_centers_

        print("\nCluster examples:")
        for i, example in enumerate(examples):
            print(f"Cluster {i}: {example}")

    def __call__(self, tags):
        '''
        - tags : array-like 
          标签名
        '''
        if len(tags) != self.n_tags:
            raise ParamatersError('标签长度与类别不符合')
        
        self.data['Mapped_Label'] = self.data['Cluster_Label'].map({i: tags[i] for i in range(self.n_tags)})
        self.data.to_csv('result.csv', index=False)

        return
    


def main():
    raise RuntimeError('该方法仅作调试用，该脚本不可调用。')
    data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0) 
    n_clusters = 4
    label_names = ['Group_A', 'Group_B', 'Group_C', 'Group_D']
    test = AutoTagger(data, n_clusters=4)
    test(label_names)

    return

if __name__ == '__main__':
    main()

        

        