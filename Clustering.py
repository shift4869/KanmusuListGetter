# coding: utf-8
import os
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

USE_EXCEPT_CLUSTER_NUM_FLAG = True
EXCEPT_CLUSTER_NUM = 7


# 適切なクラスタ数を求める
# シルエット値平均が最大となるクラスタ数を採用
def GetClusterNum(data_set):
    k_range = range(2, 9)

    most_suit_sa_k = [-1, -1]
    for k in k_range:
        clusterer = KMeans(n_clusters=k, random_state=0)
        cluster_labels = clusterer.fit_predict(data_set)
        kmeans = clusterer.fit(data_set)

        # SSE（クラスター内誤差の平方和）
        sse = kmeans.inertia_

        # シルエット値（-1～1）の平均
        silhouette_avg = silhouette_score(data_set, cluster_labels)
        if(most_suit_sa_k[0] < silhouette_avg):
            most_suit_sa_k[0] = silhouette_avg
            most_suit_sa_k[1] = k

        # print('For k =', k,
        #       'sse is :', sse,
        #       'The average silhouette_score is :', silhouette_avg)

    return most_suit_sa_k[1]


def GetClusterLabel(data_set):
    # RGB値でクラスタリング
    df = data_set[["R", "G", "B"]]

    cluster_num = GetClusterNum(df)

    if cluster_num != EXCEPT_CLUSTER_NUM:
        if USE_EXCEPT_CLUSTER_NUM_FLAG:
            cluster_num = EXCEPT_CLUSTER_NUM
        else:
            raise f"cluster_num is different from EXCEPT_CLUSTER_NUM:{cluster_num}"

    model1 = KMeans(n_clusters=cluster_num, random_state=0)
    model1.fit(df)
    y1 = model1.labels_

    return y1


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    target_file_name = "result.csv"
    df = pd.read_csv(target_file_name)
    y1 = GetClusterLabel(df)
    print(y1)
