# coding: utf-8
import cv2
import os
import pandas as pd
import urllib
import requests

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


# 適切なクラスタ数を求める
# シルエット値平均が最大となクラスタ数を採用
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


def GetClusterLabel(target_file_name):
    # RGB値でクラスタリング
    df = pd.read_csv("result.csv")
    wdf = df[["R", "G", "B"]]

    cluster_num = GetClusterNum(wdf)

    model1 = KMeans(n_clusters=cluster_num, random_state=0)
    model1.fit(wdf)
    y1 = model1.labels_
    # df["背景色分類"] = y1

    return y1


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    target_file_name = "result.csv"
    y1 = GetClusterLabel(target_file_name)
    print(y1)
