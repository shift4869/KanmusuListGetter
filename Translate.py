# coding: utf-8
import cv2
import os
import pandas as pd
import urllib
import requests


# 背景色分類から背景種類に変換する（誤判定修正も行う）
# THINK::クラスタサイズを7に決め打ちしている
def TranslateBGClusterToKind(data_set):
    df = pd.read_csv("translate.csv")
    nr = df[["艦名", "背景種類"]]

    for index, row in data_set.iterrows():
        name = data_set.at[index, "艦名"]
        old_v = int(data_set.at[index, "背景色分類"])
        new_v = -1

        if old_v == 0:
            new_v = 4
        elif old_v == 1:
            new_v = 2
        elif old_v == 2:
            new_v = 4
        elif old_v == 3:
            new_v = 1
        elif old_v == 4:
            new_v = 3

        rr = nr[nr['艦名'] == name]
        if len(rr) > 0:
            new_v = rr["背景種類"].iat[0]

        data_set.at[index, "背景種類"] = new_v

    return data_set[["背景種類"]]


if __name__ == "__main__":
    from Clustering import *
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # RGB値でクラスタリング
    df = pd.read_csv("result.csv")
    wdf = df[["R", "G", "B"]]
    k = GetClusterNum(wdf)
    model1 = KMeans(n_clusters=k, random_state=0)
    model1.fit(wdf)
    y1 = model1.labels_
    df["背景色分類"] = y1

    # 背景色分類から背景種類に変換する（誤判定修正も行う）
    wdf = df[["艦名", "背景色分類", "背景種類"]]
    y2 = TranslateBGClusterToKind(wdf)
    df["背景種類"] = y2

    print(y2)

    df.to_csv("result.csv", index=False, encoding="utf_8_sig")
