# coding: utf-8
from bs4 import BeautifulSoup
import csv
import os
import pandas as pd
import re
import requests
from sklearn.cluster import KMeans
import traceback

from HtmlParser import *
from GetBackGroundColor import *
from Clustering import *
from Translate import *

WIKI_SOURCE_DL_FLAG = False  # Trueならばwikiページを新たに取得する
WIKI_SOURCE_CACHE = "target.html"  # 取得したwikiページソース（ローカル保存名）
BG_DL_FLAG = False  # Trueならば艦娘背景画像を新たに取得する
BG_SAVE_PATH = "./bg"  # 艦娘背景画像保存パス

if __name__ == "__main__":
    # カレントディレクトリをこのファイルがある場所に設定
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # wikiページを取得してBeautifulSoupのhtml解析準備を行う
    try:
        if WIKI_SOURCE_DL_FLAG:
            res = requests.get('https://wikiwiki.jp/kancolle/艦娘カード一覧2')
            if res.status_code != 200:
                print(f"request error : http status code = {res.status_code}")
                exit(1)

            with open(WIKI_SOURCE_CACHE, mode="w") as fout:
                fout.write(res.text)

        with open(WIKI_SOURCE_CACHE, mode="r") as fin:
            res = fin.read()

        soup = BeautifulSoup(res, "html.parser")
    except Exception:
        traceback.print_exc()
        exit(1)

    # 艦名のリストを取得する
    # ["No", "艦名", "艦種", "画像URL", "R", "G", "B"]
    name_list = GetNameList(soup, new_load_flag=BG_DL_FLAG, bg_save_path=BG_SAVE_PATH)

    # RGB値でクラスタリング
    df = pd.DataFrame(name_list, columns=["No", "艦名", "艦種", "画像URL", "R", "G", "B"])
    wdf = df[["R", "G", "B"]]
    k = GetClusterNum(wdf)
    model1 = KMeans(n_clusters=k, random_state=0)
    model1.fit(wdf)
    y1 = model1.labels_
    df["背景色分類"] = y1

    # 背景色分類から背景種類に変換する（誤判定修正も行う）
    df["背景種類"] = y1  # 仮に背景種類の列を作成しておく
    y2 = TranslateBGClusterToKind(df)
    df["背景種類"] = y2  # 変換後の値を反映

    df.to_csv("result.csv", index=False, encoding="utf_8_sig")
