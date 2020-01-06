# coding: utf-8
from bs4 import BeautifulSoup
import csv
import os
import pandas as pd
import re
import requests
from sklearn.cluster import KMeans

from GetBackGroundColor import *
from Clustering import *
from Translate import *

DEBUG_FLAG = False


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # print(os.getcwd())

    if DEBUG_FLAG:
        res = requests.get('https://wikiwiki.jp/kancolle/艦娘カード一覧2')
        if res.status_code != 200:
            print("request error")
            exit(1)

        with open("target.html", mode="w") as fout:
            fout.write(res.text)

    if not os.path.isfile("result.csv"):
        with open("target.html", mode="r") as fin:
            res = fin.read()

        soup = BeautifulSoup(res, "html.parser")
        # print(soup.find(id="body").find_all("td", class_="style_td"))

        # 艦種を取得する
        kind_list = []
        for i in range(0, 100):
            h2tag = soup.find(id="body").find("h2", id=f"h2_content_1_{i}")
            if h2tag is None:
                break

            kind = h2tag.text.strip()
            kind_list.append(kind)

        # 艦名を艦種に紐づけながら取得する
        name_list = []
        kind_index = 0
        i_flag = False
        exception_index = [15, 16]
        div_ie5s = soup.find(id="body").find_all("div", class_="ie5")
        for div_ie5 in div_ie5s:
            trs = div_ie5.table.tbody.find_all("tr")
            for tr in trs:
                tds = tr.find_all("td")
                for td in tds:
                    name = td.text.strip()
                    if name == "":
                        continue

                    a_s = td.find_all("a")
                    img_url = ""
                    for a in a_s:
                        if a.get("href") is not None:
                            img_url = a.img["src"]
                            break

                    bg_color_info = GetBackGroundColor(img_url)

                    t = [len(name_list) + 1, name, kind_list[kind_index], img_url,
                         bg_color_info[0], bg_color_info[1], bg_color_info[2]]
                    # print(t)
                    name_list.append(t)

            # 一部テーブル構造がおかしいのでindexを調整する
            if kind_index in exception_index and (not i_flag):
                i_flag = True
            else:
                kind_index = kind_index + 1
                i_flag = False

        # 艦名に含まれる「No.1」などの表記を削除する
        for i in range(0, len(name_list)):
            record = name_list[i]
            record[1] = re.sub("No\.[0-9]* ", "", record[1]).strip()
            name_list[i] = record

        # csvとして保存
        df = pd.DataFrame(name_list, columns=["No", "艦名", "艦種", "画像URL", "R", "G", "B"])
        df.to_csv("result.csv", index=False, encoding="utf_8_sig")
        # with open("result.csv", "w", newline="") as fout:
        #     writer = csv.writer(fout)
        #     writer.writerow(["No", "艦名", "艦種", "画像URL", "R", "G", "B"])
        #     writer.writerows(name_list)

    # RGB値でクラスタリング
    df = pd.read_csv("result.csv")
    wdf = df[["R", "G", "B"]]
    k = GetClusterNum(wdf)
    model1 = KMeans(n_clusters=k, random_state=0)
    model1.fit(wdf)
    y1 = model1.labels_
    df["背景色分類"] = y1

    # 背景色分類から背景種類に変換する（誤判定修正も行う）
    y2 = TranslateBGClusterToKind(df[["艦名", "背景色分類"]])

    df.to_csv("result.csv", index=False, encoding="utf_8_sig")
