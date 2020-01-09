# coding: utf-8
from bs4 import BeautifulSoup
import os
import pandas as pd
import requests
import traceback

from HtmlParse import *
from Clustering import *
from Translate import *

WIKI_SOURCE_DL_FLAG = True  # Trueならばwikiページを新たに取得する
WIKI_SOURCE_CACHE_FILE = "target.html"  # 取得したwikiページソースファイル名（ローカル保存名）
BG_DL_FLAG = True  # Trueならば艦娘背景画像を新たに取得する
BG_SAVE_PATH = "./bg"  # 艦娘背景画像の保存先ディレクトリパス
RESULT_FILE = "result.csv"  # 出力ファイル名

if __name__ == "__main__":
    # カレントディレクトリをこのファイルがある場所に設定
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # wikiページを取得してBeautifulSoupのhtml解析準備を行う
    try:
        if WIKI_SOURCE_DL_FLAG:
            res = requests.get('https://wikiwiki.jp/kancolle/艦娘カード一覧2')
            res.raise_for_status()

            with open(WIKI_SOURCE_CACHE_FILE, mode="w") as fout:
                fout.write(res.text)

        with open(WIKI_SOURCE_CACHE_FILE, mode="r") as fin:
            res = fin.read()

        soup = BeautifulSoup(res, "html.parser")
    except Exception:
        traceback.print_exc()
        exit(1)

    # 艦名のリストを取得する
    # ["No", "艦名", "艦種", "画像URL", "R", "G", "B"]
    name_list = GetNameList(soup, new_load_flag=BG_DL_FLAG, bg_save_path=BG_SAVE_PATH)
    if not name_list:
        print("艦名リスト取得失敗")
        exit(1)

    # RGB値でクラスタリング
    df = pd.DataFrame(name_list, columns=["No", "艦名", "艦種", "画像URL", "R", "G", "B"])
    y1 = GetClusterLabel(df)
    df["背景色分類"] = y1

    # 背景色分類から背景種類に変換する（誤判定修正も行う）
    y2 = TranslateBGClusterToKind(df)
    df["背景種類"] = y2  # 変換後の値を反映

    df.to_csv(RESULT_FILE, index=False, encoding="utf_8_sig")
