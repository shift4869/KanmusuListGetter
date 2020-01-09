# coding: utf-8
import os
import pandas as pd
import traceback

TRANSLATE_MST_FILE = "translate_mst.csv"


# 背景色分類から背景種類に変換する（誤判定修正も行う）
def TranslateBGClusterToKind(data_set):
    # 例外カード置き換えマスタ
    df = pd.read_csv(TRANSLATE_MST_FILE)
    nr = df[["艦名", "背景種類"]]

    # 背景色分類（クラスタ番号）は毎回ランダムなので
    # それぞれのクラスタから代表の番号を採用する
    try:
        def GetBGClusterNum(name):
            return int(data_set[data_set["艦名"] == name]["背景色分類"].iat[0])

        holo1 = GetBGClusterNum("金剛改二")  # ピンクっぽいホロ背景
        holo2 = GetBGClusterNum("Littorio")  # 白っぽいホロ背景
        gold = GetBGClusterNum("金剛")  # 金背景
        silver = GetBGClusterNum("飛鷹")  # 銀背景
        common = GetBGClusterNum("鳳翔")  # コモン（水色）背景
    except Exception:
        traceback.print_exc()
        exit(1)

    # -1で初期化
    data_set["背景種類"] = -1
    for index, row in data_set.iterrows():
        name = data_set.at[index, "艦名"]
        old_v = int(data_set.at[index, "背景色分類"])
        new_v = -1

        if old_v in [holo1, holo2]:
            new_v = 4  # ホロ背景
        elif old_v == gold:
            new_v = 3  # 金背景
        elif old_v == silver:
            new_v = 2  # 銀背景
        elif old_v == common:
            new_v = 1  # コモン（水色）背景

        # 例外カードの場合はマスタに従って置き換え
        rr = nr[nr["艦名"] == name]
        if len(rr) > 0:
            new_v = rr["背景種類"].iat[0]

        if new_v == -1:
            print(f"背景種類判別エラー:{name}")
            # exit(1)

        data_set.at[index, "背景種類"] = new_v

    return data_set[["背景種類"]]


if __name__ == "__main__":
    from Clustering import *
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    RESULT_FILE = "result.csv"  # 出力ファイル名

    # RGB値でクラスタリング
    df = pd.read_csv(RESULT_FILE)
    y1 = GetClusterLabel(df)
    df["背景色分類"] = y1

    # 背景色分類から背景種類に変換する（誤判定修正も行う）
    y2 = TranslateBGClusterToKind(df)
    df["背景種類"] = y2  # 変換後の値を反映

    print(y2)

    df.to_csv(RESULT_FILE, index=False, encoding="utf_8_sig")
