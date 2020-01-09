# coding: utf-8
import cv2
import os
import urllib
import requests


def GetBackGroundColor(img_url, d=20, new_load_flag=True, bg_save_path="./bg"):
    # 画像URLチェック
    if img_url == "":
        return []

    # 画像保存ディレクトリが既にあるか確認する
    # なければディレクトリ作成
    if not os.path.exists(bg_save_path):
        try:
            os.makedirs(bg_save_path)
        except Exception:
            return []

    # 画像ファイル名を取得する
    p = urllib.parse.urlparse(img_url).path
    img_name = os.path.split(p)[-1]
    target_img_name = os.path.join(bg_save_path, img_name)

    if new_load_flag or (not os.path.exists(target_img_name)):
        # 画像DLのためにユーザーエージェントを火狐に偽装
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0"
        }

        # 画像DL
        res = requests.get(img_url, headers=headers)
        res.raise_for_status()

        # 画像保存
        with open(target_img_name, mode="wb") as fout:
            fout.write(res.content)

    # 対象画像読み込み
    try:
        img = cv2.imread(target_img_name, cv2.IMREAD_COLOR)
    except Exception:
        exit(1)

    # サイズ取得
    height, width, channels = img.shape[:3]

    # 対象範囲を切り出し
    boxFromX = width - d  # 対象範囲開始位置 X座標
    boxFromY = 0  # 対象範囲開始位置 Y座標
    boxToX = width  # 対象範囲終了位置 X座標
    boxToY = d  # 対象範囲終了位置 Y座標

    # y:y+h, x:x+w　の順で設定
    imgBox = img[boxFromY: boxToY, boxFromX: boxToX]

    # RGB平均値を出力
    # flattenで一次元化しmeanで平均を取得
    r = imgBox.T[2].flatten().mean()
    g = imgBox.T[1].flatten().mean()
    b = imgBox.T[0].flatten().mean()

    # RGB平均値を出力
    # print("B: %.2f" % (b))
    # print("G: %.2f" % (g))
    # print("R: %.2f" % (r))

    # BGRからHSVに変換
    imgBoxHsv = cv2.cvtColor(imgBox, cv2.COLOR_BGR2HSV)

    # HSV平均値を取得
    # flattenで一次元化しmeanで平均を取得
    h = imgBoxHsv.T[0].flatten().mean()
    s = imgBoxHsv.T[1].flatten().mean()
    v = imgBoxHsv.T[2].flatten().mean()

    # HSV平均値を出力
    # uHeは[0,179], Saturationは[0,255]，Valueは[0,255]
    # print("Hue: %.2f" % (h))
    # print("Salute: %.2f" % (s))
    # print("Value: %.2f" % (v))

    return [r, g, b, h, s, v]


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    img_url = "https://cdn.wikiwiki.jp/to/w/kancolle/%E9%87%91%E5%89%9B/::ref/021v2.jpg?rev=a206314e015013d864908f63953e13fd&t=20130924094057"
    bg_col_info = GetBackGroundColor(img_url, new_load_flag=False)
    print(bg_col_info)
