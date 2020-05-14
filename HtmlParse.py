# coding: utf-8
from bs4 import BeautifulSoup
import os
import re
from tqdm import tqdm  # プログレスバー

from GetBackGroundColor import *


# html構造を解析して艦名、艦種及び背景色を画像解析したRGB値を取得する
# ["No", "艦名", "艦種", "画像URL", "R", "G", "B"]を要素とするリストを返す
# html構造がうまく取得できなかった場合は空リストを返す
def GetNameList(soup, new_load_flag=True, bg_save_path="./bg"):
    # 艦種を取得する
    kind_list = []
    for i in range(0, 128):
        h2tag = soup.find(id="body").find("h2", id=f"h2_content_1_{i}")
        if h2tag is None:
            break

        kind = h2tag.text.strip()
        kind_list.append(kind)

    # 艦種が一つも取得できなかった場合はエラー
    if not kind_list:
        return []

    # 艦名を艦種に紐づけながら取得する
    name_list = []
    kind_index = 0
    i_flag = False
    exception_index = [16, 17]
    # html構造解析
    div_ie5s = soup.find(id="body").find_all("div", class_="h-scrollable")
    for div_ie5 in tqdm(div_ie5s):
        trs = div_ie5.table.tbody.find_all("tr")
        for tr in trs:
            tds = tr.find_all("td")
            for td in tds:
                # 艦名取得（tdタグのテキスト）
                name = td.text.strip()
                if name == "":
                    continue

                # 艦娘カード画像URL取得（リンクが貼られているimgタグのsrc）
                a_s = td.find_all("a")
                img_url = ""
                for a in a_s:
                    if a.get("href") is not None:
                        img_url = a.img["src"]
                        break
                if img_url == "":
                    continue

                # 艦娘カードの背景を画像解析してRGB平均を取得
                bg_color_info = GetBackGroundColor(img_url, new_load_flag=new_load_flag, bg_save_path=bg_save_path)
                # 画像解析に失敗した場合はエラー
                if not bg_color_info:
                    return []

                # レコード追加
                # ["No", "艦名", "艦種", "画像URL", "R", "G", "B"]
                t = [len(name_list) + 1, name, kind_list[kind_index], img_url,
                     bg_color_info[0], bg_color_info[1], bg_color_info[2]]
                name_list.append(t)

        # 一部テーブル構造がおかしいのでindexを調整する
        if kind_index in exception_index and (not i_flag):
            i_flag = True
        else:
            kind_index = kind_index + 1
            i_flag = False

    # 艦名に含まれる「No.1」などの表記を削除する
    for i, record in enumerate(name_list):
        record[1] = re.sub("No\.[0-9]* ", "", record[1]).strip()
        name_list[i] = record

    return name_list


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    WIKI_SOURCE_CACHE = "target.html"
    with open(WIKI_SOURCE_CACHE, mode="r") as fin:
        res = fin.read()

    soup = BeautifulSoup(res, "html.parser")
    name_list = GetNameList(soup)
    print(name_list)
