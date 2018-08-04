# -*- coding: utf-8 -*-
"""
画像をy軸に回転させるモジュール
"""

import os
import cv2
import configparser

inifile = configparser.ConfigParser()
inifile.read('../config.ini')

# 入力画像ディレクトリのパス。最後はスラッシュで終わる必要なし。
in_dir = inifile.get('rotation', 'in')
# 出力先ディレクトリのパス。最後はスラッシュで終わる必要なし。
out_dir = inifile.get('rotation', 'out')

for path, _, files in os.walk(in_dir):
    for file in files:
        if not file.startswith('.'):
            # 画像読み込み
            img = cv2.imread((path + '/' + file), cv2.IMREAD_COLOR)

            # y軸に回転
            reversed_y_img = cv2.flip(img, 1)

            # 画像保存
            save_path = out_dir + '/' + 'out_rotation_(' + file + ').jpg'
            cv2.imwrite(save_path, reversed_y_img)
