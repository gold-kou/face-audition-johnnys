# -*- coding: utf-8 -*-
"""
画像サイズを変更するモジュール
scikit-learnのモデルに入力するために全ての画像のサイズを統一する必要がある。
"""

import cv2
import os
import configparser

inifile = configparser.ConfigParser()
inifile.read('../config.ini')

# 入力画像ディレクトリのパス。最後はスラッシュで終わる必要あり。
in_dir = inifile.get('resize', 'in')
# 出力画像ディレクトリのパス。最後はスラッシュで終わる必要あり。
out_dir = inifile.get('resize', 'in')

files = os.listdir(in_dir)
for file in files:
    # 画像の読み込み
    image = cv2.imread(in_dir + file)

    # 100×100ピクセルにリサイズ
    image_resized = cv2.resize(image, (100, 100))

    # リサイズした画像の書き込み
    cv2.imwrite(out_dir + file, image_resized)
