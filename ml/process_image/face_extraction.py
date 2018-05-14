# -*- coding: utf-8 -*-
"""
OpenCV2を利用して画像から顔画像を抽出するモジュール。
このモジュールを利用して抽出した結果から不要なファイルを人力で削除する必要あり。
"""

import cv2
import os
import configparser


# 外部のコンフィグを読み込む
inifile = configparser.ConfigParser()
inifile.read('../config.ini')

# 入力画像ディレクトリのパス。最後はスラッシュで終わる必要あり。
in_dir = inifile.get('extraction', 'in')
# 出力先ディレクトリのパス。最後はスラッシュで終わる必要あり。
out_dir = inifile.get('extraction', 'out')
# カスケードファイルのパス。
cascade_file = inifile.get('extraction', 'cascade')

# ディレクトリに含まれるファイル名の取得
names = os.listdir(in_dir)

for name in names:
    # 絶対パスで画像の読み込み
    image_gs = cv2.imread(in_dir + name)

    # 顔認識用特徴量ファイルを読み込む
    cascade = cv2.CascadeClassifier(cascade_file)

    # 顔認識の実行
    face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

    # 顔だけ切り出して保存
    index = 0
    for rect in face_list:
        x = rect[0]
        y = rect[1]
        width = rect[2]
        height = rect[3]
        dst = image_gs[y:y + height, x:x + width]
        save_path = out_dir + '/' + 'out_(' + str(index) + ')' + str(index) + '.jpg'
        cv2.imwrite(save_path, dst)
        index = index + 1
