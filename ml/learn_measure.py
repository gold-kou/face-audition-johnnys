# -*- coding: utf-8 -*-
"""
学習と評価を行うモジュール。
"""
from PIL import Image
import os
import numpy as np
from sklearn.externals import joblib
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import cv2
import configparser
import logging
import logging.config

# 外部のコンフィグを読み込む
inifile = configparser.ConfigParser()
inifile.read('config.ini')

# 外部のログコンフィグを読み込む
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('root')


def get_images(path):
    """
    ディレクトリに含まれる全ての画像を取得する関数。
    :param path: 画像ファイルが格納されたディレクトリパス。
    :return: 画像ファイル群のリスト
    """
    images = []
    files = os.listdir(path)
    for file in files:
        # 画像の読み込み
        image = cv2.imread(path + file)
        images.append(image)
    return images


def convert_image_vector(img: object) -> object:
    """
    画像データから入力ベクトルを作成する関数。
    3次元(100x100x3(RGB))から1次元に変換する。
    :param img: 画像ファイル
    :return: scikit-learnに入力可能なnumpy配列
    """
    s = img.shape[0] * img.shape[1] * img.shape[2]
    img_vector = img.reshape(1, s)
    return img_vector[0]


def make_vectors_labels(path1, path2):
    """
    画像ベクトルとラベルのリストを作成する関数。
    :param path1: ジャニーズ画像ディレクトリのパス
    :param path2: ぶさいく画像ディレクトリのパス
    :return: 画像ベクトルのリストとラベルのリスト（各リストの順番が対応）
    """
    # 初期化
    labels = []
    vectors = []
    images = []

    # ジャニーズ画像の読み込みとラベル作成
    johnnys_images = get_images(path1)
    index = 0
    while index < len(johnnys_images):
        # ジャニーズのラベルは0
        labels.append(0)
        index += 1

    # ぶさいく画像の読み込みとラベル作成
    busaiku_images = get_images(path2)
    index = 0
    while index < len(busaiku_images):
        # ぶさいくのラベルは1
        labels.append(1)
        index += 1

    # 画像リスト作成（前半がジャニーズで後半がぶさいく）
    for image in johnnys_images:
        images.append(image)
    for image in busaiku_images:
        images.append(image)

    # 画像データからベクトルデータ作成
    for image in images:
        # opened_image = np.asarray(Image.open(image)) # Macだと動作しなかった
        opened_image = np.asarray(Image.fromarray(image))
        # 3次元かどうか
        if (len(opened_image.shape) == 3):
            # RGBが3がどうか
            if (opened_image.shape[2] == 3):
                # ベクトル化
                vectors.append(convert_image_vector(opened_image))
            else:
                logging.error('Making vector is skipped as the image is not rgb3.')
        else:
            logging.error('Making vector is skipped as the image is not 3 dimensions.')
    return vectors, labels


def learn(vectors, labels, path):
    """
    学習処理を行う関数。
    学習結果はpickleにする。
    :param vectors: 学習データの画像ベクトルリスト
    :param labels: 学習データのラベルリスト
    :param path: pickle先のパス
    :return: none
    """

    # 分類機生成。パラメータは下記グリッドサーチから得られた結果。
    clf = svm.SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3,
                  gamma=0.0001, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)

    # SVCをグリッドサーチ
    # parameters = [
        # {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        # {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},
        # {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [0.001, 0.0001]},
        # {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}
    # ]
    # gs = GridSearchCV(clf, parameters, cv=5, scoring='f1')
    # gs.fit(vectors, labels)
    # logging.info('cv_results_: ')
    # logging.info(gs.cv_results_)
    # logging.info('best_estimator_: ')
    # logging.info(gs.best_estimator_)

    # 学習
    clf.fit(vectors, labels)

    # pickle作成
    joblib.dump(clf, path)


def measure(vectors, labels, path):
    """
    評価スコアを算出する関数。
    :param vectors: テストデータの画像ベクトルリスト
    :param labels: 学習データのラベルリスト
    :param path: pickle先のパス
    :return: none
    """
    # pickleの読み込み
    clf = joblib.load(path)

    # tmp
    predict_result = clf.predict(vectors)
    logging.info("predict_result")
    logging.info(predict_result)

    # precision出力
    logging.info("precision: ", end="")
    logging.info(accuracy_score(labels, clf.predict(vectors)))

    # f1スコア出力
    logging.info("precision: ", end="")
    logging.info(f1_score(labels, clf.predict(vectors)))

    # decision_function出力
    tmp = clf.decision_function(vectors)
    index = 0
    for t in tmp:
        logging.info("index: ", end="")
        logging.info(index)
        logging.info(t)
        index = index + 1


def main(scale_flag):
    """
    main関数
    :param scale_flag: スケール変換を実施するかのフラグ
    :return: none
    """
    # johnnys学習データディレクトリのパス。最後はスラッシュで終わる必要あり。
    johnnys_learn_path = inifile.get('johnnys', 'learn')
    # busaiku学習データディレクトリのパス。最後はスラッシュで終わる必要あり。
    busaiku_learn_path = inifile.get('busaiku', 'learn')
    # johnnysテストデータディレクトリのパス。最後はスラッシュで終わる必要あり。
    johnnys_predict_path = inifile.get('johnnys', 'predict')
    # busaikuテストデータディレクトリのパス。最後はスラッシュで終わる必要あり。
    busaiku_predict_path = inifile.get('busaiku', 'predict')
    # pickleファイルパス。
    clf_pickle_path = inifile.get('pickle', 'clf')
    mms_pickle_path = inifile.get('pickle', 'mms')

    # 学習データのベクトルとラベル取得
    learn_vectors, learn_labels = make_vectors_labels(johnnys_learn_path, busaiku_learn_path)
    # テストデータのベクトルとラベル取得
    predict_vectors, predict_labels = make_vectors_labels(johnnys_predict_path, busaiku_predict_path)
    logger.info('Getting vectors and labels is done.')

    # スケール変換
    if scale_flag:
        mms = MinMaxScaler()
        mms.fit(learn_vectors)
        joblib.dump(mms, mms_pickle_path)
        learn_vectors = mms.transform(learn_vectors)
        predict_vectors = mms.transform(predict_vectors)
    logging.info('The scale translation is done')

    # 学習
    learn(learn_vectors, learn_labels, clf_pickle_path)
    logging.info('The learn is done')

    # テスト
    measure(predict_vectors, predict_labels, clf_pickle_path)
    logging.info('The predict is done.')


if __name__ == '__main__':
    main(True)
