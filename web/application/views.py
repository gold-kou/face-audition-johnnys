from flask import Flask, render_template, request
import cv2
from sklearn.externals import joblib
import uuid
from PIL import Image
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ml import learn_measure
from web.application.models import facedetector_gcv
import configparser
import logging
import logging.config


# 外部のコンフィグを読み込む
inifile = configparser.ConfigParser()
inifile.read('application/config.ini')
inifile_secret = configparser.ConfigParser()
inifile_secret.read('instance/config.ini')

# 外部のログコンフィグを読み込む
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('root')

app = Flask(__name__)


@app.route('/')
def index():
    """
    /にGETリクエストが来たらindex.htmlを返す関数
    :return: index.html
    """
    logging.info('Request to /')
    return render_template('index.html')


@app.route('/upload')
def upload_image():
    """
    /uploadにGETリクエストが来たらupload.htmlを返す関数
    :return: upload.html
    """
    logging.info('Request to /upload')
    return render_template('upload.html')


@app.route('/result', methods=['POST'])
def face_result():
    """
    /resultにPOSTリクエストが来たらリクエストパラメータに含まれる画像ファイルから
    ジャニーズ顔かどうかを判定してresult.htmlを返す関数
    :return: result.html
    """
    logging.info('Request to /result')

    # 各種コンフィグ値取得
    cascade = inifile.get('path', 'cascade')
    pickle = inifile.get('path', 'pickle')
    scaler = inifile.get('path', 'scaler')
    api_key = inifile_secret.get('gcv', 'api_key')
    max_results = int(inifile.get('gcv', 'face_num'))
    gcv_url = inifile.get('gcv', 'url')
    no_face_message = inifile.get('error', 'no_face_message')
    no_image_message = inifile.get('error', 'no_image_message')

    # 顔抽出画像の保存先。UUIDで重複防止。
    save_path = '/tmp/' + str(uuid.uuid4()) + 'extracted_face.jpg'

    # 許可する画像拡張子
    allowed_extension = ['png', 'jpg', 'jpeg']

    # リクエストフォームからアップロードされた画像取得
    request_img = request.files['face']

    # 画像ファイルが適切にアップロードされているかの判定
    if request_img and request_img.filename.rsplit('.', 1)[1] in allowed_extension:
        # アップロードされた画像が顔画像かの判定
        if facedetector_gcv(request_img.read(), api_key=api_key, max_results=max_results, gcv_url=gcv_url):
            # カスケードファイルのよみ込み
            cascade = cv2.CascadeClassifier(cascade)
            # リクエスト画像をnumpy配列に変換
            request_img_numpy = np.array(Image.open(request_img))
            # 顔抽出
            face_img = cascade.detectMultiScale(request_img_numpy, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
            # 顔抽出画像保存
            for rect in face_img:
                x = rect[0]
                y = rect[1]
                width = rect[2]
                height = rect[3]
                dst = request_img_numpy[y:y + height, x:x + width]
                cv2.imwrite(save_path, dst)
                logging.info('Extracted face image of uploaded file is saved.')
        else:
            logging.error('The GCV could not detect face from the uploaded file.')
            return render_template('error.html', message=no_face_message)
    else:
        logging.error('The extension of uploaded file is not valid.')
        return render_template('error.html', message=no_image_message)

    # pickleの読み込み
    clf = joblib.load(pickle)

    # 顔抽出画像を100×100ピクセルにリサイズ
    face_img = cv2.imread(save_path)
    resized_img = cv2.resize(face_img, (100, 100))

    # NumPy変換
    resized_img = np.asarray(Image.fromarray(resized_img))

    # 3次元から1次元に変換
    vector = []
    vector.append(learn_measure.convert_image_vector(resized_img))

    # スケール変換
    scaler = joblib.load(scaler)
    scaled_vector = scaler.transform(vector)

    # スコアによる分類（閾値は若干ずらす）
    score = clf.decision_function(scaled_vector).ravel()

    # ジャニーズ本人レベルの場合
    if score < -2.5:
        logging.info('The classification is done. Result: 1')
        result = 1
        return render_template('result.html', face=save_path, result=result, score=score)
    # ジャニーズぽいレベルの場合
    elif -2.5 <= score < -1.0:
        logging.info('The classification is done. Result: 2')
        result = 2
        return render_template('result.html', face=save_path, result=result, score=score)
    # ジャニーズ顔でない場合
    else:
        result = 3
        logging.info('The classification is done. Result: 3')
        return render_template('result.html', face=save_path, result=result, score=score)
