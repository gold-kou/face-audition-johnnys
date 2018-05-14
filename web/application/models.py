import base64
from requests import Request, Session
import json
import logging
import logging.config


# 外部のログコンフィグを読み込む
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('root')


def facedetector_gcv(image, api_key, max_results, gcv_url):
    """
    Google Cloud Vision(GCV) APIを利用して画像に人間の顔が含まれているかどうかを判定する関数
    :param image: 画像ファイル
    :param api_key: GCV利用のためAPI KEY
    :param max_results: いくつの顔を判定するか。数が少ないほど精度高い。
    :param gcv_url: GCV APIのURL
    :return:boolean
    """

    # GCVのRequest設定
    str_headers = {'Content-Type': 'application/json'}
    batch_request = {'requests': [{'image': {'content': base64.b64encode(image).decode('utf-8')},
                                   'features': [{'type': 'FACE_DETECTION', 'maxResults': max_results, }]}]}

    # セッション作ってリクエスト送信
    obj_session = Session()
    obj_request = Request('POST', gcv_url + api_key, data=json.dumps(batch_request), headers=str_headers)
    obj_prepped = obj_session.prepare_request(obj_request)
    obj_response = obj_session.send(obj_prepped, verify=True, timeout=180)

    # Responseからjsonを抽出
    response_json = json.loads(obj_response.text)
    logging.info('GCV request is successed')

    # リクエスト画像に顔が含まれているかどうかの判定結果（'faceAnnotations'があれば顔あり）
    if 'faceAnnotations' in response_json['responses'][0]:
        return True
    else:
        return False
