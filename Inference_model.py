# from tensorflow.keras.backend import clear_session
import tensorflow as tf
import numpy
import flask
import cv2
import magic
import gc
import keras
import os

from fractions import Fraction    # 분수
from decimal import *             # 소수
from flask_api import FlaskAPI, status, exceptions
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.models import backbone, load_model

app = flask.Flask(__name__)
predictList=['window', 'door', 'house','tree', 'triangle roof', 'roof', 'veiled window', 'mountain','smoking chimney', 'fense', 'ground line', 'patterned roof', 'side door house','poor wall', 'half sun', 'solid wall', '2nd floor window house']

model = None
graph = None
before_halfwidth = None
before_halfheight = None
beforeArea = None
roof = ""


def load_models():
    # https://github.com/keras-team/keras/issues/2397
    global model
    custom_objects = backbone('resnet50').custom_objects
    model = keras.models.load_model('house3-1.h5', custom_objects=custom_objects)
    model._make_predict_function()
    global graph
    graph = tf.compat.v1.get_default_graph()  # tf.get_default_graph()의 최신 버전


# For the root '/predict' we need to define a function named predict
# This function will take values from the ajax request and performs the prediction
# By getting response from flask to ajax
@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read image file string data
            postImage = flask.request.files["image"].read()
            # magic은 파일의 MIME type을 점검하는 라이브러리?
            extention = magic.from_buffer(postImage).split()[0].upper()
            if extention != 'JPEG' and extention != 'PNG':
                return flask.jsonify(data)
            else:
                # https://stackoverflow.com/questions/47515243/reading-image-file-file-storage-object-using-cv2
                # convert string data to numpy array
                npimg = numpy.fromstring(postImage, numpy.uint8)
                # convert numpy array to image
                img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            image, labelList, scoreList, x1, y1, x2, y2 = prepare_image(
                img, target=(256, 256))
            if image is False:
                return flask.jsonify(data)

            # preds = numpy.argmax(model(image).numpy())
            # clear_session()
            image = None
            postImage = None
            hList=[]

            data['label'] = labelList   # getHouse에서 labeList 중 house를 삭제하기 때문에 getHouse위에 위치해야 함.
            data['house'] = getHouse(labelList,scoreList,x1,y1,x2,y2)
            # data['house'] = hList.append(4)
            data['score'] = scoreList
            data['x1'] = x1        # int32 JSON으로 변환할 수 없으니 일반 int 형태로 변환
            data['y1'] = y1
            data['x2'] = x2
            data['y2'] = y2
            data["success"] = True
            app.logger.info(data)

    gc.collect()
    return flask.jsonify(data)

def prepare_image(image, target):
    with graph.as_default():
        x1 = []
        x2 = []
        y1 = []
        y2 = []
        scoreList = []
        labelList = []
        roofScore = 0
        houseScore = 0
        global roof


        height, width, channels = image.shape
        global beforeArea
        beforeArea = height * width
        global before_halfwidth
        before_halfwidth = width*0.5
        global before_halfheight
        before_halfheight = height*0.5
        print("높이 ==", height)
        print("너비 ==", width)
        print("넓이 ==", beforeArea)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = preprocess_image(image)
        # 학습할 때 672x672로 했기 때문에 Inference할 때도 672x672
        image, scale = resize_image(image, 672, 672)

        # process image
        boxes, scores, labels = model.predict_on_batch(numpy.expand_dims(
            image, axis=0))  # np.expand_dims : 3차원 이미지를 4차원으로. 왜?

        # correct for image scale
        boxes /= scale  # resize된 이미지 비율만큼 박스 크기를 조정?

        # visualize detections
        for i, (box, score, label) in enumerate(zip(boxes[0], scores[0], labels[0])):
            # scores are sorted so we can break
            if score < 0.5:
                break

            box = box.astype(int)       # int32으로 변환
            # float32 -> float64로 변환. float32는 JSON 변환이 안 되는데 float64는 JSON 변환이 된다.
            score = score.astype(float)

            # 루프 3종인 경우,
            if predictList[label] == 'roof' or predictList[label] == 'patterned roof' or predictList[label] ==   'triangle roof':
                if roofScore < score :
                    roofScore = score
                    roof = predictList[label]
                else:
                    # continue를 함으로써 append하지 않음.
                    print("CONTINUE -> ", predictList[label] , " 삭제!!")
                    continue

            # 집 2종인 경우
            if predictList[label] == 'house' or predictList[label] == 'side door house':
                if houseScore < score :
                    houseScore = score
                else:
                    print("CONTINUE -> ", predictList[label], " 삭제!!")
                    continue

            # List에 저장
            labelList.append(predictList[label])
            scoreList.append((score))
            # int32->int로 변환. int32형은 JSON으로 변환할 수 없기 때문.
            x1.append(int(box[0]))
            y1.append(int(box[1]))
            x2.append(int(box[2]))
            y2.append(int(box[3]))


        print('---------------')
        print('LABELLIST =', labelList)
        print('SCORELIST =', scoreList)
        print('x1 =', x1)
        print('y1 =', y1)
        print("x2 =", x2)
        print("y2 =", y2)

        return (image, labelList, scoreList, x1, y1, x2, y2)

def getHouse(labelList,scoreList,x1,y1,x2,y2):
    houseList = []
    append = houseList.append # 파이썬에서는 .을 쓰면 속도가 느려지는 현상이 있다고한다. 반복문에서만큼은 . 사용을 자제하기 위함.
    doorExist = False
    fenseAlreadyFound = False
    chimneyAlreadyFound = False
    veiledWindowAlreadyFound = False
    treeOrMountainAlrearFound = False
    countWindows = 0
    global roof

    # https://stackoverflow.com/questions/8197323/list-index-function-for-python-that-doesnt-throw-exception-when-nothing-found
    # labelList에 house가 존재하면 idx를 house로 초기화. house가 없으면 sdh로 초기화.
    try:
        idx = labelList.index('house') if 'house' in labelList else labelList.index('side door house')
    except:
        print("그림에 집이 없습니다.")

    # 집. 집을 먼저 찾아야 현관문, 지붕의 비율을 계산할수가 있다.
    houseArea = (x2[idx]-x1[idx]) * (y2[idx]-y1[idx])
    # house 비율 = house/그림 크기
    houseRatio = getRatio(houseArea, beforeArea)
    print('HOUSE RATIO = ', houseRatio)

    if labelList[idx] == 'side door house':
        print("측면의 현관문")
        append(20)

    # 444 지나치게 큰 집(3/4)
    if houseRatio >= 0.75:
        print('3/4 이상 -> 지나치게 큰 집')
        append(4)
    # 555 지나치게 작은집 (1/4)
    elif houseRatio <= 0.25:
        print('1/4 이하 -> 지나치게 작은 집')
        append(5)
    # 777 위치 (좌측) -> 집의 우측하단 x좌표가 이미지 가로길이 절반에 못 미치면,
    if x2[idx] < before_halfwidth:
        print('좌측에 위치한 집')
        print(x2[idx],' < ', before_halfwidth)
        append(7)
    # 888 위치 (우측) -> 집의 좌측상단 x좌표가 이미지 가로길이 절반을 넘어서면,
    elif x1[idx] > before_halfwidth:
        print('우측에 위치한 집')
        print(x1[idx], ' > ', before_halfwidth)
        append(8)
    # 999 위치 (하단) -> 집의 좌측상단 y좌표가 이미지 세로길이 절반에 못 미치면,
    # box[1] y좌표는 아래(0)로부터 값이 상승하지만, halfheight 이미지 세로의 절반은 위(0)에서부터 값이 상승하는 독특한 구조.
    if y1[idx] > before_halfheight:
        print('하단에 위치한 집')
        print(y1[idx], ' > ', before_halfheight)
        append(9)

    for i in range(len(labelList)):
        if labelList[i] =='house':
            continue
        # 창문과 같이 labelList에 여러개가 섞여 있을 확률이 높은 개체의 조건문을 위쪽에 배치하여 성능 최적화.
        # 창문
        if labelList[i] =='window' or labelList[i] == 'veiled window':
            print("window found")
            countWindows += 1
            if  veiledWindowAlreadyFound == False and labelList[i] =='veiled window':
                print("커튼, 창살 등으로 가려진 창문")
                append(25)
                veiledWindowAlreadyFound = True
            continue
        # 기타
        if fenseAlreadyFound == False and labelList[i] =='fense':
            print("울타리의 표현, 울타리처럼 지면이 표현")
            append(31)
            fenseAlreadyFound = True
            continue

        if treeOrMountainAlrearFound == False and (labelList[i] == 'mountain' or labelList[i] == 'tree'):
            print("산속이나 숲속의 집의 표현")
            append(30)
            treeOrMountainAlrearFound = True
            continue

        # 굴뚝
        if chimneyAlreadyFound == False and labelList[i] =='smoking chimney':
            print("굴뚝의 연기")
            append(27)
            chimneyAlreadyFound = True
            continue
        # 현관문
        if labelList[i] =='door':
            doorExist = True
            doorArea = (x2[i]-x1[i]) * (y2[i]-y1[i])
            doorRatio = getRatio(doorArea, houseArea)  # door 비율 = door Area / house Area
            print('door RATIO = ', doorRatio)

            # 181818 현관문이 과하게 클 경우 3/4
            if doorRatio >= 0.75:
                print("과도하게 큰 현관문")
                append(18)
            # 191919 현관문이 과하게 작을 경우 1/4
            elif doorRatio <= 0.25:
                print('과도하게 작은 현관문')
                append(19)
            continue
        # 지붕
        if labelList[i] == roof:
            roofArea = (x2[i]-x1[i]) * (y2[i]-y1[i])
            roofRatio = getRatio(roofArea, houseArea) # roof 비율 = roof Area/house Area
            print('roof RATIO = ', roofRatio)

            # 131313 과도하게 큰 지붕 3/4
            if roofRatio >= 0.75:
                print('과도하게 큰 지붕')
                append(12)
            
            if roof == 'patterned roof':
                print('과도한 지붕의 무늬 표현')
                append(13)
            elif roof == 'triangle roof':
                print("뾰족한 지붕의 표현, 세모 지붕")
                append(14)
            continue
        # 태양
        if labelList[i] == 'half sun':
            print("반만 나온 태양")
            append(28)
            continue
        
        # 벽
        if labelList[i] == 'poor wall':
            print("허술한 벽")
            append(16)
        if labelList[i] == 'solid wall':
            print("지나치게 견고한 벽돌이나 벽면의 표현")
            append(17)

    # 222222 현관문의 생략
    if doorExist == False:
        print('현관문의 생략')
        append(22)
    # 232323 창문의 생략
    if countWindows == 0:
        print('창문의 생략')
        append(23)
    # 242424 3개 이상 많은 창문
    elif countWindows >= 3:
        print('3개 이상 많은 창문')
        append(24)

    # SET으로 변환 시켰다가 LIST로 변환하면서 중복 제거. 울타리 중복이 있다.
    # return list(set(houseList))
    return houseList

def getRatio(a,b):
    frac = Fraction(a, b)
    decimal = float(frac)

    return Decimal(decimal).quantize(Decimal('0.00'))

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..." "please wait until server has fully started"))
    print("START")

    # change the host and port as 0.0.0.0. / 5000 when we need to deploy our app to AWS
    # defaualt(local) is 127.0.0.1 / 5000
    load_models()
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=5000)
