import os
import warnings
import keras
import openpyxl
import numpy as np
import tensorflow as tf
import efficientnet.tfkeras as efn
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import image_utils

warnings.filterwarnings('ignore')
# 忽略AVX2 FMA的警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def get_image(picname, model, confidence=0.70):
    keras.backend.clear_session()
    img_path = picname
    test_image = image_utils.load_img(img_path, target_size=(224, 224))
    test_image = image_utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255
    predict = model.predict(test_image)
    preds = np.argmax(predict, axis=1)[0]
    if predict[0][preds] < confidence:
        finalResult = '无关图片'

    else:
        myExcel = openpyxl.load_workbook('../data/message.xlsx')  # 获取表格文件
        mySheet = myExcel['Sheet1']  # 获取指定的sheet
        finalResult = (mySheet.cell(row=preds + 2, column=2)).value
    return finalResult, round(predict[0][preds] * 100, 2)


def get_result(path):
    # 置信度
    confidence = 0.70
    model_path = '../train/EfficientNet_B0_No_Aug_best.h5'
    model = load_model(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    keras.backend.clear_session()

    return get_image(path, model, confidence)
