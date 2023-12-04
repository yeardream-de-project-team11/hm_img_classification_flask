from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import numpy as np

app = Flask(__name__)

# 모델 불러오기
model = load_model('./model/1128_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    # 이미지 파일을 읽고 전처리하기
    img = image.load_img(file, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)
    
    predictions = model.predict(img)

    # 가장 높은 확률을 가진 클래스 인덱스 찾기
    predicted_class_index = np.argmax(predictions)

    # 결과를 JSON 형식으로 반환
    return jsonify({'class_id': int(predicted_class_index)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)