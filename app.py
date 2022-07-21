
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, request

model = tf.keras.models.load_model('retinal-oct_finalJean_smaller.h5')
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
from keras import utils

labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

def prepare_image(img):
    img = io.BytesIO(img)
    #img = img.resize((150, 150))
    #img = np.array(img)
    #img = np.expand_dims(img, 0)
    #img = np.stack((img,)*3, axis=-1)
    return img


def predict_result(img):
    test_image = utils.load_img(img, target_size = (150, 150)) 
    test_image = utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    y_pred = model.predict(test_image)
    print(y_pred)

    #predict the result
    result = labels[np.argmax(y_pred)]
    print(result)
    return result


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    file = request.files.get('file')

    if not file:
        return

    img_bytes = file.read()
    img = prepare_image(img_bytes)

    return jsonify(prediction=predict_result(img))
    

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')