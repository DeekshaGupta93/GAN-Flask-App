#!/usr/bin/env python
import random
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
from flask import Flask, render_template, request
from keras.datasets import mnist
import cv2.cv2

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

generator, discriminator = None, None
X_train = None
from_generator = False


def init():
    global generator
    global discriminator
    global X_train
    generator = load_model('static/models/gan_generator_epoch_200.h5')
    discriminator = load_model('static/models/gan_discriminator_epoch_200.h5')
    (X_train, _) = mnist.load_data()


@app.route('/')
def hello():
    author = "Deeksha"
    return render_template('index.html', author=author)


@app.route("/generator/")
def generator_route():
    global from_generator
    p = random.randint(0, 1)
    if p == 0:
        noise = np.random.normal(0, 1, size=[1, 100])
        generated_images = generator.predict(noise)
        generated_images = generated_images.reshape(1, 28, 28)
        _ = plt.imshow(generated_images[0], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
        plt.savefig('./static/raw/generated.png')
        from_generator = True

    else:
        img = X_train[0]
        img = img[random.randint(0, 60000)]
        _ = plt.imshow(img, interpolation='nearest', cmap='gray_r')
        plt.axis('off')
        plt.savefig('./static/raw/generated.png')
        from_generator = False

    return render_template('generator.html', paramsv={'message': str(from_generator)})


@app.route("/discriminator/")
def discriminator_route():
    return render_template('discriminator.html')


@app.route('/discriminator/test', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save('static/raw/discriminator/test.png')
    img = cv2.imread('static/raw/discriminator/test.png', 0)
    if not img.shape == (28, 28):
        img = cv2.resize(img, (28, 28))
    flat_img = img.reshape(1, 784)
    flat_img = (flat_img.astype(np.float32) - 127.5) / 127.5
    output = discriminator.predict(flat_img)[0][0]
    if output == 0:
        img = ~img
        img = img.reshape(1, 784)
        img = (img.astype(np.float32) - 127.5) / 127.5
        output = discriminator.predict(img)[0][0]
    return render_template('test.html', output=str(output))


@app.after_request
def add_header(response):
    response.headers[
        'Cache-Control'] = 'no-store, no-cache, must-revalidate, post - check = 0, pre - check = 0, max - age = 0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


init()
if __name__ == '__main__':
    print("* Loading Keras model and Flask starting server..."
          "please wait until server has fully started")
    app.run(host='0.0.0.0', debug=False)
