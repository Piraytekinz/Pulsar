from flask import Flask, render_template, redirect, request
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import soundfile as sf
import io
from tensorflow.keras.models import load_model


latent_dim = 100
output_shape = (301824, 1)

def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

generator = load_model('C:\\Users\\anang\\OneDrive\\Desktop\\echo\\static\\new_gen_gen_audio.h5')


app = Flask(__name__)

@app.route("/")
def build():
    return render_template('index.html')



@app.route('/upload', methods=['GET', 'POST'])
def upload():
    # Get the image file from the request
    image_data = request.files['file'].read()

    # Decode the image from base64
    # img = Image.open(io.BytesIO(image_data))


    image = load_img(io.BytesIO(image_data), target_size=(128, 128))
    # convert image pixels to numpy array
    image = img_to_array(image)
    # reshape data for model
    image = image.reshape((image.shape[0], image.shape[1], image.shape[2]))
    # preprocess image for vgg
    image = (image / 127.5) - 1

    generated_audio = generate_audio(generator , latent_dim, output_shape[0], image)

    # Save the generated audio using soundfile

    sf.write('static/generated_audio.wav', generated_audio, 16000)

    return render_template('index.html')


def generate_audio(generator, latent_dim, length, image):
    # noise = np.random.normal(0, 1, (1, latent_dim))
    image = tf.expand_dims(image, axis=0)
    generated_audio = generator.predict(image)
    print(generated_audio)
    return generated_audio.flatten()




# @app.route("/upload", methods=['POST'])
# def upload():
#     return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
