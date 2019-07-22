# Usage: python app.py
from flask import send_from_directory
import os

from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
from model import SimpsonClassifier
import time
import uuid
import base64

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])

model = SimpsonClassifier(
    json_path='./models/simpson_model.json',
    weights_path='./models/simpson_model_weights.h5',
    pic_size=64)


def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)


def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4())  # Convert UUID format to a Python string.
    random = random.upper()  # Make all characters uppercase.
    random = random.replace("-", "")  # Remove the UUID '-'.
    return random[0:string_length]  # Return the random string.


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def template_test():
    return render_template('template.html', label='', imagesource='../uploads/no_image_available.jpg')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time()
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file.seek(0)
            # Run model
            result = model.run(file)

            print("RESULT:", result)

            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(
                app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str(time.time() - start_time))
            return render_template('template.html', label=result, imagesource='../uploads/' + filename)

        else:
            print('no file')
            return render_template('template.html', label=None, imagesource='../uploads/no_image_available.jpg')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == "__main__":
    app.debug = True

    app.run(host='0.0.0.0', port=3000, use_reloader=False)
