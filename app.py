from flask import Flask, render_template, request,redirect,url_for,flash
from werkzeug.utils import secure_filename
import os
from main_NER import NameEntitiyRecognitionClinicla
UPLOAD_FOLDER = '/home'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
main=NameEntitiyRecognitionClinicla()
app = Flask(__name__)
app.secret_key = 'zuhzuh1122'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/submit', methods=['POST'])
def submit():
    print(request.form['text'])
    return 'You entered: {}'.format(request.form['text'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(file)
            data= main.full_process(filename)
            print(data)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            return redirect(url_for('uploaded_file',
                                    filename=filename))

if __name__== "__main__":
    app.run(debug=True)