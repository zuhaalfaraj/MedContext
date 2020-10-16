from flask import Flask, render_template, request,redirect,url_for,flash
from werkzeug.utils import secure_filename
import os
from main_NER import NameEntitiyRecognitionClinicla
UPLOAD_FOLDER = ''
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
    if request.method == 'POST':
        print(request.form['text'])
        if request.form['text'] is not None:
            text = request.form['text'].strip("\n").strip("\r").split("\t")
            print(text)
            out= main.get_entities('model',text)
            return render_template('index.html', out=out)



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        print("i'm here")
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            data = main.img_to_txt('list.png')
            out = main.get_entities("model", data)

            return render_template('index.html',
                                    out=out)

if __name__== "__main__":
    app.run(debug=True)