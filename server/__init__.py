import os
from flask import Flask, url_for, render_template, request
import template_detection as te
upload_path = "uploads/"
app = Flask(__name__)

def getExtension(filename):
    name_array = filename.split(".")
    return name_array[len(name_array) - 1]

@app.route('/')
def hello_world():
    return "hello world!"

@app.route('/index')
def intdex_load():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def handle_upload():
    if request.method == 'POST':
        print "POST Method"
        file = request.files['file']
        filepath  = upload_path+"queryimage."+getExtension(file.filename)
        file.save(filepath)
        te.__main__(filepath)
        return "file received"
    else:
        return "Did not get uploaded"

if  __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')
    
