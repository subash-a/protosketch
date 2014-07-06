from flask import Flask, url_for, render_template, request
import template_detection as te
app = Flask(__name__)

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
        print request.form
        return "file received"
#       te.__main__(None)
    else:
        return "Did not get uploaded"

if  __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')
    
