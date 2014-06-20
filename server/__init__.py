from flask import Flask, url_for, render_template, request
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
        if request.files["name"]:
            print "file received"
        else:
            return "Did not get uploaded"

if  __name__ == '__main__':
    app.debug = True
    app.run()
    
