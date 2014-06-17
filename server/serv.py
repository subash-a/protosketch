from flask import Flask, url_for, render_template
app = Flask(__name__)

@app.route('/')
def hello_world():
    return "hello world!"

@app.route('/index')
def index_load():
    return render_template("index.html")

if  __name__ == '__main__':
    url_for("static/css", filename="bootstrap.css")
    url_for("static/js", filename="jquery-1.7.1.js")
    app.debug = True
    app.run()
    
