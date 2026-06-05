from flask import *
import numpy as np
import os
from functools import wraps
import webbrowser
import ctypes

from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from flask_mysqldb import MySQL
from tqdm import tqdm
import hashlib
import controller as ct
import cv2
import time
import key_validator as gk
app=Flask(__name__, template_folder='templates', static_folder='static')
app.config['MYSQL_HOST']='localhost'
app.config['MYSQL_USER']='root'
app.config['MYSQL_PASSWORD']='root'
app.config['MYSQL_DB']='hmp'
app.config['MYSQL_CURSORCLASS']='DictCursor'
mysql=MySQL(app)     

##@app.route('/login',methods=['POST','GET'])
##def login():
##    status=True
##    if request.method=='POST':
##        email=request.form["email"]
##        pwd=request.form["upass"]
##        cur=mysql.connection.cursor()
##        cur.execute("select * from admin where email=%s and password=%s",(email,ct.md5(pwd)))
##        data=cur.fetchone()
##        if data:
##            session['logged_in']=True
##            session['username']=data["username"]
##            flash('Login Successfully','success')
##            return redirect('home')
##        else:
##            flash('Invalid Login. Try Again','danger')
##    return render_template("login.html",url = url)

import hashlib  # Make sure to import this at the top of your file

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        email = request.form["email"]
        pwd = request.form["upass"]

        # Convert plain password to MD5
        hashed_pwd = hashlib.md5(pwd.encode()).hexdigest()

        # Connect to the database and check for matching email
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM admin WHERE email = %s", (email,))
        data = cur.fetchone()
        cur.close()

        # Check if record exists and verify credentials
        if data and data['email'] == 'admin@mail.com' and data['password'] == hashed_pwd:
            session['logged_in'] = True
            session['username'] = data['username']
            flash('Login Successful', 'success')
            return redirect('home')
        else:
            flash('Invalid Login. Try Again', 'danger')

    return render_template("login.html", url=url)

def is_logged_in(f):
	@wraps(f)
	def wrap(*args,**kwargs):
		if 'logged_in' in session:
			return f(*args,**kwargs)
		else:
			flash('Unauthorized, Please Login','danger')
			return redirect(url_for('login'))
	return wrap
global output_file
@app.route('/train', methods=['GET', 'POST'])
@is_logged_in
def train():
    global output_file
    return render_template('train.html',url=url,data = session['username'])

@app.route('/get_dataset', methods=['GET', 'POST'])
@is_logged_in
def get_dataset():
    if (os.listdir('../Dataset')):
        count = 0
        for root_dir, cur_dir, files in os.walk(r'../Dataset'):
            count += len(files)
        time.sleep(3)
        return str(count) + " images found"
    else:
        return "No dataset Found in the path specified. Copy the files to path and refresh and try again"
@app.route('/start_training', methods=['GET', 'POST'])
@is_logged_in
def start_training():
    ct.train()
    return "Training Completed"

@app.route('/save_model', methods=['GET', 'POST'])
@is_logged_in
def save_model():
    if(ct.save_model()):
        return "Model Saved Successfully"
    else:
        return "Failed to save model"
@app.route('/save_memo', methods=['GET', 'POST'])
@is_logged_in
def save_memo():
    time.sleep(2)
    return "Memo Saved Successfully"
@app.route('/show_accuracy', methods=['GET', 'POST'])
@is_logged_in
def show_accuracy():
    time.sleep(2)
    return send_file('../Plots/accuracy.jpg', mimetype='image/jpg')

@app.route('/show_loss', methods=['GET', 'POST'])
@is_logged_in
def show_loss():
    time.sleep(2)
    return send_file('../Plots/loss.png', mimetype='image/png')
@app.route('/predict', methods=['GET', 'POST'])
@is_logged_in
def predict():
    global output_file
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = os.path.join('static','input_images',secure_filename(f.filename))
        f.save(file_path)
        output_file = f.filename
    return render_template('demo.html',url=url,filename = file_path)

@app.route('/')
def index():
    return render_template('login.html',url=url)



  
#Registration  


#Home page
@app.route("/home",methods=['POST','GET'])
@is_logged_in
def home():
    global url
    return render_template('index.html',data = session['username'],url = url)
@app.route("/logout")
def logout():
	session.clear()
	flash('You are now logged out','success')
	return redirect(url_for('login'))

if __name__ == '__main__':
    global url
    app.secret_key='secret123'
    myIP = ct.get_ip_address_of_host()
    url = 'http://' + myIP + ':5001'
    app.run(debug=False, host='0.0.0.0',port = 5001)

