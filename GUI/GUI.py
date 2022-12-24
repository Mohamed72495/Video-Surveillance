import sys
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication , QDialog,QWidget
import face_recognition
import os
import numpy as np
import sqlite3
from PyQt5.QtCore import QTimer
import cv2
from PyQt5.QtGui import QImage ,QPixmap

path = 'sample images for recognition'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodingList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodingList.append(encode)
    return encodingList

encodingListKnown = findEncodings(images)


class WelcomeScreen(QDialog):
    def __init__(self):
        super(WelcomeScreen,self).__init__()
        loadUi('WelcomeScreen.ui',self)
        self.login.clicked.connect(self.gotologin)
        self.createacc.clicked.connect(self.gotocreate)
        
    def gotologin(self):
        login = LoginScreen()
        widget.addWidget(login)
        widget.setCurrentIndex(widget.currentIndex()+1)

    def gotocreate(self):
        create = CreateAccScreen()
        widget.addWidget(create)
        widget.setCurrentIndex(widget.currentIndex() + 1)

class CreateAccScreen(QDialog):
    def __init__(self):
        super(CreateAccScreen, self).__init__()
        loadUi("createacc.ui",self)
        self.passwordfield.setEchoMode(QtWidgets.QLineEdit.Password)
        self.confirmpasswordfield.setEchoMode(QtWidgets.QLineEdit.Password)
        self.signup.clicked.connect(self.signupfunction)
        self.Home.clicked.connect(self.gotoHome)
    def gotoHome(self):
        Welcome = WelcomeScreen()
        widget.addWidget(Welcome)
        widget.setCurrentIndex(widget.currentIndex() + 1)
    
    def signupfunction(self):
        user = self.emailfield.text()
        password = self.passwordfield.text()
        confirmpassword = self.confirmpasswordfield.text()

        if len(user)==0 or len(password)==0 or len(confirmpassword)==0:
            self.error.setText("Please fill in all inputs.")

        elif password!=confirmpassword:
            self.error.setText("Passwords do not match.")
        else:
            conn = sqlite3.connect("Database.db")
            cur = conn.cursor()

            user_info = [user, password]
            cur.execute('INSERT INTO login_info (username, password) VALUES (?,?)', user_info)

            conn.commit()
            conn.close()
            self.gotologin()
    def gotologin(self):
        login = LoginScreen()
        widget.addWidget(login)
        widget.setCurrentIndex(widget.currentIndex()+1)

   
class Camera(QDialog):
    def __init__(self):
        super(Camera,self).__init__()
        loadUi('CameraSet.ui',self)
        self.image=None

        self.startButton.clicked.connect(self.start_webcam)
        self.stopButton.clicked.connect(self.stop_webcam)
        self.pauseButton.clicked.connect(self.pause_webcam)
        self.Home.clicked.connect(self.gotoHome)
        self.detectButton.setCheckable(True)
        self.detectButton.toggled.connect(self.detect_webcam_face)
        self.face_Enabled = False
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.ip = ""
        self.passwordfield.setEchoMode(QtWidgets.QLineEdit.Password)
        self.usernamefield.setEchoMode(QtWidgets.QLineEdit.Password)
        self.ret = False

    def gotoHome(self):
        if self.ret == True:
            self.stop_webcam()

        Welcome = WelcomeScreen()
        widget.addWidget(Welcome)
        widget.setCurrentIndex(widget.currentIndex() + 1)
    
    

    def start_webcam(self):
        if self.ret == False:

            self.capture=cv2.VideoCapture()
            ip = self.ipfield.text()
            username = self.usernamefield.text()
            password = self.passwordfield.text()

            if self.checkBox.isChecked():
                self.error.setText("")
                self.capture=cv2.VideoCapture(0)
                self.start(self.capture)
            else:
                if len(ip) == 0 or len(username) == 0 or len(password) == 0:
                    self.error.setText("Please enter all fields.")
                else:
                    self.error.setText("")
                    self.capture.open("rtsp://" + username + ":" + password + "@" +ip+ "/Streaming/channels/101")
                    self.start(self.capture)
            
    def start(self, capture):
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,500)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,500)
        self.timer= QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    
    def update_frame(self):
        self.ret,self.image=self.capture.read()
        self.image=cv2.flip(self.image,1)
        if self.face_Enabled:
            detected_image = self.detect_face(self.image)
            self.displayImage(detected_image,1)
        else:
            self.displayImage(self.image,1)

    def detect_webcam_face(self , status):
        if self.ret == True:
            self.error.setText("")
            if status:
                self.detectButton.setText("Stop Recognition")
                self.face_Enabled = True
            else :
                self.detectButton.setText("Face Recognition")
                self.face_Enabled = False
        else:
            self.error.setText("start video to enable Recognition")

    def detect_face(self , frame):
        frameS = cv2.resize(frame, (0,0), None, 0.25, 0.25)
        frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(frameS)
        encodesCurFrame = face_recognition.face_encodings(frameS, facesCurFrame)

        for encodeFace,faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodingListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodingListKnown, encodeFace)

            
            matchIndex = np.argmin(faceDis)
            
            if faceDis[matchIndex] > 0.6:
                name = 'unknown'

            else:
                name = classNames[matchIndex].upper()

            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(frame, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.7 ,(255,255,255), 1)

        return frame

    

    def displayImage(self,img,window=1):
        qformat=QImage.Format_Indexed8
        if len(img.shape)==3:
            if img.shape[2]==4:
                qformat=QImage.Format_RGBA8888
            else:
                qformat =QImage.Format_RGB888 
        outImage= QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)   
        outImage=outImage.rgbSwapped()

        if window==1 :
            self.imgLabel.setPixmap(QPixmap.fromImage(outImage))
            self.imgLabel.setScaledContents(True)

    def stop_webcam(self):
        if self.ret == True:
            self.ret = False
            self.timer.stop()
            self.imgLabel.clear()
            self.capture.release()

    def pause_webcam(self):
        if self.ret == True:
            self.ret = False
            self.timer.stop()




class LoginScreen(QDialog):
    def __init__(self):
        super(LoginScreen, self).__init__()
        loadUi("login.ui",self)
        self.passwordfield.setEchoMode(QtWidgets.QLineEdit.Password)
        self.login.clicked.connect(self.loginfunction)
        self.Home.clicked.connect(self.gotoHome)

    def gotoHome(self):
        Welcome = WelcomeScreen()
        widget.addWidget(Welcome)
        widget.setCurrentIndex(widget.currentIndex() + 1)
        
    def loginfunction(self):
        user = self.emailfield.text()
        password = self.passwordfield.text()

        if len(user)==0 or len(password)==0:
            self.error.setText("Please input all fields.")

        else:
            conn = sqlite3.connect("Database.db")
            cur = conn.cursor()

            query = 'SELECT password FROM login_info WHERE username =\''+user+"\'"
            cur.execute(query)
            result_pass = cur.fetchone()[0]
            if result_pass == password:
                self.error.setText("")
                self.gotocamera()
                
            else:
                self.error.setText("Invalid username or password")
    def gotocamera(self):
        camera = Camera()
        widget.addWidget(camera)
        widget.setCurrentIndex(widget.currentIndex()+1)


#main
app=QApplication(sys.argv)
# Welcome=WelcomeScreen()
Welcome = Camera()
widget = QtWidgets.QStackedWidget()
widget.addWidget(Welcome)
widget.setFixedHeight(600)
widget.setFixedWidth(1000)
#widget.setWindowTitle('Welcome')
widget.show()
try:
    sys.exit(app.exec_())
except:
    print("Exiting")
