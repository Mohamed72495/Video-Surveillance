import sys
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QDialog, QWidget, QMessageBox
import face_recognition
import os
import numpy as np
import sqlite3
from PyQt5.QtCore import QTimer
import cv2
from PyQt5.QtGui import QImage ,QPixmap

import paddle

import torch
import time
import re
import easyocr
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import ImageFont, ImageDraw, Image
import arabic_reshaper 
from bidi.algorithm import get_display
import matplotlib.gridspec as gridspec
from paddleocr import PaddleOCR




OCR_TH = 0.5
ocr = PaddleOCR(lang="arabic", use_angle_cls=True)
license_model =  torch.hub.load('ultralytics/yolov5', 'custom', path= 'E:\Studying\ASU\Senior-2\Graduation Project\GUI\License Detection/best.pt',force_reload=True) 
license_classes = license_model.names 
plates = []


faceDetection_model =  torch.hub.load('ultralytics/yolov5', 'custom', path= 'E:\Studying\ASU\Senior-2\Graduation Project\GUI\Face Detection/best.pt',force_reload=True) 

faceDetection_classes = faceDetection_model.names 


# path = 'sample images for recognition'
# images = []
# classNames = []
# myList = os.listdir(path)

# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')
#     images.append(curImg)
#     classNames.append(os.path.splitext(cl)[0])

# def findEncodings(images):
#     encodingList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]
#         encodingList.append(encode)
#     return encodingList

# encodingListKnown = findEncodings(images)


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



class Camera(QDialog):
    def __init__(self):
        super(Camera,self).__init__()
        loadUi('CameraSet.ui',self)
        self.image=None

        self.startButton.clicked.connect(self.start_cam)
        self.stopButton.clicked.connect(self.stop_cam)

        self.pauseButton.setCheckable(True)
        self.pauseButton.clicked.connect(self.check_pause)

        self.Home.clicked.connect(self.gotoHome)

        self.detectButton.setCheckable(True)
        self.detectButton.toggled.connect(self.check_faceDetection)
        self.faceDetection_Enabled = False
        

        self.recognitionButton.setCheckable(True)
        self.recognitionButton.toggled.connect(self.check_faceRecognition)
        self.faceRecognition_Enabled = False

        self.licenseButton.setCheckable(True)
        self.licenseButton.toggled.connect(self.check_licenseDetection)
        self.licenseDetection_Enabled = False

        self.carButton.setCheckable(True)
        self.carButton.toggled.connect(self.check_carDetection)
        self.lcarDetection_Enabled = False

        self.checkBox.stateChanged.connect(self.checkBox_state)
        
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.ip = ""
        self.passwordfield.setEchoMode(QtWidgets.QLineEdit.Password)
        self.usernamefield.setEchoMode(QtWidgets.QLineEdit.Password)
        self.ret = False

        
            
        

    def checkBox_state(self):

        if self.checkBox.isChecked():
            self.ipfield.setReadOnly(True)
            self.ipfield.setStyleSheet("QLineEdit{background-color : rgba(211, 211, 211, 1);}")
            self.usernamefield.setReadOnly(True)
            self.usernamefield.setStyleSheet("QLineEdit{background-color : rgba(211, 211, 211, 1);}")
            self.passwordfield.setReadOnly(True)
            self.passwordfield.setStyleSheet("QLineEdit{background-color : rgba(211, 211, 211, 1);}")
        else :
            self.ipfield.setReadOnly(False)
            self.ipfield.setStyleSheet("QLineEdit{background-color : rgba(255, 255, 255, 1);}")
            self.usernamefield.setReadOnly(False)
            self.usernamefield.setStyleSheet("QLineEdit{background-color : rgba(255, 255, 255, 1);}")
            self.passwordfield.setReadOnly(False)
            self.passwordfield.setStyleSheet("QLineEdit{background-color : rgba(255, 255, 255, 1);}")
    
    
    


    def start_cam(self):
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
        # self.image=cv2.flip(self.image,1)

        if self.faceDetection_Enabled:
            detected_image = self.detect_face(self.image)
            self.displayImage(detected_image,1)
        else:
            self.displayImage(self.image,1)

        # if self.faceRecognition_Enabled:
        #     detected_image = self.recognize_face(self.image)
        #     self.displayImage(detected_image,1)
        # else:
        #     self.displayImage(self.image,1)

        if self.licenseDetection_Enabled:
            detected_image = self.detect_license(self.image)
            self.displayImage(detected_image,1)
        else:
            self.displayImage(self.image,1)

    
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


# ---------------------------------------------------------------------------------------------------------
# --------------------------------------- Face Dedtection Functions ---------------------------------------
# ---------------------------------------------------------------------------------------------------------


    def check_faceDetection (self, status):
        if self.ret == True:
            if status:
                self.detectButton.setText("Stop Face Detection")
                self.faceDetection_Enabled = True
            else :
                self.detectButton.setText("Face Detection")
                self.faceDetection_Enabled = False


    def detect_face(self, frame):
        # 
        # 
        #   Face Detection algorithm
        # 
        # 
        results = faceDetection_model(frame)
        labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        
        #2. Plot boxes
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        
        for i in range(n):
            row = cordinates[i]
            if row[4] >= 0.5: 

                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
                text_d = faceDetection_classes[int(labels[i])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1) ## BBox
                cv2.putText(frame, text_d, (x1 , y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1 , cv2.LINE_AA)
        return frame
    

# ---------------------------------------------------------------------------------------------------------
# --------------------------------------- Face Recognition Functions --------------------------------------
# ---------------------------------------------------------------------------------------------------------


    def check_faceRecognition(self , status):
        if self.ret == True:
            if status:
                self.recognitionButton.setText("Stop Face Recognition")
                self.faceRecognition_Enabled = True
            else :
                self.recognitionButton.setText("Face Recognition")
                self.faceRecognition_Enabled = False

    def recognize_face(self , frame):
        # frameS = cv2.resize(frame, (0,0), None, 0.25, 0.25)
        # frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)
        # facesCurFrame = face_recognition.face_locations(frameS)
        # encodesCurFrame = face_recognition.face_encodings(frameS, facesCurFrame)

        # for encodeFace,faceLoc in zip(encodesCurFrame, facesCurFrame):
        #     matches = face_recognition.compare_faces(encodingListKnown, encodeFace)
        #     faceDis = face_recognition.face_distance(encodingListKnown, encodeFace)

            
        #     matchIndex = np.argmin(faceDis)
            
        #     if faceDis[matchIndex] > 0.6:
        #         name = 'unknown'

        #     else:
        #         name = classNames[matchIndex].upper()

        #     y1,x2,y2,x1 = faceLoc
        #     y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
        #     cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        #     cv2.rectangle(frame, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
        #     cv2.putText(frame, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.7 ,(255,255,255), 1)

        return frame

# ---------------------------------------------------------------------------------------------------------
# --------------------------------------- License Plate Detection Functions -------------------------------
# ---------------------------------------------------------------------------------------------------------


    def check_licenseDetection(self, status):
        if self.ret == True:
            if status:
                self.licenseButton.setText("Stop License Detection")
                self.licenseDetection_Enabled = True
            else :
                self.licenseButton.setText("License Detection")
                self.licenseDetection_Enabled = False

    

    def detect_license(self, frame):
        # 
        # 
        # License Detection algorithm
        # 
        # 


        results = license_model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        

        for i in range(n):
            row = cord[i]
            if row[4] >= 0.5: 
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
                text_d = license_classes[int(labels[i])]

                coords = [x1,y1,x2,y2]

                xmin, ymin, xmax, ymax = coords
        
                nplate = frame[int(ymin)+10:int(ymax), int(xmin)-10:int(xmax)+10]
                
            
                kernel = np.ones((1, 1), np.uint8)
                dilated = cv2.dilate(nplate, kernel, iterations=1)
                eroded = cv2.erode(dilated, kernel, iterations=1)
                ocr_result = ocr.ocr(nplate, det=False, cls=True)
                

                rectangle_size = nplate.shape[0]*nplate.shape[1]
        
                plate = [] 
                print(ocr_result)
                for result in ocr_result:
            
                    if result[0][1] >= OCR_TH and result[0][0] != "مصر":
                        plate.append(result[0][0].replace(":",""))
                plate = plate[::-1]
                plate = [" ".join(plate)]
                print(plate)
                plates.append(plate)

                text = plate


                if len(text) ==1:
                    text = text[0].upper()

                plate_num = text

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
                cv2.rectangle(frame, (x1-30, y1-30), (x2+50, y1), (0, 255,0), -1) ## for text label background
                reshaped_text = arabic_reshaper.reshape(plate_num)
                bidi_text = get_display(reshaped_text)
                fontpath = "E:\Studying\ASU\Senior-2\Graduation Project\GUI\License Detection/arial.ttf"
                font = ImageFont.truetype(fontpath, 24)
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                draw.text((x1, y1-30),bidi_text, font = font, fill="black")
                frame = np.array(img_pil)

        return frame
    

# ---------------------------------------------------------------------------------------------------------
# --------------------------------------- Car Dedtection Functions ----------------------------------------
# ---------------------------------------------------------------------------------------------------------


    def check_carDetection(self, status):
        if self.ret == True:
            if status:
                self.carButton.setText("Stop Car Detection")
                self.carDetection_Enabled = True
            else :
                self.carButton.setText("Car Detection")
                self.carDetection_Enabled = False

    def detect_car(self, frame):
        # 
        # 
        # Car Detection algorithm
        # 
        # 
        return frame



    

    def stop_cam(self):
        if self.ret == True:
            self.ret = False
            self.timer.stop()
            self.imgLabel.clear()
            self.capture.release()


    def check_pause(self, status):
        self.error.setText("")
        if status:
            self.pauseButton.setText("Resume")
            self.ret = False
            self.timer.stop()
        else :
            self.pauseButton.setText("Pause")
            self.start_cam()



    def gotoHome(self):
        msg = QMessageBox()
        msg.setWindowTitle("Exiting")
        msg.setText("Back to Home page!")
        msg.setIcon(QMessageBox.Question)

        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No )
        choice = msg.exec_()

        if choice == QMessageBox.Yes:
            if self.ret == True:
                self.stop_cam()

            Welcome = WelcomeScreen()
            widget.addWidget(Welcome)
            widget.setCurrentIndex(widget.currentIndex() + 1)
        else:
            pass

        



   



#main
app=QApplication(sys.argv)
# Welcome=WelcomeScreen()
Welcome = Camera()
widget = QtWidgets.QStackedWidget()
widget.addWidget(Welcome)
widget.setMinimumHeight(600)
widget.setMinimumWidth(1000)
widget.setWindowTitle('Intelligent video Surveillance')
widget.show()
try:
    sys.exit(app.exec_())
except:
    print("Exiting")
