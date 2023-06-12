import sys
import re
import os
import shutil
import openpyxl
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMessageBox, QMainWindow, QTableWidgetItem, QFileDialog
import face_recognition
from tracking_and_recognition import Tracker
import pandas as pd
import numpy as np
import sqlite3
from PyQt5.QtCore import QTimer
import cv2
from PyQt5.QtGui import QImage ,QPixmap
import time
from tkinter import messagebox

import torch
from PIL import ImageFont, ImageDraw, Image
import arabic_reshaper 
from bidi.algorithm import get_display
from paddleocr import PaddleOCR

from keras.models import load_model
# from tracker import *



# ---------------------------------------------------------------------------------------------------------
# --------------------------------------- Load License Detection Model ---------------------------------------
# ---------------------------------------------------------------------------------------------------------

OCR_TH = 0.5
ocr = PaddleOCR(lang="arabic", use_angle_cls=True)
try:
    license_model =  torch.hub.load('ultralytics/yolov5', 'custom', path= '.\License Detection Model/best.pt',force_reload=True) 
except:
    messagebox.showerror("Error", """Cannot Load License Detection Model.
Ensure that you have the model in '\License Detection Model' location.""")
    sys.exit()
license_classes = license_model.names 
plates = []


# ---------------------------------------------------------------------------------------------------------
# --------------------------------------- Load Face Detection Model ---------------------------------------
# ---------------------------------------------------------------------------------------------------------

try:
    faceDetection_model =  torch.hub.load('ultralytics/yolov5', 'custom', path= '.\Face Detection Model/best.pt',force_reload=True)
except:
    messagebox.showerror("Error", """Cannot Load Face Detection Model.
Ensure that you have the model in '\Face Detection Model' location.""")
    sys.exit()
faceDetection_classes = faceDetection_model.names

# ---------------------------------------------------------------------------------------------------------
# --------------------------------------- Load Face Recognition Model ---------------------------------------
# ---------------------------------------------------------------------------------------------------------

# try:
#     faceRecognition_model = load_model('.\Face Recognition Model/face recognisation4.h5')
# except:
#     messagebox.showerror("Error", """Cannot Load Face Recognition Model.
# Ensure that you have the model in '\Face Recognition Model' location.""")
#     sys.exit()
# dict = {"[0]": "Dina ", 
#         "[1]": "Said ",
#         "[2]": "Mohsen ",
#         "[3]": "Kly ",
#         "[4]": "Aya ",
#         "[5]": "Nada"}

# names_path = 'names.xlsx'
# weights_path = 'weights.xlsx'
# tracker = Tracker(names_path, weights_path)


# ---------------------------------------------------------------------------------------------------------
# --------------------------------------- Read Knownfaces Encodings ---------------------------------------
# ---------------------------------------------------------------------------------------------------------
    

    


# ---------------------------------------------------------------------------------------------------------
# --------------------------------------- Load Car Detection Model ---------------------------------------
# ---------------------------------------------------------------------------------------------------------

# vehicleDetection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True , force_reload=True)
# tracker = Tracker()

# ---------------------------------------------------------------------------------------------------------
# --------------------------------------- Welcome Screen ---------------------------------------
# ---------------------------------------------------------------------------------------------------------


class WelcomeScreen(QMainWindow):
    def __init__(self):
        super(WelcomeScreen,self).__init__()
        try:
            loadUi('WelcomeScreen.ui',self)
        except:
            messagebox.showerror("Error", "Some UI files not found")
            sys.exit()
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



# ---------------------------------------------------------------------------------------------------------
# --------------------------------------- Create Account Screen ---------------------------------------
# ---------------------------------------------------------------------------------------------------------



class CreateAccScreen(QMainWindow):
    def __init__(self):
        super(CreateAccScreen, self).__init__()
        try:
            loadUi("createacc.ui",self)
        except:
            messagebox.showerror("Error", "Some UI files not found")
            sys.exit()
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
            try:
                conn = sqlite3.connect("Database.db")
            except:
                messagebox.showerror("Error", "Some DB files not found")
                sys.exit()
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

   


# ---------------------------------------------------------------------------------------------------------
# --------------------------------------- Login Screen ---------------------------------------
# ---------------------------------------------------------------------------------------------------------
 

class LoginScreen(QMainWindow):
    def __init__(self):
        super(LoginScreen, self).__init__()
        try:
            loadUi("Login.ui",self)
        except:
            messagebox.showerror("Error", "Some UI files not found")
            sys.exit()
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
            try:
                conn = sqlite3.connect("Database.db")
            except:
                messagebox.showerror("Error", "Some DB files not found")
                sys.exit()
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
        camera = CameraSet()
        widget.addWidget(camera)
        widget.setCurrentIndex(widget.currentIndex()+1)



# ---------------------------------------------------------------------------------------------------------
# --------------------------------------- CameraSet Screen ---------------------------------------
# ---------------------------------------------------------------------------------------------------------

class CameraSet(QMainWindow):
    def __init__(self):
        super(CameraSet,self).__init__()
        try:
            loadUi('CameraSet.ui',self)
        except:
            messagebox.showerror("Error", "Some UI files not found")
            sys.exit()
        self.image=None

        # self.names_encodings = namesAndEncoding()
        # global classNames
        # global encodingListKnown
        try:
            df = pd.read_excel("./sample images for recognition/weights/names.xlsx" , header=None)
        except:
            messagebox.showerror("Error", """Cannot Read names of your sample known faces.
        Ensure that you have the names.xlsx file in
        '/sample images for recognition/weights' location.""")
            sys.exit()
        self.classNames = df.values.flatten()
        try:
            self.encodingListKnown = pd.read_excel("./sample images for recognition/weights/weights.xlsx" , header=None)
        except:
            messagebox.showerror("Error", """Cannot Read weights of your sample known faces.
        Ensure that you have the weights.xlsx file in
        '/sample images for recognition/weights' location""")
            sys.exit()
        self.encodingListKnown = [np.array(row) for _, row in self.encodingListKnown.iterrows()]

        self.startButton.clicked.connect(self.start_cam)
        self.stopButton.clicked.connect(self.stop_cam)

        self.pauseButton.setCheckable(True)
        self.pauseButton.clicked.connect(self.check_pause)

        self.Home.clicked.connect(self.gotoHome)
        self.Settings.clicked.connect(self.goToSettings)

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
        self.carDetection_Enabled = False

        self.checkBox.stateChanged.connect(self.checkBox_state)
        
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
                    try:
                        self.capture.open("rtsp://" + username + ":" + password + "@" +ip+ "/Streaming/channels/101")
                    except:
                        messagebox.showerror("Error", "cannot connect with your camera")
                        sys.exit()
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
            start_time = time.time()
            detected_image = self.detect_face(self.image)
            end_time = time.time()
            timeTaken = end_time - start_time
            print("Time taken: " , timeTaken , " seconds")
            self.displayImage(detected_image,1)
        else:
            self.displayImage(self.image,1)

        if self.faceRecognition_Enabled:
            start_time = time.time()
            detected_image = self.recognize_face(self.image)
            end_time = time.time()
            timeTaken = end_time - start_time
            print("Time taken: " , timeTaken , " seconds")
            self.displayImage(detected_image,1)
        else:
            self.displayImage(self.image,1)

        if self.licenseDetection_Enabled:
            start_time = time.time()
            detected_image = self.detect_license(self.image)
            end_time = time.time()
            timeTaken = end_time - start_time
            print("Time taken: " , timeTaken , " seconds")
            self.displayImage(detected_image,1)
        else:
            self.displayImage(self.image,1)

        if self.carDetection_Enabled:
            start_time = time.time()
            detected_image = self.vehicle_detection_and_tracking(self.image)
            end_time = time.time()
            timeTaken = end_time - start_time
            print("Time taken: " , timeTaken , " seconds")
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
# --------------------------------------- Face Detection Functions ---------------------------------------
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
        
        # results = faceDetection_model(frame)
        # labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    
        # #2. Plot boxes
        # n = len(labels)
        # x_shape, y_shape = frame.shape[1], frame.shape[0]
        
        # for i in range(n):
        #     row = cordinates[i]
        #     if row[4] >= 0.5: 

        #         x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
        #         #text_d = classes[int(labels[i])]
        #         cropped = frame[y1:y2, x1:x2]
        #         input_im = cv2.resize(cropped, (224, 224), interpolation = cv2.INTER_LINEAR)
        #         input_im = input_im / 255.
        #         input_im = input_im.reshape(1,224,224,3) 
        
        #         # Get Prediction
        #         res = np.argmax(faceRecognition_model.predict(input_im, 1, verbose = 0), axis=1)

        #         print(dict[str(res)])
        #         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3) ## BBox
        #         cv2.putText(frame, dict[str(res)], (x1 , y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2 , cv2.LINE_AA)


        
        frameS = cv2.resize(frame, (0,0), None, 0.25, 0.25)
        frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)
        results = faceDetection_model(frameS)
        
        labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        #cordinates --> (x-min, y-min, x-max, y-max)
        n = len(labels)
        x_shape, y_shape = frameS.shape[1], frameS.shape[0]

        cordinates = cordinates.clone()
        for row in cordinates:
            top = row[1] * y_shape      #top
            right = row[2] * x_shape    #right   
            bottom = row[3] * y_shape   #bottom
            left = row[0] * x_shape     #left
            row[0] = top
            row[1] = right
            row[2] = bottom
            row[3] = left

        faceLocations = [arr[:4] for arr in cordinates]
        #faceLocations --> (top, right, bottom, left)
        
        faceEncodings = face_recognition.face_encodings(frameS, faceLocations)
        
        for i,faceEncoding,faceLocation in zip(range(n) , faceEncodings, faceLocations):
            row_for_weight = cordinates[i]

            if row_for_weight[4] >= 0.5:
                y1,x2,y2,x1 = int(faceLocation[0]), int(faceLocation[1]), int(faceLocation[2]), int(faceLocation[3]) ## BBOx coordniates

                faceDis = face_recognition.face_distance(self.encodingListKnown, faceEncoding)
                
                matchIndex = np.argmin(faceDis)
                
                if faceDis[matchIndex] > 0.6:
                    name = 'unknown'
                else:
                    name = self.classNames[matchIndex].upper()

                y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.rectangle(frame, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
                cv2.putText(frame, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.7 ,(0,0,0), 1)

        # frameS = cv2.resize(frame, (0,0), None, 0.25, 0.25)
        # frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)
        # results = faceDetection_model(frameS)
            
        # labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        # #cordinates --> (x-min, y-min, x-max, y-max)
        # n = len(labels)
        # x_shape, y_shape = frameS.shape[1], frameS.shape[0]

        # cordinates = cordinates.clone()
        # for row in cordinates:
        #     if row[4] > 0.5:
        #         top = row[1] * y_shape      #top y1
        #         right = row[2] * x_shape    #right   x2
        #         bottom = row[3] * y_shape   #bottom y2
        #         left = row[0] * x_shape     #left x1
        #         row[0] = top
        #         row[1] = right
        #         row[2] = bottom
        #         row[3] = left
        #     else:
        #         del(row)

        # faceLocations = [arr[:4] for arr in cordinates]
        # #print(faceLocations)
        # faceEncodings = face_recognition.face_encodings(frameS, faceLocations)


        # tracks = tracker.update(faceLocations, faceEncodings) 
        # for track in tracks:
        #     y1,x2,y2,x1, id, name = track
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

    def vehicle_detection_and_tracking(self, frame):
        # 
        # 
        # Car Detection algorithm
        # 
        # 
        # area1 = [(10,590),(1522,590),(1530,560),(2,560)]
        # c = set()
        # C = set()

        # frame = cv2.resize(frame, (1530,800))
        # cv2.polylines(frame, [np.array(area1, np.int32)], True, (0,0,255), 4, -1)

        # results = vehicleDetection_model(frame)

        # points = []
        # for index, row in results.pandas().xyxy[0].iterrows():
        #     x1 = int(row['xmin'])
        #     y1 = int(row['ymin'])
        #     x2 = int(row['xmax'])
        #     y2 = int(row['ymax'])
            
        #     n = (row['name'])
        #     if 'car' in n:
        #         points.append([x1, y1, x2, y2])
            
        # boxes_id = tracker.update(points) 
        # for box_id in boxes_id:
        #     x, y, w, h, idd = box_id
        #     cv2.rectangle(frame, (x, y), (w, h), (255,30,0), 5)
        #     cv2.putText(frame,str(idd),((x-25),(y+35)),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),4)
        #     cv2.circle(frame, (w, h), 8, (255,255,255), -1)
        #     results1 = cv2.pointPolygonTest(np.array(area1, np.int32), (w, h), False)
        #     if results1>= 0:
        #         C.add(idd)
                
        # b = len(C)
        # cv2.rectangle(frame, (5, 5), (480, 65), (255, 255, 255), -1)
        # cv2.putText(frame, 'No. of Cars is ='+str(b), (15, 46), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
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


    def goToSettings(self):
        msg = QMessageBox()
        msg.setWindowTitle("Exiting")
        msg.setText("Go to Settings page!")
        msg.setIcon(QMessageBox.Question)

        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No )
        choice = msg.exec_()

        if choice == QMessageBox.Yes:
            if self.ret == True:
                self.stop_cam()

            settings = SettingsWindow()
            widget.addWidget(settings)
            widget.setCurrentIndex(widget.currentIndex() + 1)
        else:
            pass

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


    
# ---------------------------------------------------------------------------------------------------------
# --------------------------------------- Settings Screen ---------------------------------------
# ---------------------------------------------------------------------------------------------------------
 
images_path = './sample images for recognition/images'

class SettingsWindow(QMainWindow):

    def __init__(self):
        super(SettingsWindow,self).__init__()
        try:
            loadUi('Settings.ui',self)
        except:
            messagebox.showerror("Error", "Some UI files not found")
            sys.exit()

        # self.names_encodings = namesAndEncoding()
        self.images = []
        self.classNames = []
        self.ReloadNames.clicked.connect(self.getNames)
        self.CameraSet.clicked.connect(self.goToCameraSet)
        self.actionOpenPhoto.triggered.connect(self.openPhoto)


    def findEncodings(self):

        self.images.clear()
        self.classNames.clear()
        myList = os.listdir(images_path)
        pattern = re.compile(r'(\D+)')  
        

        for cl in myList:
            curImg = cv2.imread(f'{images_path}/{cl}')
            self.images.append(curImg)
            self.classNames.append(pattern.findall(os.path.splitext(cl)[0]))
        
        names_df = pd.DataFrame(self.classNames)
        names_df.to_excel("./sample images for recognition/weights/names.xlsx", index=False, header=False)
        encodingList = []
        for img in self.images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            face_encodings = face_recognition.face_encodings(img)
            if len(face_encodings) > 0:
                encode = face_encodings[0]
                encodingList.append(encode)
            else:
                encodingList.append([0])
        return encodingList 


    def getNames(self):
        # global classNames
        # global encodingListKnown

        encodingListKnown = self.findEncodings()

        weights_df = pd.DataFrame(encodingListKnown)
        weights_df.to_excel("./sample images for recognition/weights/weights.xlsx", index=False, header=False)
        # Load data from Excel file
        try:
            wb = openpyxl.load_workbook("./sample images for recognition/weights/names.xlsx")
        except:
            messagebox.showerror("Error", """Cannot Read names of your sample known faces.
        Ensure that you have the names.xlsx file in
        '/sample images for recognition/weights' location.""")
            sys.exit()
        
        ws = wb.active
        data = ws.values
        data_list = list(data)

        self.NamesTable.setRowCount(len(data_list))

        row = 0
        for row_data in data_list:
            col = 0
            for cell_data in row_data:
                item = QTableWidgetItem(str(cell_data))
                self.NamesTable.setItem(row, col, item)
                col += 1
            row += 1
        self.Count.setText("People: " + str(row))
        


    def openPhoto(self):

        # Show the file dialog to let the user choose an image file
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Image files (*.jpg *.jpeg *.png)")
        file_dialog.setViewMode(QFileDialog.Detail)

        if file_dialog.exec_() == QFileDialog.Accepted:
            # Get the path of the selected image file
            selected_image_path = file_dialog.selectedFiles()[0]
            # Specify the destination folder where the image file will be copied
            destination_folder = images_path

            # Create the destination folder if it doesn't exist
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)

            # Copy the selected image file to the destination folder
            shutil.copy(selected_image_path, destination_folder)
        else:
            # If the user canceled the file dialog, exit the program
            file_dialog.close()

    def goToCameraSet(self):
        msg = QMessageBox()
        msg.setWindowTitle("Exiting")
        msg.setText("Go to CameraSet page!")
        msg.setIcon(QMessageBox.Question)

        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No )
        choice = msg.exec_()

        if choice == QMessageBox.Yes:
            Camera = CameraSet()
            widget.addWidget(Camera)
            widget.setCurrentIndex(widget.currentIndex() + 1)
        else:
            pass

        



   



#main
app=QApplication(sys.argv)
# Welcome=WelcomeScreen()
Welcome = WelcomeScreen()
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
