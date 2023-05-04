import pandas as pd
import os
import cv2
import face_recognition
import numpy as np
import re
path = './sample images for recognition/images'
images = []
classNames = []
myList = os.listdir(path)
pattern = re.compile(r'(\D+)')  

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(pattern.findall(os.path.splitext(cl)[0]))
    
names_df = pd.DataFrame(classNames)
names_df.to_excel("./sample images for recognition/weights/names.xlsx", index=False, header=False)

# read = pd.read_excel("names.xlsx" , header=None)
# my_list = read.values.flatten()

def findEncodings(images):
    encodingList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        face_encodings = face_recognition.face_encodings(img)
        if len(face_encodings) > 0:
            encode = face_encodings[0]
            encodingList.append(encode)
        else:
            encodingList.append([0])
    return encodingList

# my_list = list(range(1, 129))

encodingListKnown = findEncodings(images)

weights_df = pd.DataFrame(encodingListKnown)
weights_df.to_excel("./sample images for recognition/weights/weights.xlsx", index=False, header=False)

# read = pd.read_excel("weights.xlsx" , header=None)
# my_list = [np.array(row) for _, row in read.iterrows()]
 
