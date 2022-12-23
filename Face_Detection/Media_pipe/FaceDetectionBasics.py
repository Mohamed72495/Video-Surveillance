import cv2 as cv
import mediapipe as mp
import time
from moviepy.editor import VideoFileClip

#cap = cv.VideoCapture('Project-Videos-12.mp4')
#cap = cv.VideoCapture('M G ROAD CROWD WALKING _ STOCK FOOTAGES.mp4')
#cap = cv.VideoCapture(0)
cap = cv.VideoCapture('video_1.mp4')

pTime = 0

#use media pipe
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

img1 = cv.imread('image_1.jpg')
def image_face_detection(img):
    pTime = 0
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)
    if results.detections:
        for id, detection in enumerate(results.detections):

            #get the points of the bounded boxes
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv.rectangle(img, bbox, (170, 100, 10), 2)
            cv.putText(img, f'{int(detection.score[0] * 100)}%',
                        (bbox[0], bbox[1] - 20), cv.FONT_HERSHEY_PLAIN,
                        2, (170, 100, 10), 2)

    #caculate framed per second
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 2)

    cv.imshow("Image", img)
    cv.waitKey(0)
    #return img


def video_face_detection(video_path):
    pTime = 0
    cap = cv.VideoCapture(video_path)
    while True:
        success, img = cap.read()

        # convert to rgb
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = faceDetection.process(imgRGB)
        print(results)

        if results.detections:
            for id, detection in enumerate(results.detections):

                # get the points of the bounded boxes
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                cv.rectangle(img, bbox, (170, 100, 10), 2)
                cv.putText(img, f'{int(detection.score[0] * 100)}%',
                           (bbox[0], bbox[1] - 20), cv.FONT_HERSHEY_PLAIN,
                           2, (170, 100, 10), 2)

        # caculate framed per second
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN,
                   3, (0, 255, 0), 2)

        cv.imshow("Image", img)
        cv.waitKey(1)


def Create_Video(input_path,output_path):
    video_input = VideoFileClip(input_path)
    #video_input = video_input.subclip(0,30)
    processed_video = video_input.fl_image(image_face_detection)
    processed_video.write_videofile(output_path, audio=False)


video_face_detection('M G ROAD CROWD WALKING _ STOCK FOOTAGES.mp4')
#image_face_detection(img1)

#Create_Video('Project-Videos-12.mp4', 'output2.mp4')
