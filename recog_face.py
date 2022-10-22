import os, sys, pathlib, time, re, glob, math
import numpy as np
import cv2


FIGLOC = "./faceImages/"
########################################################################################################
def main():
    imgPath = FIGLOC + "3.jpeg"
    verbosity = 1 

    img = cv2.imread(imgPath)
    img = cv2.resize(img, (700, int(700*len(img)/len(img[0]))))
    cv2.imshow("img", img); cv2.moveWindow("img", 0, 0)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("imgGray", imgGray); cv2.moveWindow("imgGray", 100, 200)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
    imgFaceDet = img.copy()

    faceSel = []
    for (x, y, w, h) in faces:
        selected = True
        for (x_, y_, w_, h_) in faceSel:
            if x == x_ or y == y_: selected = False
        if selected == True: faceSel.append((x, y, w, h))
    for (x, y, w, h) in faceSel: cv2.rectangle(imgFaceDet, (x, y), (x+w, y+h), (225, 0, 0), 2)
    cv2.imshow("imgFaceDet", imgFaceDet); cv2.moveWindow("imgFaceDet", 200, 400) 
 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

########################################################################################################
if __name__ == "__main__": main()




















