import os, sys, pathlib, time, re, glob, math
import numpy as np
import cv2


########################################################################################################
def main():
    verbosity = 1 

    cap = cv2.VideoCapture(0)
    cap.set(3, 500)
    cap.set(4, 300)
    cap.set(10, 130)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    while True:
        success, img = cap.read()
        #cv2.imshow("img", img); cv2.moveWindow("img", 0, 0)

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
        imgFaceDet = img.copy()
        faceSel = []
        for (x, y, w, h) in faces:
            selected = True
            for (x_, y_, w_, h_) in faceSel:
                if x == x_ or y == y_: selected = False
            if selected == True: faceSel.append((x, y, w, h))
        for (x, y, w, h) in faceSel: cv2.rectangle(imgFaceDet, (x, y), (x+w, y+h), (225, 0, 0), 2)
        cv2.imshow("imgFaceDet", imgFaceDet); cv2.moveWindow("imgFaceDet", 0, 0)

        if cv2.waitKey(1) != -1: break

    cv2.destroyAllWindows()

########################################################################################################
if __name__ == "__main__": main()




















