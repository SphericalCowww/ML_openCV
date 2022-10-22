import os, sys, pathlib, time, re, glob, math
import numpy as np
import cv2
import pytesseract


########################################################################################################
def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 500)
    cap.set(4, 300)
    cap.set(10, 130)

    if cap.isOpened() == True:
        while True:
            success, img = cap.read()
            if success == True:
                imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                imgFilt = cv2.bilateralFilter(imgGray, 15, 30, 30)
                imgEdge = cv2.Canny(imgFilt, 30, 200)
       
                imgCont, imgContSel = img.copy(), img.copy()
                contours, hierarchy = cv2.findContours(imgEdge.copy(), cv2.RETR_TREE,\
                                                       cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                contours = [contour for contour in contours if cv2.contourArea(contour) > 100.0]
                contoursSel, pointsSel = [], []
                cv2.drawContours(imgCont, contours, -1, (0, 255, 0), 2)
                for contour in contours:
                    perimeter   = cv2.arcLength(contour, True)
                    curveLength = cv2.arcLength(contour, False)
                    area        = cv2.contourArea(contour)
                    points = cv2.approxPolyDP(contour, 0.02*perimeter, True) 
                    if (len(points) == 4) and (perimeter/area < 1.0):
                        contoursSel.append(contour)
                        pointsSel.append(points)
        
                texts = []
                for contN, contour in enumerate(contoursSel):
                    mask = np.zeros(imgFilt.shape, np.uint8)
                    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1) #-1 thickness for filling
                    imgMasked = cv2.bitwise_and(imgFilt, imgFilt, mask=mask)
        
                    lowX  = np.min([point[0][0] for point in pointsSel[contN]])
                    highX = np.max([point[0][0] for point in pointsSel[contN]])
                    lowY  = np.min([point[0][1] for point in pointsSel[contN]])
                    highY = np.max([point[0][1] for point in pointsSel[contN]])
                    width  = highX - lowX
                    height = highY - lowY
                    lowX  = int(lowX  + 0.03*width); highX = int(highX - 0.03*width)
                    lowY  = int(lowY  + 0.1*height); highY = int(highY - 0.1*height)
                    imgMasked = imgMasked[lowY:highY, lowX:highX]
        
                    text = pytesseract.image_to_string(imgMasked, config="--psm 11")
                    text = text.replace("\n", "").replace("\x0c", "")
                    if len(text.replace(" ", "")) >= 1: texts.append(text)
                texts = list(set(texts))

                print(texts)
                for lineN, text in enumerate(texts):
                    cv2.putText(img, text, (0, 50+30*lineN), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),\
                                2, cv2.LINE_AA, False)
                cv2.drawContours(img, contoursSel, -1, (0, 255, 0), 2)
                cv2.imshow("img", img); cv2.moveWindow("img", 0, 0)
                time.sleep(0.5)
            else:
                print("Failing to read camera")
                if cap.isOpened() == False: break
                time.sleep(1) 
            if cv2.waitKey(1) != -1: break
    cap.release()
    cv2.destroyAllWindows()

########################################################################################################
if __name__ == "__main__": main()




















