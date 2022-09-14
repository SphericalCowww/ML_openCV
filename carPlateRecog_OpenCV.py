import os, sys, pathlib, time, re, glob, math
import numpy as np
import cv2
import pytesseract


FIGLOC = "./carPlateImages/"
########################################################################################################
def main():
    imgPath = FIGLOC + "1.jpg"
    verbosity = 1 

    img = cv2.imread(imgPath)
    img = cv2.resize(img, (700, int(700*len(img)/len(img[0]))))
    cv2.imshow("img", img); cv2.moveWindow("img", 0, 0)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("imgGray", imgGray); cv2.moveWindow("imgGray", 100, 200)
   
    imgFilt = imgGray.copy()
    #imgFilt = cv2.GaussianBlur(imgFilt, (3, 5), cv2.BORDER_DEFAULT) 
    imgFilt = cv2.bilateralFilter(imgFilt, 15, 30, 30)
    #_, imgFilt = cv2.threshold(imgFilt, 150, 225, cv2.THRESH_BINARY)#+cv2.THRESH_OTSU)
    #imgFilt = cv2.adaptiveThreshold(imgFilt, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,35,1)
    cv2.imshow("imgFilt", imgFilt); cv2.moveWindow("imgFilt", 200, 400) 

    imgEdge = cv2.Canny(imgFilt, 30, 200)
    #imgEdge = cv2.Sobel(imgFilt, cv2.CV_8U, 1, 0, ksize=3) 
    #element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(20, 20))
    #imgEdge = cv2.morphologyEx(src=imgEdge, op=cv2.MORPH_CLOSE, kernel=element)
    #imgEdge = cv2.Canny(imgEdge, 30, 200)
    #imgEdge = cv2.dilate(imgEdge, np.ones((3, 5)), iterations=1)
    #imgEdge = cv2.erode (imgEdge, np.ones((3, 5)), iterations=1)
    cv2.imshow("imgEdge", imgEdge); cv2.moveWindow("imgEdge", 300, 0)

    imgCont, imgContSel = img.copy(), img.copy()
    contours, hierarchy = cv2.findContours(imgEdge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 100.0]
    contoursSel, pointsSel = [], []
    cv2.drawContours(imgCont, contours, -1, (0, 255, 0), 2)
    cv2.imshow("imgCont", imgCont); cv2.moveWindow("imgCont", 400, 200)
    for contour in contours:
        perimeter   = cv2.arcLength(contour, True)
        curveLength = cv2.arcLength(contour, False)
        area        = cv2.contourArea(contour)
        points = cv2.approxPolyDP(contour, 0.02*perimeter, True) 
        if (len(points) == 4) and (perimeter/area < 1.0):
            contoursSel.append(contour)
            pointsSel.append(points)
    cv2.drawContours(imgContSel, contoursSel, -1, (0, 255, 0), 2)
    cv2.imshow("imgContSel", imgContSel); cv2.moveWindow("imgContSel", 500, 400)
    if verbosity >= 1: print("Number of contours selected:", len(contoursSel))

    contSelPlate, imgMaskedPlate, plateNames = [], [], []
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
        #imgMasked = cv2.adaptiveThreshold(imgMasked, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        #                                  cv2.THRESH_BINARY, 35, 1)
        plateName = pytesseract.image_to_string(imgMasked, config="--psm 11")
        plateName = plateName.replace("\n", "").replace("\x0c", "")
        if len(plateName.replace(" ", "")) >= 1:
            imgMasked = cv2.resize(imgMasked, (400, int(400*(highY-lowY)/(highX - lowX))))
            contSelPlate  .append(contour)
            imgMaskedPlate.append(imgMasked)
            plateNames    .append(plateName)
            cv2.imshow(plateName, imgMaskedPlate[-1]) 
            cv2.moveWindow(plateName, 1000, 100*(len(plateNames)-1))
            if verbosity >= 1: print(repr(plateName))
        #cv2.imshow(str(contN), imgMasked); cv2.moveWindow(str(contN), 900, 100*(contN-1)+400) 
    cv2.drawContours(imgContSel, contSelPlate, -1, (225, 0, 0), 3)
    cv2.imshow("imgContSel", imgContSel); cv2.moveWindow("imgContSel", 500, 400)
    if verbosity >= 1: print("Number of plates found:     ", len(plateNames))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

########################################################################################################
if __name__ == "__main__": main()




















