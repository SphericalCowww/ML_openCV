import os, sys, pathlib, time, re, glob, math
import numpy as np
import cv2


FIGLOC = "./sampleImages/"
########################################################################################################
# ref: learnopencv.com/super-resolution-in-opencv/
def main():
    imgPath = FIGLOC + "566.png"
    verbosity = 1 
    
    model = cv2.dnn_superres.DnnSuperResImpl_create()
    # source: github.com/Saafke/EDSR_Tensorflow/tree/master/models
    #model.readModel("upscaleModels/EDSR_x4.pb")
    #model.setModel("edsr", 4) 
    # source: github.com/fannymonori/TF-ESPCN/blob/master/export/ 
    #model.readModel("upscaleModels/ESPCN_x4.pb")
    #model.setModel("espcn", 4)
    # source: github.com/Saafke/FSRCNN_Tensorflow/tree/master/models
    #model.readModel("upscaleModels/FSRCNN_x4.pb")
    #model.setModel("fsrcnn", 4)
    # source: github.com/fannymonori/TF-LapSRN/tree/master/export
    model.readModel("upscaleModels/LapSRN_x8.pb")
    model.setModel("lapsrn", 8)

    img = cv2.imread(imgPath)
    imgUpscaled = model.upsample(img)
    img = cv2.resize(img, (imgUpscaled.shape[1], imgUpscaled.shape[0])) 
    #imgUpscaled = cv2.resize(imgUpscaled, (img.shape[1], img.shape[0])) 
    cv2.imshow("img", img); cv2.moveWindow("img", 0, 0)
    cv2.imshow("imgUpscaled", imgUpscaled); cv2.moveWindow("imgUpscaled", 600, 0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
 

########################################################################################################
if __name__ == "__main__": main()




















