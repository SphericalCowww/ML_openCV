import os, sys, pathlib, time, re, glob, math
import numpy as np
import cv2
from tqdm import tqdm


VIDLOC = "./sampleVideos/"
########################################################################################################
# ref: learnopencv.com/super-resolution-in-opencv/
def main():
    verbosity = 1
    vidPath    = VIDLOC + "/handOverYourFlesh.mp4"
    #vidFormat  = cv2.VideoWriter_fourcc(*"mp4v")
    windowWidth = 600

    model = cv2.dnn_superres.DnnSuperResImpl_create()
    # source: github.com/Saafke/EDSR_Tensorflow/tree/master/models
    #model.readModel("upscaleModels/EDSR_x4.pb")
    #model.setModel("edsr", 4) 
    # source: github.com/fannymonori/TF-ESPCN/blob/master/export/ 
    #model.readModel("upscaleModels/ESPCN_x4.pb")
    #model.setModel("espcn", 4)
    # source: github.com/Saafke/FSRCNN_Tensorflow/tree/master/models
    model.readModel("upscaleModels/FSRCNN_x4.pb")
    model.setModel("fsrcnn", 4)
    # source: github.com/fannymonori/TF-LapSRN/tree/master/export
    #model.readModel("upscaleModels/LapSRN_x8.pb")
    #model.setModel("lapsrn", 8)

    cap = cv2.VideoCapture(vidPath)
    vidFrameRate = cap.get(cv2.CAP_PROP_FPS)
    totFrameN    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    windowSize, windowMoved = None, False 
    if cap.isOpened() == True:
        while cap.isOpened():
            success, img = cap.read()
            if success == True:
                imgUp = model.upsample(img)
                if windowMoved == False: 
                    windowSize = (windowWidth, int((windowWidth/img.shape[1])*img.shape[0]))
                img   = cv2.resize(img,   windowSize)
                imgUp = cv2.resize(imgUp, windowSize)
                cv2.imshow("cap", img)
                cv2.imshow("capUp", imgUp)
                if windowMoved == False: 
                    cv2.moveWindow("cap",   0, 0)
                    cv2.moveWindow("capUp", 600, 0)
                    windowMoved = True
            else:
                if img is None: cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else: raise IOError("main(): video capture failed.")
            if cv2.waitKey(1) != -1: break
            #if cv2.waitKey(25) & 0xFF == ord("q"): break   # press q to quit
    cap.release() 

    cv2.waitKey(0)
    cv2.destroyAllWindows()

########################################################################################################
if __name__ == "__main__": main()




















