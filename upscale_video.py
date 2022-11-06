import os, sys, pathlib, time, re, glob, math
import numpy as np
import cv2
import moviepy.editor as mpe
from tqdm import tqdm


VIDLOC = "./sampleVideos/"
OUTLOC = "./output/"
########################################################################################################
# ref: learnopencv.com/super-resolution-in-opencv/
def main():
    verbosity = 1
    vidPath = VIDLOC + "/handOverYourFlesh.mp4"
    outPath = OUTLOC + "/out_handOverYourFlesh.mp4"
    windowWidth = 600
    showVidInProgress = False

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

    capVid = cv2.VideoCapture(vidPath)
    outVid = None
    vidFrameRate = capVid.get(cv2.CAP_PROP_FPS)
    vidTotFrameN = int(capVid.get(cv2.CAP_PROP_FRAME_COUNT))
    progressBar  = tqdm(total=vidTotFrameN)
    vidFormat    = cv2.VideoWriter_fourcc(*"mp4v")    
 
    windowSize, windowMoved = None, False 
    frameProcessed = 0
    if capVid.isOpened() == True:
        while capVid.isOpened():
            success, img = capVid.read()
            if success == True:
                imgUp = model.upsample(img)
                if windowMoved == False:
                    windowSize = (windowWidth, int((windowWidth/img.shape[1])*img.shape[0])) 
                    outVid = cv2.VideoWriter(outPath, vidFormat, vidFrameRate, windowSize)
                img   = cv2.resize(img,   windowSize)
                imgUp = cv2.resize(imgUp, windowSize)
                cv2.imshow("capVid", img)
                cv2.imshow("capVidUp", imgUp)
                if windowMoved == False: 
                    cv2.moveWindow("capVid",   0, 0)
                    cv2.moveWindow("capVidUp", 600, 0)
                    windowMoved = True
                outVid.write(imgUp)
                progressBar.update(1)
                frameProcessed += 1
            else:
                progressBar.close()
                if img is None: capVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else: raise IOError("main(): video capture failed.")
            if cv2.waitKey(1) != -1: 
                progressBar.close()
                break
            #if cv2.waitKey(25) & 0xFF == ord("q"): break   # press q to quit
    capVid.release() 
    outVid.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if verbosity >= 1: print("saving:", outPath)

    #adding audio
    outVid    = mpe.VideoFileClip(outPath)
    inAudio   = mpe.AudioFileClip(vidPath)
    outAudVid = outVid.set_audio(inAudio.subclip(0, frameProcessed/vidFrameRate))
    outAudVid.write_videofile(outPath.replace(".mp4", "_audio.mp4"), fps=vidFrameRate)
    if verbosity >= 1: print("saving:", outPath.replace(".mp4", "_audio.mp4"))

########################################################################################################
if __name__ == "__main__": main()




















