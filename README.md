# Projects using OpenCV
Since these are projects using OpenCV:

    pip3 install opencv-python
    
The following codes  The code runs on python3:

**`carPlateRecog_OpenCV.py`**: takes in an image that has car plates (PATH to change directly in the code), uses the OpenCV edge finder to capture rectangular structures in the image that resembles a car plate, and applies  `pytesseract` to read out the plate numbers (note that `pytesseract` has its own installation procedure).

**`paperRecogCam_OpenCV.py`**: read from the live PC camera, uses the OpenCV edge finder to capture rectangular structures in the video image, and applies `pytesseract` to read out the text within the rectangule if resolvable.

**`faceRecog_OpenCV.py`**: takes in an image that has human faces and uses OpenCV's convinient `CascadeClassifier` class for facial recognition.
    
**`faceRecogCam_OpenCV.py`**: read from the live PC camera and uses OpenCV's convinient `CascadeClassifier` class for facial recognition.
    
References:
There are many resources online and only main references are listed:
- Murtaza's Workshop - Robotics and AI's Youtube Channel (2020) (<a href="https://www.youtube.com/watch?v=WQeoO7MI0Bs">Youtube</a>)
    





