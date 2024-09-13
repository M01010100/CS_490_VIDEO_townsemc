# MIT LICENSE
#
# Copyright 2024 Michael J. Reale
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

###############################################################################
# IMPORTS
###############################################################################

import sys
import numpy as np
import torch
import cv2
import pandas
import sklearn

def filterNeighborhood2D(image, kernal, crow, ccol):
    halfH = kernal.shape[0]//2
    halfW = kernal.shape[1]//2
    
    startOFFH = (1-kernal.shape[0]%2)
    startOFFW = (1 - kernal.shape[1]%2)
    
    endRow = crow + halfH
    endCol = ccol + halfW
    
    startRow = crow - halfH + startOFFH
    startCol = ccol = halfW + startOFFW
    
    
    clamp_startRow = max(0, startRow)
    clamp_startCol = max(0, startCol)
    neighborhood = image[clamp_startRow:(endRow+1), clamp_startCol:(endCol+1)]
    
    if startRow < 0:
        kernal = kernal[-startRow:]
    elif endRow > (image.shape[0]-1):
        off = image.shape[0] -1 - endRow
        kernal = kernal[0:(kernal.shape[0]-off)]
    if startCol < 0:
        kernal = kernal[:, -startCol:]
    elif endCol > (image.shape[1]-1):
        off = image.shape[1] - 1 - endCol
        kernal = kernal[:, 0:(kernal.shape[1]+off)]
    
    print("NEIGHBORHOOD: ", neighborhood.shape)
    print("KERNAL:", kernal.shape)
    
    value = kernal * neighborhood
    value = np.sum(value)
    
    return value

def filter2D(image, kernal):
    output = np.copy(image)
    
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            output[row,col] = filterNeighborhood2D(image, kernal, row, col)
    return output

def filter(video, kernal):
    output = np.copy(video)
    
    for t in range(video.shape[0]):
        for row in range(video.shape[1]):
            for col in range(video.shape[2]):

###############################################################################
# MAIN
###############################################################################

def main():        
    dummy_image = np.zeros((10,10), dtype="float64")
    dummy_filter= np.zeros((3,3), dtype="float64")
    dummy_output = filter2D(dummy_image, dummy_filter)
    
    ###############################################################################
    # PYTORCH
    ###############################################################################
    
    b = torch.rand(5,3)
    print(b)
    print("Torch CUDA?:", torch.cuda.is_available())
    
    ###############################################################################
    # PRINT OUT VERSIONS
    ###############################################################################

    print("Torch:", torch.__version__)
    print("Numpy:", np.__version__)
    print("OpenCV:", cv2.__version__)
    print("Pandas:", pandas.__version__)
    print("Scikit-Learn:", sklearn.__version__)
        
    ###############################################################################
    # OPENCV
    ###############################################################################
    if len(sys.argv) <= 1:
        # Webcam
        print("Opening webcam...")

        # Linux/Mac (or native Windows) with direct webcam connection
        capture = cv2.VideoCapture(0) #, cv2.CAP_DSHOW) # CAP_DSHOW recommended on Windows 
        # WSL: Use Yawcam to stream webcam on webserver
        # https://www.yawcam.com/download.php
        # Get local IP address and replace
        #IP_ADDRESS = "192.168.0.7"    
        #capture = cv2.VideoCapture("http://" + IP_ADDRESS + ":8081/video.mjpg")
        
        # Did we get it?
        if not capture.isOpened():
            print("ERROR: Cannot open capture!")
            exit(1)
            
        # Set window name
        windowName = "Webcam"
            
    else:
        # Trying to load video from argument

        # Get filename
        filename = sys.argv[1]
        
        # Load video
        capture = cv2.VideoCapture(filename)
        
        # Check if data is invalid
        if not capture.isOpened():
            print("ERROR: Could not open or find the video!")
            exit(1)

        # Set window name
        windowName = "Video"
        
    # Create window ahead of time
    cv2.namedWindow(windowName)
    
    # While not closed...
    key = -1
    prev_frame = None
    
    kfx = np.array([[-1, 1],
                    [-1, 1]], dtype="float64")
    
    while key == -1:
        # Get next frame from capture
        ret, frame = capture.read()
        
        if ret == True:        
            # Show the image
            cv2.imshow(windowName, frame)
            
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("float64")
            gray_image /= 255.0
            
            kernel_size = 17
            gray_image = cv2.GaussianBlur(gray_image, 
                                          ksize=(kernel_size, kernel_size),
                                          sigmaX=0)
            
            cv2.imshow("GRAY", gray_image)
            
            #fx = cv2.filter2D(gray_image, cv2.CV_64F, kfx)
            
            cv2.imshow("FX", np.absolute(fx)*4.0)
            
            if prev_frame is None:
                prev_frame = np.copy(gray_image)
            
            diff_image = gray_image - prev_frame
            diff_image = np.absolute(diff_image)
            
            cv2.imshow("DIFF", diff_image)
            
            
            prev_frame = np.copy(gray_image)    
            
            
        else:
            break

        # Wait 30 milliseconds, and grab any key presses
        key = cv2.waitKey(30)

    # Release the capture and destroy the window
    capture.release()
    cv2.destroyAllWindows()

    # Close down...
    print("Closing application...")

if __name__ == "__main__": 
    main()
    # The end