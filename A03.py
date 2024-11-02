import cv2
import numpy as np

def compute_optical_flow(video_frames, method):
    # Placeholder function for computing optical flow
    # Replace this with the actual implementation or import statement
    return [np.zeros_like(frame) for frame in video_frames]

def track_doggo(video_frames, first_box):
    # Calculate optical flow 
    flow_frames = compute_optical_flow(video_frames, method='HORN_SHUNCK') 

    # While not closed... 
    key = -1 
    ESC_KEY = 27 
    SPACE_KEY = 32 
    index = 0 
    
    # Get the current image and flow image 
    image = video_frames[index] 
    flow = flow_frames[index] 
     
    flow = np.absolute(flow) 
     
    # Show the images 
    cv2.imshow("ORIGINAL", image) 
    cv2.imshow("FLOW", flow) 
         
    # Wait 30 milliseconds, and grab any key presses 
    key = cv2.waitKey(30) 

    # If space, move forward 
    if key == SPACE_KEY: 
        index += 1 
        if index >= len(video_frames): 
            index = 0 

    return first_box