import numpy as np
import cv2
from enum import Enum 
class OPTICAL_FLOW(Enum): 
    HORN_SHUNCK = "horn_shunck" 
    LUCAS_KANADE = "lucas_kanade"
    
    
def compute_video_derivatives(video_frames, size):
    all_fx = []
    all_fy = []
    all_ft = []
    if size == 2:
        kfx = np.array([[-1, 1], [-1, 1]])
        kfy = np.array([[-1, -1], [1, 1]])
        kft1 = np.array([[-1, -1], [-1, -1]])
        kft2 = np.array([[1, 1], [1, 1]])
        scale_fx_fy = 4.0
        scale_ft = 4.0
        
    elif size == 3:
        kfx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kfy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        kft1 = np.array([[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]])
        kft2 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        scale_fx_fy = 8.0
        scale_ft = 16.0

    else:
        return None
    
    prev_frame = None
    for i in range(len(video_frames)):
            
            new_frame = video_frames[i]
            
            #Make current frame grayscale            
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY).astype("float64")
            new_frame /= 255.0
            
            print("apply the filers")
            
            #if previous frame(old_frame) not set, set it to current(now_frame)
            if prev_frame is None:
                prev_frame = new_frame
            
            fx = ((cv2.filter2D(prev_frame, cv2.CV_64F, kfx) + cv2.filter2D(new_frame, cv2.CV_64F, kfx))) / scale_fx_fy

            fy = ((cv2.filter2D(prev_frame, cv2.CV_64F, kfy) + cv2.filter2D(new_frame, cv2.CV_64F, kfy))) / scale_fx_fy

            ft = ((cv2.filter2D(prev_frame, cv2.CV_64F, kft1) + cv2.filter2D(new_frame, cv2.CV_64F, kft2))) / scale_ft
            
            
            all_fx.append(fx) 
            all_fy.append(fy)
            all_ft.append(ft)
            
            prev_frame = new_frame
    
    return all_fx, all_fy, all_ft

def compute_one_optical_flow_horn_shunck(fx, fy, ft, max_iter, max_error, weight=1.0):
    height, width = fx.shape
    u = np.zeros((height, width), dtype=np.float64)
    v = np.zeros((height, width), dtype=np.float64)

    kfx = np.array([[-1, 1]], dtype=np.float64)
    kfy = np.array([[-1], [1]], dtype=np.float64)
    
    lap_filter = np.array([[0, 0.25, 0],
                          [0.25,0,0.25],
                          [0,0.25,0]], dtype="float64")
    
    converged = False
    iterCount = 0
    lamb = weight
    print_inc = 5

    while not converged:
        # MAGIC      
        uav = cv2.filter2D(u, cv2.CV_64F, lap_filter)
        vav = cv2.filter2D(v, cv2.CV_64F, lap_filter)
        
        P = fx*uav + fy*vav + ft
        D = lamb + fx*fx + fy*fy
        
        PD = P/D
        
        u = uav - fx*PD
        v = vav - fy*PD
        
        ux = cv2.filter2D(u, cv2.CV_64F, kfx)
        uy = cv2.filter2D(u, cv2.CV_64F, kfy)
        vx = cv2.filter2D(v, cv2.CV_64F, kfx)
        vy = cv2.filter2D(v, cv2.CV_64F, kfy)
        
        error = lamb * np.mean(ux*ux + uy*uy + vx*vx + vy*vy)
        
        one_equation = (fx*u + fy*v + ft)*(fx*u + fy*v + ft)
              
        error += np.mean(one_equation)
        
        iterCount += 1
        
        if iterCount % print_inc == 0:
            print("ITERATION", iterCount, "DONE...")
        
        if error <= max_error:
            converged = True
            
        if iterCount >= max_iter:
            converged = True
   
    extra = np.zeros_like(u)
    combo = np.stack([u,v,extra], axis=-1)

    return combo, error, iterCount

def compute_optical_flow(video_frames, method=OPTICAL_FLOW.HORN_SHUNCK,
                         max_iter=10, max_error=1e-4, horn_weight=1.0, kanade_win_size=19):
    if method == OPTICAL_FLOW.HORN_SHUNCK:
        size = 2
        all_fx, all_fy, all_ft = compute_video_derivatives(video_frames, size)

        flow_frames = []
        for fx, fy, ft in zip(all_fx, all_fy, all_ft):
            flow, _, _ = compute_one_optical_flow_horn_shunck(fx, fy, ft, max_iter, max_error, horn_weight)
            flow_frames.append(flow)

        return flow_frames
    else:
        raise NotImplementedError("Lucas-Kanade method not implemented")


def main():      
    # Load video frames 
    video_filepath = "assign02/input/simple/image_%07d.png" 
    #video_filepath = "assign02/input/noice/image_%07d.png" 
    video_frames = A01.load_video_as_frames(video_filepath) 
     
    # Check if data is invalid 
    if video_frames is None: 
        print("ERROR: Could not open or find the video!") 
        exit(1) 
         
    # OPTIONAL: Only grab the first five frames 
    video_frames = video_frames[0:5] 
         
    # Calculate optical flow 
    flow_frames = compute_optical_flow(video_frames, method=OPTICAL_FLOW.HORN_SHUNCK) 
 
    # While not closed... 
    key = -1 
    ESC_KEY = 27 
    SPACE_KEY = 32 
    index = 0 
     
    while key != ESC_KEY: 
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
 
    # Destroy the windows     
    cv2.destroyAllWindows() 
     
if __name__ == "__main__":  
    main()