import cv2
import numpy as np

def main():
    image = np.zeros((480,640,3), dtype="uint8")
    
    state = np.array([100,100,3,3], dtype="float64")
    
    A = np.array([[1,0,1,0],
                  [0,1,0,1],
                  [0,0,1,0],
                  [0,0,0,1]], dtype="float64")
    
    kalman = cv2.KalmanFilter(4,2, type=cv2.CV_64F)
    kalman.measurementMatrix = np.array([[1,0,0,0],
                                         [0,1,0,0]], dtype="float64")
    kalman.transitionMatrix = A
    kalman.processNoiseCov = A*1e-4
    
    key = -1 
    ESC_KEY = 27
    while key != ESC_KEY:
        image[:,:,:] = 0
        pos = (int(state[0,0], int(state[1,0])))
        cv2.circle(image, pos, 4, (0,0,255), -1)
        #draw_logo(image,logo, pos[0],pos[1])
        pred = kalman.predict()
        kalman.correct(np.array([[state[0,0]],
                                 [state[0,1]]], dtype="float64"))
        
        pred_pos = (int(pred[0,0]), int(pred[1,0]))
        cv2.circle(image,pred_pos, 3, (0,255,0), -1)
                
        print("pred:", pred.shape)    
        
        cv2.imshow("image", image)
        key = cv2.waitKey(30)
        pred = np.transpose(np.matmul(A, np.transpose(state)))
        state = pred
        
        if state[0,0] < 0 or state[0,0] >= image.shape[1]:
            state[0,2] =  -state[0,2]
            
        if state[0,1] < 0 or state[0,1] >= image.shape[0]:
            state[0,3] =  -state[0,3]
            
if __name__ == "__main__": main()
