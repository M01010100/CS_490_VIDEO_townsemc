import cv2
import numpy as np

def compute_optical_flow(video_frames, method):
    # Placeholder function for computing optical flow
    # Replace this with the actual implementation or import statement
    return [np.zeros_like(frame) for frame in video_frames]

def calculate_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    iou = intersection / union if union != 0 else 0
    return iou

def calculate_accuracy(pred_mask, true_mask):
    correct = (pred_mask == true_mask).sum()
    total = true_mask.size
    accuracy = correct / total
    return accuracy

def get_predicted_mask(flow):
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    threshold = 10  
    mask = mag > threshold

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    largest_label = np.argmax(stats[1:, -1]) + 1
    pred_mask = labels == largest_label

    return pred_mask

def track_doggo(video_frames, first_box):
    term_criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 5, 1)
    
    
    track_window = (
        first_box[1],                
        first_box[0],                
        first_box[3] - first_box[1], 
        first_box[2] - first_box[0]  
    )
    
    first_frame = video_frames[0]
    roi = first_frame[first_box[0]:first_box[2], first_box[1]:first_box[3]]
    
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [32], [0,180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    
    predicted_boxes = []
    
    for frame in video_frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
        
        mask = get_hue_mask(hsv)
        back_proj *= mask
        
        ret, track_window = cv2.CamShift(back_proj, track_window, term_criteria)
        
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        x = min(pts[:,0])
        y = min(pts[:,1]) 
        w = max(pts[:,0]) - x
        h = max(pts[:,1]) - y
        
        pred_box = (y, x, y+h, x+w)
        predicted_boxes.append(pred_box)
        
    return predicted_boxes

def get_hue_mask(hsv):

    hue = hsv[:,:,0]
    
    lower = 10
    upper = 30
    mask = cv2.inRange(hue, lower, upper)
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask