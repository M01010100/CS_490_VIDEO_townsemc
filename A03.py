import cv2
import numpy as np

def track_doggo(video_frames, first_box):    
    params = {
        'search_margin': 40,
        'min_confidence': 0.1,
        'hist_bins': 32, 
        'learning_rate': 0.01,   
        'smooth_factor': 0.5,   
        'size_change_thresh': 0.9 
    }
    
    tracker = cv2.TrackerMIL.create()
    
    x = first_box[1]
    y = first_box[0]
    w = first_box[3] - first_box[1]
    h = first_box[2] - first_box[0]
    init_size = np.array([h, w])
    aspect_ratio = w/h
    bbox = (x, y, w, h)
    
    first_frame = video_frames[0]
    ok = tracker.init(first_frame, bbox)
    
    roi = first_frame[first_box[0]:first_box[2], first_box[1]:first_box[3]]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, 
                           [params['hist_bins']]*2, [0, 180, 0, 256])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    
    predicted_boxes = []
    prev_box = first_box
    center = np.array([(x+x+w)/2, (y+y+h)/2])
    velocity = np.zeros(2)
    
    for frame in video_frames:
        pred_center = center + velocity
        
        ok, bbox = tracker.update(frame)
        
        if ok:
            x, y, w, h = [int(v) for v in bbox]
            current_size = np.array([h, w])
            
            size_change = np.abs(current_size - init_size) / init_size
            if np.any(size_change > params['size_change_thresh']):
                if w/h > aspect_ratio:
                    w = int(h * aspect_ratio)
                else:
                    h = int(w / aspect_ratio)
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            back_proj = cv2.calcBackProject([hsv], [0, 1], roi_hist, 
                                          [0, 180, 0, 256], 1)
            
            roi_back_proj = back_proj[y:y+h, x:x+w]
            confidence = np.mean(roi_back_proj) / 255.0
            
            if confidence > params['min_confidence']:
                new_center = np.array([x + w/2, y + h/2])
                center = params['smooth_factor'] * center + \
                        (1 - params['smooth_factor']) * new_center
                
                velocity = new_center - center
                
                pred_box = (
                    max(0, int(center[1] - h/2)),
                    max(0, int(center[0] - w/2)),
                    min(frame.shape[0], int(center[1] + h/2)),
                    min(frame.shape[1], int(center[0] + w/2))
                )
                
                roi = frame[pred_box[0]:pred_box[2], pred_box[1]:pred_box[3]]
                if roi.size > 0:
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    temp_hist = cv2.calcHist([hsv_roi], [0, 1], None,
                                           [params['hist_bins']]*2, 
                                           [0, 180, 0, 256])
                    cv2.normalize(temp_hist, temp_hist, 0, 255, cv2.NORM_MINMAX)
                    roi_hist = (1-params['learning_rate'])*roi_hist + \
                              params['learning_rate']*temp_hist
                
                prev_box = pred_box
            else:
                pred_box = prev_box
        else:
            pred_box = prev_box
            tracker = cv2.TrackerMIL.create()
            tracker.init(frame, (prev_box[1], prev_box[0],
                               prev_box[3]-prev_box[1],
                               prev_box[2]-prev_box[0]))
        
        predicted_boxes.append(pred_box)
    return predicted_boxes