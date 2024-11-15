import cv2
import numpy as np

def track_doggo(video_frames, first_box):
    params = {
        'search_margin': 50,
        'min_confidence': 0.1,
        'hist_bins': 48, 
        'learning_rate': 0.01,   
        'smooth_factor': 0.5,   
        'size_change_thresh': 2.0, 
        'min_size_ratio': 0.9,    
        'max_size_ratio': 1.0,    
        'size_momentum': 0.1      
    }
    
    # Initialize box
    x = first_box[1]
    y = first_box[0]
    w = first_box[3] - first_box[1]
    h = first_box[2] - first_box[0]
    init_size = np.array([h, w])
    min_size = init_size * params['min_size_ratio']
    max_size = init_size * params['max_size_ratio']
    aspect_ratio = w/h
    bbox = (x, y, w, h)
    
    # Initialize tracker
    tracker = cv2.TrackerMIL.create()
    first_frame = video_frames[0]
    ok = tracker.init(first_frame, bbox)
    
    # Color model
    roi = first_frame[first_box[0]:first_box[2], first_box[1]:first_box[3]]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, 
                           [params['hist_bins']]*2, [0, 180, 0, 256])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    
    predicted_boxes = []
    prev_box = first_box
    current_size = init_size.copy()
    size_velocity = np.zeros(2)
    
    for frame in video_frames:
        ok, bbox = tracker.update(frame)
        
        if ok:
            x, y, w, h = [int(v) for v in bbox]
            new_size = np.array([h, w])
            
            # Smooth size changes with momentum
            size_velocity = params['size_momentum'] * size_velocity + \
                          (1 - params['size_momentum']) * (new_size - current_size)
            current_size = current_size + size_velocity
            
            # Enforce size limits
            current_size = np.clip(current_size, min_size, max_size)
            h, w = map(int, current_size)
            
            # Maintain aspect ratio
            if abs(w/h - aspect_ratio) > 0.1:
                if w/h > aspect_ratio:
                    w = int(h * aspect_ratio)
                else:
                    h = int(w / aspect_ratio)
            
            # Create prediction box
            pred_box = (
                max(0, y),
                max(0, x),
                min(frame.shape[0], y + h),
                min(frame.shape[1], x + w)
            )
            
            prev_box = pred_box
        else:
            pred_box = prev_box
            
        predicted_boxes.append(pred_box)
    
    return predicted_boxes