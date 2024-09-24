import cv2
import shutil
import os
import sys
from pathlib import Path

def load_video_as_frames(video_filepath):
    capture = cv2.VideoCapture(video_filepath)
    if not capture.isOpened():
        print("ERROR: Could not open or find the video!")
        return None
    frames=[]
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)
    capture.release()
    return frames

def compute_wait(fps):
    return int(1000.0/fps)
    
def display_frames(all_frames, title, fps=30):
    wait = compute_wait(fps)
    for frame in all_frames:
        cv2.imshow(title, frame)
        if cv2.waitKey(wait):
            break

    cv2.destroyAllWindows()
    
def save_frames(all_frames, output_dir, basename, fps=30):
    video_folder = f"{basename}_{fps}"
    outputPath = os.path.join(output_dir, video_folder)
    
    if os.path.exists(outputPath):
        try:        
            shutil.rmtree(outputPath)
        except OSError as e:
            print(f"Warning: Could not remove existing directory {outputPath}. ({e})")
    
    os.makedirs(outputPath)
    
    for i, frame in enumerate(all_frames):
        filename = f"image_{i:07d}.png"
        full_path = os.path.join(outputPath, filename)
        cv2.imwrite(full_path, frame)
        
def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <video_filepath> <output_directory>")
        sys.exit(1)
    
    video_filepath = sys.argv[1]
    output_dir = sys.argv[2]
    
    basename = Path(video_filepath).stem
    
    all_frames = load_video_as_frames(video_filepath)
    if all_frames is None:
        print("Error: Failed to load frames.")
        sys.exit(1)
    
    display_frames(all_frames, title="Input Video", fps=30)
    
    save_frames(all_frames, output_dir, basename, fps=30)

if __name__ == "__main__":
    main()