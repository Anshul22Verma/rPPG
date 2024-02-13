#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
import cv2
import csv
import pandas as pd





# Function to extract frames from video and save them as images
def extract_frames(video_path, output_dir, start_frame):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract frames
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"{start_frame + i:05d}.jpg")
        cv2.imwrite(frame_path, frame)
    
    cap.release()
    return frame_rate, total_frames




# Function to calculate average heart rate from gt_HR.csv file
def calculate_avg_heart_rate(gt_hr_path):
    df = pd.read_csv(gt_hr_path)
    avg_hr = round(df['HR'].mean(), 2)
    return avg_hr




# Main function
def process_directory(root_dir):
    total_csv_path = os.path.join(root_dir, 'total.csv')
    with open(total_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['VideoPath', 'StartFrame', 'FrameRate', 'AvgHeartRate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Variables to track total frames and start frame for each folder
        total_frames = 0
        start_frame = 0

        # Iterate through directories
        for root, dirs, files in os.walk(root_dir):
            if len(files) > 0:  # Check if there are files in the current directory
                unit_dir = root
                video_path = None
                wave_path = None
                gt_hr_path = None
                output_dir = None

                # Find video, wave, and gt_HR files in the current directory
                for file in files:
                    if file.endswith('.avi'):
                        video_path = os.path.join(unit_dir, file)
                    elif file == 'wave.csv':
                        wave_path = os.path.join(unit_dir, file)
                    elif file == 'gt_HR.csv':
                        gt_hr_path = os.path.join(unit_dir, file)

                if video_path is not None:
                    # Extract frames from video
                    output_dir = os.path.join(unit_dir, 'frames')
                    frame_rate, total_video_frames = extract_frames(video_path, output_dir, start_frame)
                   
                   # Increment total frames and update start frame for the next video
                    total_frames += total_video_frames
                    start_frame += total_video_frames
                    avg_heart_rate=None
                    if gt_hr_path is not None:
                        # Calculate average heart rate
                        avg_heart_rate = calculate_avg_heart_rate(gt_hr_path)
                    # Record total frames, frame rate, and start frame number in total.csv
                    writer.writerow({
                        'VideoPath': video_path,
                        'StartFrame': start_frame,
                        'FrameRate': frame_rate,
                        'AvgHeartRate': avg_heart_rate
                    })

        # Write the total number of frames for all videos
        #writer.writerow({'TotalFrames': total_frames})




# Example usage
root_directory = 'C:/Users/16475/PhysFormer-main/PhysFormer-main/vipl_hr_partial/p0'
process_directory(root_directory)






