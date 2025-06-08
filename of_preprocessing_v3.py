import cv2
import numpy as np
import os
import pandas as pd
import time
import gc  # Importing garbage collection module

# Function to process each frame
import cv2
import numpy as np

start_time = time.time()

def process_frame(frame, prev_frame, flow_type, feature_params, lk_params, first_frame):
    # Handle case when frame is None
    if frame is None:
        return None, None, first_frame

    # Ensure frame is in BGR format
    if len(frame.shape) == 2 or frame.shape[2] == 1:  # Grayscale or single channel
        gray = frame  # Already grayscale
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    similarity_score = None
    if flow_type == 'sparse':
        p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
        if p0 is not None:
            # Handle case when prev_frame is None or in a different format
            if first_frame or prev_frame is None:
                prev_gray = gray  # Use current frame as previous if prev_frame is None
                first_frame = False
            else:
                if len(prev_frame.shape) == 2 or prev_frame.shape[2] == 1:
                    prev_gray = prev_frame
                else:
                    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)


            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            if len(good_new) > 0:
                flow = good_new - good_old
                similarity_score = np.mean(np.linalg.norm(flow, axis=1))
    else:  # 'dense'
        # Same check for prev_frame
        if len(prev_frame.shape) == 2 or prev_frame.shape[2] == 1:
            flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        else:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        similarity_score = np.mean(flow)

    return gray, similarity_score, first_frame


# Function to process frames from a folder
def process_frames_from_folder(folder_path, flow_type):
    valid_extensions = ('.jpg', '.png')
    feature_params = dict(maxCorners=150, qualityLevel=0.01, minDistance=10, blockSize=7)
    lk_params = dict(winSize=(25, 25), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    similarity_scores = []
    first_frame = True
    prev_frame = None

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(valid_extensions):
            frame_path = os.path.join(folder_path, filename)
            frame = cv2.imread(frame_path)
            prev_frame, similarity_score, first_frame = process_frame(frame, prev_frame, flow_type, feature_params, lk_params, first_frame)
            similarity_scores.append(similarity_score)
            del frame  # Free memory
            gc.collect()  # Force garbage collection

    return similarity_scores

def count_images_in_folder(folder_path):
    image_extensions = ['.png', '.jpg', '.jpeg']  # Add or remove extensions as needed
    return sum(1 for filename in os.listdir(folder_path) 
               if os.path.isfile(os.path.join(folder_path, filename)) 
               and filename.lower().endswith(tuple(image_extensions)))

# Main function
folder_path = 'C:/Users/anton/Desktop/GNN/data/Heico/frames/heico_sig_1'
flow_type = 'sparse'  # Options: 'dense' or 'sparse'
similarity_scores = process_frames_from_folder(folder_path, flow_type)
frame_count = count_images_in_folder(folder_path)



def output_to_excel(similarity_scores, folder_path, frame_count, processing_time):
    # Ensure similarity_scores length matches the number of frames
    if len(similarity_scores) != frame_count:
        # Adjust the length of similarity_scores to match frame_count
        similarity_scores += [None] * (frame_count - len(similarity_scores))
    folder_name = os.path.basename(folder_path)
    output_path = f'{folder_name}_optical_flow_scores.xlsx'
    df = pd.DataFrame(similarity_scores, columns=['Optical Flow Score'])
    df['Frame Number'] = range(1, frame_count + 1)
    # Create a list with the same processing time for each frame
    processing_times = [processing_time] * frame_count
    df['Processing Time (seconds)'] = processing_times
    df.to_excel(output_path, index=False)

# Main function
# folder_path = 'C:/Users/anton/Desktop/GNN/data/HeiChole/frames/hei_chole17'
# frames = read_frames_from_folder(folder_path)
# frame_count = len(frames)
# flow_type = 'sparse'  # Options are 'dense' or 'sparse' which are Farneback and Lucas-Kanade respectively

#if flow_type == 'sparse':
#    similarity_scores = calculate_sparse_optical_flow(frames)
#else:
#    similarity_scores = calculate_dense_optical_flow(frames)
end_time = time.time()

processing_time = end_time - start_time
output_to_excel(similarity_scores, folder_path, frame_count, processing_time)
