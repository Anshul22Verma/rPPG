import cv2
import numpy as np
from math import ceil


def face_detection(frame, backend, use_larger_box=True, larger_box_coef=1.0):
    """Face detection on a single frame.

    Args:
        frame(np.array): a single frame.
        backend(str): backend to utilize for face detection.
        use_larger_box(bool): whether to use a larger bounding box on face detection.
        larger_box_coef(float): Coef. of larger box.
    Returns:
        face_box_coor(List[int]): coordinates of face bouding box.
    """
    if backend == "HC":
        # Use OpenCV's Haar Cascade algorithm implementation for face detection
        # This should only utilize the CPU
        detector = cv2.CascadeClassifier('/home/ubuntu/rPPG/rPPG_pipe/haarcascade_frontalface_default.xml')

        # Computed face_zone(s) are in the form [x_coord, y_coord, width, height]
        # (x,y) corresponds to the top-left corner of the zone to define using
        # the computed width and height.
        face_zone = detector.detectMultiScale(frame)

        if len(face_zone) < 1:
            print("ERROR: No Face Detected")
            face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
        elif len(face_zone) >= 2:
            # Find the index of the largest face zone
            # The face zones are boxes, so the width and height are the same
            max_width_index = np.argmax(face_zone[:, 2])  # Index of maximum width
            face_box_coor = face_zone[max_width_index]
            print("Warning: More than one faces are detected. Only cropping the biggest one.")
        else:
            face_box_coor = face_zone[0]
    elif backend == "RF":
        # Use a TensorFlow-based RetinaFace implementation for face detection
        # This utilizes both the CPU and GPU
        # res = RetinaFace.detect_faces(frame)

        # if len(res) > 0:
        #     # Pick the highest score
        #     highest_score_face = max(res.values(), key=lambda x: x['score'])
        #     face_zone = highest_score_face['facial_area']

        #     # This implementation of RetinaFace returns a face_zone in the
        #     # form [x_min, y_min, x_max, y_max] that corresponds to the 
        #     # corners of a face zone
        #     x_min, y_min, x_max, y_max = face_zone

        #     # Convert to this toolbox's expected format
        #     # Expected format: [x_coord, y_coord, width, height]
        #     x = x_min
        #     y = y_min
        #     width = x_max - x_min
        #     height = y_max - y_min

        #     # Find the center of the face zone
        #     center_x = x + width // 2
        #     center_y = y + height // 2
            
        #     # Determine the size of the square (use the maximum of width and height)
        #     square_size = max(width, height)
            
        #     # Calculate the new coordinates for a square face zone
        #     new_x = center_x - (square_size // 2)
        #     new_y = center_y - (square_size // 2)
        #     face_box_coor = [new_x, new_y, square_size, square_size]
        # else:
        #     print("ERROR: No Face Detected")
        #     face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
        raise ValueError("RF not supported yet, for face detection backend!")
    else:
        raise ValueError("Unsupported face detection backend!")
    
    if use_larger_box:
        face_box_coor[0] = max(0, face_box_coor[0] - (larger_box_coef - 1.0) / 2 * face_box_coor[2])
        face_box_coor[1] = max(0, face_box_coor[1] - (larger_box_coef - 1.0) / 2 * face_box_coor[3])
        face_box_coor[2] = larger_box_coef * face_box_coor[2]
        face_box_coor[3] = larger_box_coef * face_box_coor[3]
    return face_box_coor


def crop_face_resize(frames, backend, use_larger_box, larger_box_coef, 
                     detection_freq, width, height):
    """Crop face and resize frames.

    Args:
        frames(np.array): Video frames.
        
        detection_freq(int): The frequency of dynamic face detection e.g., every detection_freq frames.
        width(int): Target width for resizing.
        height(int): Target height for resizing.
        use_larger_box(bool): Whether enlarge the detected bouding box from face detection.
        use_face_detection(bool):  Whether crop the face.
        larger_box_coef(float): the coefficient of the larger region(height and weight),
                            the middle point of the detected region will stay still during the process of enlarging.
    Returns:
        resized_frames(list[np.array(float)]): Resized and cropped frames
    """
    # Face Cropping
    num_dynamic_det = ceil(frames.shape[0] / detection_freq)
    face_region_all = []
    # Perform face detection by num_dynamic_det" times.
    for idx in range(num_dynamic_det):
        face_region_all.append(face_detection(frames[detection_freq * idx], backend, use_larger_box, larger_box_coef))

    face_region_all = np.asarray(face_region_all, dtype='int')

    # Frame Resizing
    resized_frames = np.zeros((frames.shape[0], height, width, 3))
    for i in range(0, frames.shape[0]):
        frame = frames[i]
        # use the (i // detection_freq)-th facial region.
        reference_index = i // detection_freq
        face_region = face_region_all[reference_index]
        frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]
        resized_frames[i] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return resized_frames
