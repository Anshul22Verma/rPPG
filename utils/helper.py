from collections import defaultdict
import cv2
import h5py
import math
import numpy as np
from PIL import Image
import sys
import xml.etree.ElementTree as ET
from retinaface import RetinaFace   # Source code: https://github.com/serengil/retinaface

# from torchvideotransforms import video_transforms, volume_transforms

# sys.path.append(r"C:\Users\transponster\Documents\anshul\rPPG")
# from models.rhythmNet.video2st_maps import preprocess_video_to_st_maps
import logger
import logging

logrPPG = logging.getLogger(__name__)

def get_config_keys(config: dict, key: str, mandatory: bool = True):
    if key in config.keys():
        return config[key]
    else:
        if mandatory:
            logger.error(f"key {key} not in {config.keys()}")
            raise KeyError(f"Cant retrieve key {key} from config")    
        else:
            logger.warning(f"key {key} not in {config.keys()}")
            return None


def resample_ppg(input_signal, target_length):
    """
        Samples a PPG sequence into specific length.
        To resample the PPG signal into the length of videos
        # bvp values == # frames
    """
    return np.interp(
        np.linspace(
            1, input_signal.shape[0], target_length), np.linspace(
            1, input_signal.shape[0], input_signal.shape[0]), input_signal)

def etree_to_dict(tree: ET.Element) -> dict:
    d = {tree.tag: {} if tree.attrib else None}
    children = list(tree)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {tree.tag: {k: v[0] if len(v) == 1 else v
                        for k, v in dd.items()}}
    if tree.attrib:
        d[tree.tag].update(('@' + k, v)
                           for k, v in tree.attrib.items())
    if tree.text:
        text = tree.text.strip()
        if children or tree.attrib:
            if text:
                d[tree.tag]['#text'] = text
        else:
            d[tree.tag] = text
    return d


def read_xml(xml_file: str) -> dict:
    xml_tree = ET.parse(xml_file)
    return etree_to_dict(xml_tree.getroot())


def age(dob, today):
    years = today.year - dob.year
    if today.month < dob.month or (today.month == dob.month and today.day < dob.day):
        years -= 1
    return years


def read_hdf5(hdf5_file: str):
    h5py_f = h5py.File(hdf5_file)
    print(hdf5_file)
    return h5py_f


def face_detection(frame, backend, use_larger_box=False, larger_box_coef=1.0):
        """
            Face detection on a single frame.

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
            detector = cv2.CascadeClassifier(
            './dataset/haarcascade_frontalface_default.xml')

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
            res = RetinaFace.detect_faces(frame)

            if len(res) > 0:
                # Pick the highest score
                highest_score_face = max(res.values(), key=lambda x: x['score'])
                face_zone = highest_score_face['facial_area']

                # This implementation of RetinaFace returns a face_zone in the
                # form [x_min, y_min, x_max, y_max] that corresponds to the 
                # corners of a face zone
                x_min, y_min, x_max, y_max = face_zone

                # Convert to this toolbox's expected format
                # Expected format: [x_coord, y_coord, width, height]
                x = x_min
                y = y_min
                width = x_max - x_min
                height = y_max - y_min

                # Find the center of the face zone
                center_x = x + width // 2
                center_y = y + height // 2
                
                # Determine the size of the square (use the maximum of width and height)
                square_size = max(width, height)
                
                # Calculate the new coordinates for a square face zone
                new_x = center_x - (square_size // 2)
                new_y = center_y - (square_size // 2)
                face_box_coor = [new_x, new_y, square_size, square_size]
            else:
                print("ERROR: No Face Detected")
                face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
        else:
            raise ValueError("Unsupported face detection backend!")

        if use_larger_box:
            face_box_coor[0] = max(0, face_box_coor[0] - (larger_box_coef - 1.0) / 2 * face_box_coor[2])
            face_box_coor[1] = max(0, face_box_coor[1] - (larger_box_coef - 1.0) / 2 * face_box_coor[3])
            face_box_coor[2] = larger_box_coef * face_box_coor[2]
            face_box_coor[3] = larger_box_coef * face_box_coor[3]
        return face_box_coor


def crop_face_resize(frames, DO_CROP_FACE, BACKEND, USE_LARGE_FACE_BOX,
                     LARGE_BOX_COEF, DO_DYNAMIC_DETECTION, 
                     DYNAMIC_DETECTION_FREQUENCY, USE_MEDIAN_FACE_BOX,
                     RESIZE_W, RESIZE_H):
    """
        Perform face detection and frame related pre-processing

        Args:
            frames(np.array): Video frames.
            DO_CROP_FACE(bool):  Whether crop the face.
            BACKEND(str): Method to use for cropping faces out of frames.
            USE_LARGE_FACE_BOX(bool): Whether enlarge the detected bouding box from face detection.
            LARGE_BOX_COEF(float): the coefficient of the larger region(height and weight),
                                    the middle point of the detected region will stay still during the process of enlarging.
            
            DO_DYNAMIC_DETECTION(bool): If False, all the frames use the first frame's bouding box to crop the faces
                                        and resizing.
                                        If True, it performs face detection every "detection_freq" frames.
            DYNAMIC_DETECTION_FREQUENCY(int): The frequency of dynamic face detection e.g., every detection_freq frames.
            USE_MEDIAN_FACE_BOX(bool): If True, mean detection landmarks are used
            RESIZE_W(int): Target width for resizing.
            RESIZE_H(int): Target height for resizing.
        Returns:
            resized_frames(list[np.array(float)]): Resized and cropped frames
    """
    # Face Cropping
    if DO_DYNAMIC_DETECTION:
        num_dynamic_det = math.ceil(frames.shape[0] / DYNAMIC_DETECTION_FREQUENCY)
    else:
        num_dynamic_det = 1
    
    face_region_all = []
    # Perform face detection by num_dynamic_det" times.
    for idx in range(num_dynamic_det):
        if DO_CROP_FACE:
            face_region_all.append(face_detection(frames[DYNAMIC_DETECTION_FREQUENCY * idx], BACKEND,
                                                   USE_LARGE_FACE_BOX, LARGE_BOX_COEF))
        else:
            face_region_all.append([0, 0, frames.shape[1], frames.shape[2]])
    face_region_all = np.asarray(face_region_all, dtype='int')
    if USE_MEDIAN_FACE_BOX:
        # Generate a median bounding box based on all detected face regions
        face_region_median = np.median(face_region_all, axis=0).astype('int')
    
    # Frame Resizing
    resized_frames = np.zeros((frames.shape[0], RESIZE_H, RESIZE_W, 3))
    for i in range(0, frames.shape[0]):
        frame = frames[i]
        if DO_DYNAMIC_DETECTION:  # use the (i // detection_freq)-th facial region.
            reference_index = i // DYNAMIC_DETECTION_FREQUENCY
        else:  # use the first region obtrained from the first frame.
            reference_index = 0
        if DO_CROP_FACE:
            if USE_MEDIAN_FACE_BOX:
                face_region = face_region_median
            else:
                face_region = face_region_all[reference_index]
            frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                    max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]
        resized_frames[i] = cv2.resize(frame, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_AREA)
    return resized_frames
