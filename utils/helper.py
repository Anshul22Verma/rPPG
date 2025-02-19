import cv2
import h5py
import numpy as np


def read_vid(video_path: str):
    """Read video from a file

    Args:
        video_path(np.array): Path of video to load data from
    """
    frames = []
    VidObj = cv2.VideoCapture(video_path)
    VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
    success, frame = VidObj.read()
    while success:
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
        frame = np.asarray(frame)
        frames.append(frame)
        success, frame = VidObj.read()
    return np.asarray(frames)

def read_h5py_cohface_gt(h5py_path: str):
    """Read pulse from a file in COHFACE datase

    Args:
        h5py_path(np.array): Path of h5py label to load label from
    """
    h5py_file = h5py.File(h5py_path, "r")
    pulse = h5py_file["pulse"][:]
    h5py_file.close()
    return pulse
