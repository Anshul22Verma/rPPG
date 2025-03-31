import cv2
import h5py
import numpy as np
import pandas as pd
from scipy.signal import welch

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


def read_csv_gt(csv_path: str):
    df = pd.read_csv(csv_path)
    return np.array(df["bpm"].values.tolist())


# HR calculation based on ground truth label
def get_hr(y, sr, min=30, max=180):
    p, q = welch(y, sr, nfft=1e5 / sr, nperseg=np.min((len(y) - 1, 256)))
    return p[(p > min / 60) & (p < max / 60)][np.argmax(q[(p > min / 60) & (p < max / 60)])] * 60
