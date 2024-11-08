import cv2
import h5py
import numpy as np


def read_and_preprocess_video(video_file, dataset, pp_config):
    # load the video in (T, H, W, 3)
    vid = read_video(video_file=video_file, dataset=dataset)

    # preprocess the videos based on pre-process config

def read_video(video_file, dataset):
    match dataset:
        case "COHFACE":
            """Reads a video file, returns frames (T,H,W,3) """
            VidObj = cv2.VideoCapture(video_file)
            VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
            success, frame = VidObj.read()
            frames = list()
            while (success):
                frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
                frame = np.asarray(frame)
                frame[np.isnan(frame)] = 0  # TODO: maybe change into avg
                frames.append(frame)
                success, frame = VidObj.read()
            return np.asarray(frames)
        case "UBFCrPPG":
            """Reads a video file, returns frames(T, H, W, 3) """
            VidObj = cv2.VideoCapture(video_file)
            VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
            success, frame = VidObj.read()
            frames = list()
            while success:
                frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
                frame = np.asarray(frame)
                frames.append(frame)
                success, frame = VidObj.read()
            return np.asarray(frames)
        case "V4V":
            """Reads a video file, returns frames(T, H, W, 3) """
            VidObj = cv2.VideoCapture(video_file)
            VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
            success, frame = VidObj.read()
            frames = list()
            while success:
                frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
                frame = np.asarray(frame)
                frames.append(frame)
                success, frame = VidObj.read()
            return np.asarray(frames)



def read_gt(gt_file, dataset):
    match dataset:
        case "COHFACE":
            """Reads a bvp signal file."""
            f = h5py.File(gt_file, 'r')
            pulse = f["pulse"][:]
            return pulse
        case "UBFCrPPG":
            """Reads a bvp signal file."""
            with open(gt_file, "r") as f:
                str1 = f.read()
                str1 = str1.split("\n")
                bvp = [float(x) for x in str1[0].split()]
            return np.asarray(bvp)
        case "MAHNOB":
            """Read the BVP signal in a txt file."""
            return None
        case "V4V":
            """Read the BVP signal in a txt file."""
            return None


def extra_info(clip_path, gt_path, dataset):
    match dataset:
        case "COHFACE":
            """Gives additional information about the data sample like P-id, P-age, P-sex, and P-ethnicity."""
            return {}
        case "UBFCrPPG":
            """Gives additional information about the data sample like P-id, P-age, P-sex, and P-ethnicity."""
            return {}
        case "MAHNOB":
            """Gives additional information about the data sample like P-id, P-age, P-sex, and P-ethnicity."""
            return {}
        case "V4V":
            """Gives additional information about the data sample like P-id, P-age, P-sex, and P-ethnicity."""
            return {}
