import cv2
import h5py
import numpy as np

from caching import rPPGcache
from utils.helper import resample_ppg, get_config_keys, crop_face_resize
from utils.normalization import diff_normalize_data, diff_normalize_label, standardized_data, standardized_label


@rPPGcache
def preprocess_data(video_file, gt_file, dataset, pp_config):
    vid, gt = read_video(video_file, dataset), read_gt(gt_file, dataset)
    # load the video in (T, H, W, 3)
    target_length = vid.shape[0]
    bvps = resample_ppg(bvps, target_length)

    frames_clips, bvps_clips = preprocess(vid, bvps, pp_config)
    return frames_clips, bvps_clips


def chunk(frames, bvps, chunk_length):
    """
        Chunk the data into small chunks.
        
        Args:
            frames(np.array): video frames.
            bvps(np.array): blood volumne pulse (PPG) labels.
            chunk_length(int): the length of each chunk.
        Returns:
            frames_clips: all chunks of face cropped frames
            bvp_clips: all chunks of bvp frames
    """

    clip_num = frames.shape[0] // chunk_length
    frames_clips = [frames[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
    bvps_clips = [bvps[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
    return np.array(frames_clips), np.array(bvps_clips)


def preprocess(vid, bvps, config):
    """
        preprocess the videos based on pre-process config    
    """
    # resize frames and crop for face region
    vid = crop_face_resize(
        vid,
        get_config_keys(get_config_keys(config, "CROP_FACE"), "DO_CROP_FACE"),
        get_config_keys(get_config_keys(config, "CROP_FACE"), "BACKEND"),
        get_config_keys(get_config_keys(config, "CROP_FACE"), "USE_LARGE_FACE_BOX"),
        get_config_keys(get_config_keys(config, "CROP_FACE"), "LARGE_BOX_COEF"),
        get_config_keys(get_config_keys(config, "CROP_FACE"), "DO_DYNAMIC_DETECTION"),
        get_config_keys(get_config_keys(config, "CROP_FACE"), "DYNAMIC_DETECTION_FREQUENCY"),
        get_config_keys(get_config_keys(config, "CROP_FACE"), "USE_MEDIAN_FACE_BOX"),
        get_config_keys(get_config_keys(config, "RESIZE"), "H"),
        get_config_keys(get_config_keys(config, "RESIZE"), "W"),
    )

    # Check data transformation type
    data = list()  # Video data
    for data_type in get_config_keys(config, "DATA_TYPE"):
        f_c = vid.copy()
        if data_type == "Raw":
            data.append(f_c)
        elif data_type == "DiffNormalized":
            data.append(diff_normalize_data(f_c))
        elif data_type == "Standardized":
            data.append(standardized_data(f_c))
        else:
            raise ValueError("Unsupported data type!")
    
    # concatenate all channels
    data = np.concatenate(data, axis=-1)
    if get_config_keys(config, "LABEL_TYPE") == "Raw":
        pass
    elif get_config_keys(config, "LABEL_TYPE") == "DiffNormalized":
        bvps = diff_normalize_label(bvps)
    elif get_config_keys(config, "LABEL_TYPE") == "Standardized":
        bvps = standardized_label(bvps)
    else:
        raise ValueError("Unsupported label type!")

    if get_config_keys(config, "DO_CHUNK"):  # chunk data into snippets
        frames_clips, bvps_clips = chunk(data, bvps, 
                                         get_config_keys(config, "CHUNK_LENGTH"))
    else:
        frames_clips = np.array([data])
        bvps_clips = np.array([bvps])

    return frames_clips, bvps_clips


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
