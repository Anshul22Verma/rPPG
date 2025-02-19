import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset

from utils.helper import read_vid, read_h5py_cohface_gt
from preprocess.face_detection import crop_face_resize
from preprocess.video import chunk, diff_normalize_data, diff_normalize_label, standardized_data, resample_ppg


class TSCanDataset(Dataset):
    def __init__(self, patient_session_ids: list, 
                 root_loc: str, cache_dir: str, 
                 width: int, height: int, chunk_length: int):
        """
            patients_idxs: list of patients in the dataset
            patients_session_ids: list of patient-session unique identifier
            root_loc: root location of the COHFACE dataset
        """
        self.root_loc = root_loc
        self.patient_session_ids = patient_session_ids
        self.cache_dir = cache_dir

        self.backend = "HC"
        self.use_larger_box = True
        self.larger_box_coef=1.0
        self.detection_freq=20
        self.width = width
        self.height = height
        self.chunk_length = chunk_length
    
    def __len__(self):
        """Return the length of the dataset."""
        return len(self.patient_session_ids)
    
    def __getitem__(self, idx):
        patient_session_id = self.patient_session_ids[idx]

        # Check if cached processed data and ground truth exist
        cached_data_path = os.path.join(self.cache_dir, f"{patient_session_id}_data.npy")
        cached_gt_path = os.path.join(self.cache_dir, f"{patient_session_id}_gt.npy")
        if os.path.exists(cached_data_path) and os.path.exists(cached_gt_path):
            # load the data and gt from cache
            data = np.load(cached_data_path)
            gt = np.load(cached_gt_path)
        else:
            # if cache doesn't exist then load the dataset and cache the dataset
            p, s = patient_session_id.split("_")
            video_path = os.path.join(self.root_loc, p, s, "data.avi")
            gt_path = os.path.join(self.root_loc, p, s, "data.hdf5")

            data = read_vid(video_path=video_path)
            gt = read_h5py_cohface_gt(h5py_path=gt_path)
            gt = resample_ppg(gt, target_length=data.shape[0])  # or len(data)
            data = crop_face_resize(data, backend=self.backend, use_larger_box=self.use_larger_box,
                                    larger_box_coef=self.larger_box_coef, detection_freq=self.detection_freq,
                                    width=self.width, height=self.height)
            
            # normalize frames 
            data_list = []
            # 3-channels for diffNormalized image
            f_c = data.copy()
            data_list.append(diff_normalize_data(data=f_c))
            # 3-channels of standardize data
            f_c = data.copy()
            data_list.append(standardized_data(data=f_c))
            # have a 6-channel video
            data = np.concatenate(data_list, axis=-1)

            # normalize the gt BVP values
            gt = diff_normalize_label(label=gt)

            # perform chunking
            data, gt = chunk(frames=data, bvps=gt, chunk_length=self.chunk_length)

            # save the cache in a numpy file
            np.save(cached_data_path, data)
            np.save(cached_gt_path, gt)
        
        num_chunks = data.shape[0]
        # select a random chunk
        random_chunk_idx = random.randint(0, num_chunks - 1)
        data = data[random_chunk_idx]
        gt = gt[random_chunk_idx]

        # data is in NDCHW format
        data = np.transpose(data, (0, 3, 1, 2))
        data = torch.tensor(data, dtype=torch.float32)
        gt = torch.tensor(gt, dtype=torch.float32)
        return data, gt, patient_session_id, random_chunk_idx
