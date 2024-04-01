import glob
import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm
import multiprocessing

sys.path.append("C:\\Users\\transponster\\Documents\\anshul\\rPPG")

from models.rhythmNet.video2st_maps import get_frames_and_video_meta_data, preprocess_video_to_st_maps, get_st_maps, \
    get_hr
import models.rhythmNet.config as config



def maps_parallel(file_path):
    # Your calculation logic here
    # Example: return parameter * 2
    save_path = config.ST_MAPS_PATH
    csv_path = config.SAVE_CSV_PATH
    # glob.glob(config.FACE_DATA_DIR + '/**/*avi')

    stacked_maps = preprocess_video_to_st_maps(video_path=file_path, output_shape=(180, 180),
                                               clip_size=config.CLIP_SIZE)  # 0.5 sec

    r = get_st_maps(file_path, clip_size=300, group_clip_size=15, frames_dim=(224, 224))  # 0.5 sec, 15 clip size
    hr = get_hr(video_path=file_path, sliding_window_stride=15, num_stl_maps=len(r))
    hr_stl_maps = []
    file_names = []
    for idx, (r_, hr_) in enumerate(zip(r, hr)):
        file_name = f"{file_path.split(os.sep)[-4]}_{file_path.split(os.sep)[-3]}_{file_path.split(os.sep)[-2]}_STL{idx}"
        # save_path = os.path.join(save_path, f"{file_name}.npy")
        np.save(f"{config.ST_MAPS_PATH}{file_name}.npy", r_)
        hr_stl_maps.append(hr_)
        file_names.append(f"{config.ST_MAPS_PATH}{file_name}.npy")

    # np.save(save_path, maps)
    df = pd.DataFrame()
    df["STL_map"] = file_names
    df["HR"] = hr_stl_maps
    df.to_csv(os.path.join(os.path.dirname(config.SAVE_CSV_PATH), "HR",
                           f"{file_path.split(os.sep)[-4]}_{file_path.split(os.sep)[-3]}_{file_path.split(os.sep)[-2]}.csv"),
              index=False)
    return None


if __name__ == "__main__":
    get_frames_and_video_meta_data(r'D:\anshul\remoteHR\VIPL-HR-V1\data\p104\v1\source2\video.avi',
                                   num_frames=300, frames_dim=(224, 224), sliding_window_stride=15)
    # get_spatio_temporal_map()
    # get_spatio_temporal_map_threaded_wrapper()
    video_files = glob.glob(r'D:\anshul\remoteHR\VIPL-HR-V1\data\**\**\**\*.avi')
    # let's not include 'source4' because they contain NIR videos
    video_files = [f for f in video_files if 'source4' not in f]
    print(video_files)
    file_path = video_files[0]

    # for file_path in tqdm(video_files, total=len(video_files)):
    #     # [r'D:\anshul\remoteHR\VIPL-HR-V1\data\p104\v1\source2\video.avi']
    save_path = config.ST_MAPS_PATH
    csv_path = config.SAVE_CSV_PATH
    # glob.glob(config.FACE_DATA_DIR + '/**/*avi')

    stacked_maps = preprocess_video_to_st_maps(video_path=file_path, output_shape=(180, 180),
                                               clip_size=config.CLIP_SIZE)  # 0.5 sec

    r = get_st_maps(file_path, clip_size=850, group_clip_size=25, frames_dim=(440, 500))  # 0.5 sec, 15 clip size
    print(r.shape)
    import matplotlib.pyplot as plt
    for r_ in r:
        plt.imshow(r_)
        plt.waitforbuttonpress()
    hr = get_hr(video_path=file_path, sliding_window_stride=25, num_stl_maps=len(r))
    hr_stl_maps = []
    file_names = []
    for idx, (r_, hr_) in enumerate(zip(r, hr)):
        file_name = f"{file_path.split(os.sep)[-4]}_{file_path.split(os.sep)[-3]}_{file_path.split(os.sep)[-2]}_STL{idx}"
        # save_path = os.path.join(save_path, f"{file_name}.npy")
        np.save(f"{config.ST_MAPS_PATH}{file_name}.npy", r_)
        hr_stl_maps.append(hr_)
        file_names.append(f"{config.ST_MAPS_PATH}{file_name}.npy")

    # np.save(save_path, maps)
    df = pd.DataFrame()
    df["STL_map"] = file_names
    df["HR"] = hr_stl_maps
    df.to_csv(os.path.join(os.path.dirname(config.SAVE_CSV_PATH),
                           f"{file_path.split(os.sep)[-4]}_{file_path.split(os.sep)[-3]}_{file_path.split(os.sep)[-2]}.csv"),
              index=False)
    # offset = 1  # Adjust as needed
    # pool = multiprocessing.Pool(8)  # Create a pool with 4 processes
    # results = pool.map(maps_parallel, video_files)
