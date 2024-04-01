from collections import defaultdict
import cv2
import h5py
import numpy as np
from PIL import Image
import sys
import xml.etree.ElementTree as ET
from torchvideotransforms import video_transforms, volume_transforms


sys.path.append(r"C:\Users\transponster\Documents\anshul\rPPG")
from models.rhythmNet.video2st_maps import preprocess_video_to_st_maps


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


def load_sample(f_path: str, th: int = 300, from_fps: int = 20, to_fps: int = 20) -> list[Image]:
    """
    :param f_path: a video with .avi extension
    :return: a list of PIL images, of shape frames X H X W X C
    """
    cap = cv2.VideoCapture(f_path)

    frames = []
    last_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            last_frame = Image.fromarray(frame)
            frames.append(Image.fromarray(frame))
        else:
            break
    cap.release()
    print(f"Number of frames before changing fps: {len(frames)}")
    cap.set(int(cap.get(int(cv2.CAP_PROP_FPS))), int(to_fps))
    print(f"From FPS: {cap.get(cv2.CAP_PROP_FPS)} -> {from_fps}")
    frames = []
    last_frame = None
    counter = 1
    from_fps = int(cap.get(int(cv2.CAP_PROP_FPS)))

    frames_to_drop = []
    if from_fps != to_fps:
        frames_to_drop = [int(f) for f in np.linspace(1, from_fps - 1, from_fps - to_fps)]
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if counter % from_fps not in frames_to_drop:
                last_frame = Image.fromarray(frame)
                frames.append(Image.fromarray(frame))
        else:
            break
        counter += 1
    print(f"Number of frames after changing fps: {len(frames)}")
    cap.release()
    # there are some videos with 294 frames some others might have more so just adding this for caution
    if len(frames) > th:
        frames = frames[:th]
    while len(frames) != th:
        frames.append(last_frame)

    print(f"Checking {len(frames)}")
    # frames = [frame for i, frame in enumerate(frames) if i % 10 == sampling]
    return frames


def load_stl_map_sample(f_path: str, th: int = 300, group_clip_size: int = 15,
                        frames_dim: tuple = (224, 224), from_fps: int = 20, to_fps: int = 20) -> list[Image]:
    """
    :param f_path: a video with .avi extension
    :return: a list of PIL images, of stacked_maps X H X W X C
    """
    cap = cv2.VideoCapture(f_path)

    frames = []
    last_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            last_frame = Image.fromarray(frame)
            frames.append(Image.fromarray(frame))
        else:
            break
    # there are some videos with 294 frames some others might have more so just adding this for caution
    if len(frames) > th:
        frames = frames[:th]
    while len(frames) != th:
        frames.append(last_frame)
    cap.release()
    # frames = [frame for i, frame in enumerate(frames) if i % 10 == sampling]
    stacked_maps = preprocess_video_to_st_maps(video_path=f_path, clip_size=th, group_clip_size=group_clip_size,
                                               frames_dim=(224, 224))  # 0.5 sec
    return stacked_maps


def get_transforms(train: bool = True, mean: tuple = None, std: tuple = None) -> video_transforms.Compose:
    if train:
        transforms__ = [
            video_transforms.Resize((224, 224)),
            # video_transforms.RandomGrayscale(p=0.4),
            # video_transforms.RandomHorizontalFlip(p=0.3),
            # video_transforms.RandomRotation(degrees=15),
            # video_transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            # volume_transforms.ClipToTensor()  # this scales the image as well /255
        ]
        if mean is None or std is None:
            transforms__.append(
                volume_transforms.ClipToTensor()
            )
        else:
            transforms__.append(
                # video_transforms.Normalize(mean=mean, std=std),
                volume_transforms.ClipToTensor(div_255=False)
            )
    else:
        transforms__ = [
            video_transforms.Resize((224, 224)),
        ]
        if mean is None or std is None:
            transforms__.append(
                volume_transforms.ClipToTensor()
            )
        else:
            transforms__.append(
                # video_transforms.Normalize(mean=mean, std=std),
                volume_transforms.ClipToTensor(div_255=False)
            )
    return video_transforms.Compose(transforms__)
