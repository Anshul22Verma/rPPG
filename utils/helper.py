from collections import defaultdict
import cv2
import h5py
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from torchvideotransforms import video_transforms, volume_transforms


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


def load_sample(f_path: str, th: int = 300) -> list[Image]:
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
    # there are some videos with 294 frames some others might have more so just adding this for caution
    if len(frames) > th:
        frames = frames[:th]
    while len(frames) != th:
        frames.append(last_frame)
    cap.release()
    # frames = [frame for i, frame in enumerate(frames) if i % 10 == sampling]
    return frames


def get_transforms(train: bool = True, mean: tuple = None, std: tuple = None) -> video_transforms.Compose:
    if train:
        transforms__ = [
            video_transforms.Resize((224, 224)),
            video_transforms.RandomGrayscale(p=0.4),
            video_transforms.RandomHorizontalFlip(p=0.3),
            video_transforms.RandomRotation(degrees=15),
            video_transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            # volume_transforms.ClipToTensor()  # this scales the image as well /255
        ]
        if mean is None or std is None:
            transforms__.append(
                volume_transforms.ClipToTensor()
            )
        else:
            transforms__.append(
                video_transforms.Normalize(mean=mean, std=std),
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
                video_transforms.Normalize(mean=mean, std=std),
                volume_transforms.ClipToTensor(div_255=False)
            )
    return video_transforms.Compose(transforms__)