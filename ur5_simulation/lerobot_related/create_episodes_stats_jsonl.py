import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import pyarrow as pa
import pyarrow.parquet as pq
import json
import numpy as np
from pathlib import Path
from PIL import Image as PILImage
import cv2
import shutil

def estimate_num_samples(
    dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75
) -> int:
    """Heuristic to estimate the number of samples based on dataset size.
    The power controls the sample growth relative to dataset size.
    Lower the power for less number of samples.

    For default arguments, we have:
    - from 1 to ~500, num_samples=100
    - at 1000, num_samples=177
    - at 2000, num_samples=299
    - at 5000, num_samples=594
    - at 10000, num_samples=1000
    - at 20000, num_samples=1681
    """
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))

def load_image_as_numpy(
    fpath: str | Path, dtype: np.dtype = np.float32, channel_first: bool = True
) -> np.ndarray:
    img = PILImage.open(fpath).convert("RGB")
    img_array = np.array(img, dtype=dtype)
    if channel_first:  # (H, W, C) -> (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
    if np.issubdtype(dtype, np.floating):
        img_array /= 255.0
    return img_array

def sample_indices(data_len: int) -> list[int]:
    num_samples = estimate_num_samples(data_len)
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


def auto_downsample_height_width(img: np.ndarray, target_size: int = 150, max_size_threshold: int = 300):
    _, height, width = img.shape

    if max(width, height) < max_size_threshold:
        # no downsampling needed
        return img

    downsample_factor = int(width / target_size) if width > height else int(height / target_size)
    return img[:, ::downsample_factor, ::downsample_factor]


def sample_images(image_paths: list[str]) -> np.ndarray:
    sampled_indices = sample_indices(len(image_paths))

    images = None
    for i, idx in enumerate(sampled_indices):
        path = image_paths[idx]
        # we load as uint8 to reduce memory usage
        img = load_image_as_numpy(path, dtype=np.uint8, channel_first=True)
        img = auto_downsample_height_width(img)

        if images is None:
            images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

        images[i] = img

    return images


def get_feature_stats(array: np.ndarray, axis: tuple, keepdims: bool) -> dict[str, np.ndarray]:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims).tolist(),
        "max": np.max(array, axis=axis, keepdims=keepdims).tolist(),
        "mean": np.mean(array, axis=axis, keepdims=keepdims).tolist(),
        "std": np.std(array, axis=axis, keepdims=keepdims).tolist(),
        "count": np.array([len(array)]).tolist(),
    }


def get_feature_stats_img(array: np.ndarray, axis: tuple, keepdims: bool) -> dict[str, np.ndarray]:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([len(array)]),
    }

mypath = os.environ['HOME'] + '/training_data/lerobot/my_pusht/data/chunk-000'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles.sort()

jsonl_data = []

for file in onlyfiles:
    df = pd.read_parquet(mypath + '/' + file)
    print(f"file:{file} first index:{df['index'][0]}, last index:{df['index'][len(df)-1]}")
    episode_dic = {}
    episode_dic['episode_index'] = int(df['episode_index'][0])
    episode_dic['stats'] = {}

    # converting mp4 file into images and store it into a temporary folder
    observation_image_path = os.environ['HOME'] + '/training_data/lerobot/my_pusht/videos/chunk-000/observation.image/'
    video_path = observation_image_path + file.replace('parquet', '') + 'mp4'
    temp_img_path = observation_image_path + 'temp_imgs/'
    img_index = 0
    #video_data = [video_path + file.replace('parquet', '') + 'mp4']
    os.makedirs(temp_img_path)
    print(f"video_path:{video_path}")
    cap = cv2.VideoCapture(video_path) # says we capture an image from a webcam
    while(cap.isOpened()):
        ret,cv2_img = cap.read()
        if ret:
            #converted_img = cv2.cvtColor(cv2_img,cv2.COLOR_BGR2RGB)
            img_name = temp_img_path + f"img{img_index}.png"
            #print(f"ret:{ret}, img_name:{img_name}")
            cv2.imwrite(img_name, cv2_img)
            img_index += 1
        else:
            break
    cap.release()
    img_paths = [temp_img_path + f for f in listdir(temp_img_path) if isfile(join(temp_img_path, f))]
    #img_paths.sort()
    #print(f"len img_paths:{len(img_paths)}")
    ep_ft_array = sample_images(img_paths)
    #print(f"ep_ft_array.shape:{ep_ft_array.shape}, ep_ft_array.dim:{ep_ft_array.ndim}")
    temp_video_stats = get_feature_stats_img(ep_ft_array, axis=(0, 2, 3), keepdims=True)
    video_stats = {k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in temp_video_stats.items()}
    video_stats = {k: v.tolist() for k, v in video_stats.items()}
    episode_dic['stats']['observation.image'] = video_stats
    shutil.rmtree(temp_img_path)

    observation_state_list = []
    for i in range(len(df['observation.state'])):
        observation_state_list.append(df['observation.state'][i].tolist())
    episode_dic['stats']['observation.state'] = get_feature_stats(observation_state_list, axis=0, keepdims=0)

    action_list = []
    for i in range(len(df['action'])):
        action_list.append(df['action'][i].tolist())
    episode_dic['stats']['action'] = get_feature_stats(action_list, axis=0, keepdims=0)

    episode_dic['stats']['episode_index'] = get_feature_stats(df['episode_index'].to_numpy(), axis=0, keepdims=1)
    episode_dic['stats']['frame_index'] = get_feature_stats(df['frame_index'].to_numpy(), axis=0, keepdims=1)
    episode_dic['stats']['timestamp'] = get_feature_stats(df['timestamp'].to_numpy(), axis=0, keepdims=1)
    episode_dic['stats']['next.reward'] = get_feature_stats(df['next.reward'].to_numpy(), axis=0, keepdims=1)
    episode_dic['stats']['next.done'] = get_feature_stats(df['next.done'].to_numpy(), axis=0, keepdims=1)
    episode_dic['stats']['next.success'] = get_feature_stats(df['next.success'].to_numpy(), axis=0, keepdims=1)
    episode_dic['stats']['index'] = get_feature_stats(df['index'].to_numpy(), axis=0, keepdims=1)
    episode_dic['stats']['task_index'] = get_feature_stats(df['task_index'].to_numpy(), axis=0, keepdims=1)
    #print(episode_dic['stats']['task_index'])
    jsonl_data.append(episode_dic)
#'''

with open('episodes_stats.jsonl', 'w') as f:
    for l in jsonl_data:
        f.writelines([json.dumps(l)])
        f.writelines("\n")
