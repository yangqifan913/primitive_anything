import numpy as np
import torch
from torch.utils.data import Dataset
import random
import os
from collections.abc import Sequence, Mapping
import glob
import sys
from tqdm import tqdm
from pathlib import Path
from lakefs_spec import LakeFSFileSystem
import zipfile
import shutil


def default_data_dir():
    import git
    repo_dir = git.Repo('.', search_parent_directories=True).working_dir
    return os.path.join(repo_dir, "data", "container-segmentation")

def unzip_all_files(directory):
    main_train_folder = os.path.join(directory, 'train')
    main_val_folder = os.path.join(directory, 'val')
    if os.path.exists(main_train_folder) and os.path.exists(main_val_folder):
        print(f"Data exists.")
        return
    else:
        if os.path.exists(main_train_folder):
            print(f"Folder {main_train_folder} exists. Deleting it...")
            shutil.rmtree(main_train_folder)
        os.makedirs(main_train_folder)
    
        if os.path.exists(main_val_folder):
            print(f"Folder {main_val_folder} exists. Deleting it...")
            shutil.rmtree(main_val_folder)
        os.makedirs(main_val_folder)
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith('.zip'):
            try:
                folder_name = os.path.splitext(filename)[0]
                folder_path = os.path.join(directory, folder_name)

                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(folder_path)
                    print(f"Unzipped: {filename}")
                    
                for root, dirs, files in os.walk(folder_path):
                    if 'train' in dirs:
                        train_folder_path = os.path.join(root, 'train')
                        for file in os.listdir(train_folder_path):
                            src_file = os.path.join(train_folder_path, file)
                            dst_file = os.path.join(main_train_folder, file)
                            shutil.move(src_file, dst_file)
                    if 'val' in dirs:
                        val_folder_path = os.path.join(root, 'val')
                        for file in os.listdir(val_folder_path):
                            src_file = os.path.join(val_folder_path, file)
                            dst_file = os.path.join(main_val_folder, file)
                            shutil.move(src_file, dst_file)
                shutil.rmtree(folder_path)  
            except zipfile.BadZipFile:
                print(f"Error: {filename} is not a valid zip file")

def download_from_lakefs(uri, local_path):
    if not uri.startswith('lakefs://'):
        raise ValueError(f'Invalid lakefs uri: {uri}')
    repo, branch = uri.replace('lakefs://', '').split('/')
    
    fs = LakeFSFileSystem()
    
    print('Downloading dataset...')
    for obj in tqdm(fs.ls(f'lakefs://{repo}/{branch}/'), file=sys.stdout):
        name_splits = obj['name'].split('/')
        repo, branch, filepath = name_splits[0], name_splits[1], '/'.join(name_splits[2:])
        target_path = os.path.join(local_path, filepath)
        fs.get_file(f'lakefs://{repo}/{branch}/{filepath}', target_path)

    unzip_all_files(local_path)

class GridSample(object):
    def __init__(
        self,
        grid_size=0.05,
        mode="train",
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec
        if mode == "train":
            self.keys=("coord", "segment")
        elif mode == "test":
            self.keys=("coord", )

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        idx_select = (
            np.cumsum(np.insert(count, 0, 0)[0:-1])
            + np.random.randint(0, count.max(), count.size) % count
        )
        idx_unique = idx_sort[idx_select]
        data_dict["grid_coord"] = grid_coord[idx_unique]
        for key in self.keys:
            data_dict[key] = data_dict[key][idx_unique]
        return data_dict
    
    @staticmethod
    def fnv_hash_vec(arr):
        assert arr.ndim == 2
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr
            
class ToTensor(object):
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        elif isinstance(data, Mapping):
            result = {sub_key: self(item) for sub_key, item in data.items()}
            return result
        elif isinstance(data, Sequence):
            result = [self(item) for item in data]
            return result
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")

class Collect(object):
    def __init__(self, keys, offset_keys_dict=None, **kwargs):
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="coord")
        self.keys = keys
        self.offset_keys = offset_keys_dict
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            data[key] = data_dict[key]
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            if len(keys) > 3: 
                keys = [keys]
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        return data

class RandomCrop(object):
    def __init__(self, point_max=80000):
        self.point_max = point_max

    def __call__(self, data_dict):
        point_max = self.point_max
        assert "coord" in data_dict.keys()
        if data_dict["coord"].shape[0] > point_max:
            center = data_dict["coord"][
                np.random.randint(data_dict["coord"].shape[0])
            ]
            idx_crop = np.argsort(np.sum(np.square(data_dict["coord"] - center), 1))[
                :point_max
            ]
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx_crop]
            if "grid_coord" in data_dict.keys():
                data_dict["grid_coord"] = data_dict["grid_coord"][idx_crop]
            if "segment" in data_dict.keys():
                data_dict["segment"] = data_dict["segment"][idx_crop]
        return data_dict
        
class RandomDropout(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data_dict):
        if random.random() < self.dropout_application_ratio:
            n = len(data_dict["coord"])
            idx = np.random.choice(n, int(n * (1 - self.dropout_ratio)), replace=False)
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx]
            if "segment" in data_dict.keys():
                data_dict["segment"] = data_dict["segment"][idx]
        return data_dict      
        
class SegmentDataset(Dataset):
    def __init__(
        self,
        split="train",
        data_root="data/container",
        grid_size = 0.05,
        point_max = 50000,
        dropout_ratio = 0.3,
    ):
        super(SegmentDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.data_list = self.get_data_list()
        self.grid = GridSample(grid_size=grid_size, mode='train' if split != 'test' else 'test')
        self.totensor = ToTensor()
        self.collect = Collect(keys=("coord", "grid_coord", "segment") if split != 'test' else ("coord", "grid_coord"),
            feat_keys=("coord"),)
        self.crop = RandomCrop(point_max=point_max)
        self.drop = RandomDropout(dropout_ratio=dropout_ratio)
    def get_data_list(self):
        data_list = glob.glob(os.path.join(self.data_root, self.split, "*.npy"))
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        data = np.load(data_path)
        coord = data[:,:3]
        if self.split != "test":
            segment = data[:,-1].astype(np.int64)
            data_dict = dict(
                coord=coord,
                segment=segment,
                name=self.get_data_name(idx) 
            )
        else:
            data_dict = dict(
                coord=coord,
                name=self.get_data_name(idx) 
            )
        return data_dict
        
    def prepare_train_data(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.drop(data_dict)
        data_dict = self.grid(data_dict)
        data_dict = self.crop(data_dict)
        data_dict = self.totensor(data_dict)
        data_dict = self.collect(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.grid(data_dict)
        data_dict = self.totensor(data_dict)
        data_dict = self.collect(data_dict)
        return data_dict
        
    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]
        
    def __getitem__(self, idx):
        if self.split == "test":
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list)      