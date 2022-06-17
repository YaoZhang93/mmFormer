import os
import torch
from torch.utils.data import Dataset
import random
# random.seed(1000)
import numpy as np
# np.random.seed(1000)
from torchvision.transforms import transforms
import pickle
from scipy import ndimage
import glob


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


class MaxMinNormalization(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        Max = np.max(image)
        Min = np.min(image)
        image = (image - Min) / (Max - Min)

        return {'image': image, 'label': label}


class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = np.flip(image, 0)
            label = np.flip(label, 0)
        if random.random() < 0.5:
            image = np.flip(image, 1)
            label = np.flip(label, 1)
        if random.random() < 0.5:
            image = np.flip(image, 2)
            label = np.flip(label, 2)

        return {'image': image, 'label': label}


class Random_Crop(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        H = random.randint(0, 240 - 128)
        W = random.randint(0, 240 - 128)
        D = random.randint(0, 160 - 128)

        image = image[H: H + 128, W: W + 128, D: D + 128, ...]
        label = label[..., H: H + 128, W: W + 128, D: D + 128]

        return {'image': image, 'label': label}


class Random_intencity_shift(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        label = sample['label']

        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[1, image.shape[1], 1, image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

        image = image*scale_factor+shift_factor

        return {'image': image, 'label': label}


class Random_rotate(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
        label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False)

        return {'image': image, 'label': label}


class Pad(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
        label = np.pad(label, ((0, 0), (0, 0), (0, 5)), mode='constant')
        return {'image': image, 'label': label}
    #(240,240,155)>(240,240,160)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
        label = sample['label']
        label = np.ascontiguousarray(label)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}


def transform(sample):
    trans = transforms.Compose([
        Pad(),
        # Random_rotate(),  # time-consuming
        Random_Crop(),
        Random_Flip(),
        Random_intencity_shift(),
        ToTensor()
    ])

    return trans(sample)


def transform_valid(sample):
    trans = transforms.Compose([
        Pad(),
        # MaxMinNormalization(),
        ToTensor()
    ])

    return trans(sample)


def split_dataset(data_root, nfold=3, seed=0, select=0):
    patients_dir = glob.glob(os.path.join(data_root, "*GG", "Brats18*"))
    n_patients = len(patients_dir)
    print(f"total patients: {n_patients}")
    pid_idx = np.arange(n_patients)
    np.random.seed(seed)
    np.random.shuffle(pid_idx)
    n_fold_list = np.split(pid_idx, nfold)
    print("***********no pro**********")
    print(f"split {len(n_fold_list)} folds and every fold have {len(n_fold_list[0])} patients")
    val_patients_list = []
    train_patients_list = []

    for i, fold in enumerate(n_fold_list):
        if i == select:
            for idx in fold:
                val_patients_list.append(patients_dir[idx])
        else:
            for idx in fold:
                train_patients_list.append(patients_dir[idx])
    print(f"train patients: {len(train_patients_list)}, test patients: {len(val_patients_list)}")
    
    # print(np.sort(train_patients_list))

    return train_patients_list, val_patients_list


def read_split():
    with open('./data/3folds.txt') as f:
        lines = f.readlines()
        train_patients_list = lines[0].strip().replace('[', '').replace(']', '').replace(' ', '').replace('\'', '').split(',')
        val_patients_list = lines[1].strip().replace('[', '').replace(']', '').replace(' ', '').replace('\'', '').split(',')

        train_patients_list = [x.replace('.', './data') for x in train_patients_list]
        val_patients_list = [x.replace('.', './data') for x in val_patients_list]

        print(f"train patients: {len(train_patients_list)}, test patients: {len(val_patients_list)}")
    
        # print(np.sort(train_patients_list))
    
        return train_patients_list, val_patients_list


class BraTS(Dataset):
    def __init__(self, list_file, root='', mode='train', drop_modal=False):
        train_list, val_list = split_dataset(root)
        # train_list, val_list = read_split()

        paths, names, missing_modal_list = [], [], []
        if mode == 'train':
            for line in train_list:
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(line, name + '_')
                paths.append(path)

                if drop_modal:
                    missing_num = np.random.randint(4)
                    # missing_num = 1
                    missing_modal = random.sample([0, 1, 2, 3], missing_num)
                    missing_modal_list.append(missing_modal)
                else:
                    missing_modal_list.append([])
        elif mode == 'valid':
            for line in val_list:
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(line, name + '_')
                paths.append(path)

                missing_modal_list.append([2])

        self.mode = mode
        self.names = names
        self.paths = paths
        self.missing_modal_list = missing_modal_list

    def __getitem__(self, item):
        path = self.paths[item]
        if self.mode == 'train':
            image, label = pkload(path + 'data_f32b0.pkl')
            sample = {'image': image, 'label': label}
            sample = transform(sample)
            return sample['image'], sample['label'], self.missing_modal_list[item]
        elif self.mode == 'valid':
            image, label = pkload(path + 'data_f32b0.pkl')
            sample = {'image': image, 'label': label}
            sample = transform_valid(sample)
            return sample['image'], sample['label'], self.missing_modal_list[item]
        else:
            image = pkload(path + 'data_f32b0.pkl')
            image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
            image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
            image = torch.from_numpy(image).float()
            return image, self.missing_modal_list[item]

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]
