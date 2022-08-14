import os
import torch
from torch.utils.data import Dataset

from .rand import Uniform
from .transforms import Rot90, Flip, Identity, Compose
from .transforms import GaussianBlur, Noise, Normalize, RandSelect
from .transforms import RandCrop, CenterCrop, Pad,RandCrop3D,RandomRotion,RandomFlip,RandomIntensityChange
from .transforms import NumpyType
from .data_utils import pkload

import numpy as np


class Brats_loadall(Dataset):
    def __init__(self, transforms='', root=None, settype='train', split='split1'):
        datalist = np.load(os.path.join(root, settype+'_'+split+'.npy'))
        datalist.sort()
        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.p = [1/2, 1/2]

    def __getitem__(self, index):
        #print ('p', self.p)

        volpath = self.volpaths[index]
        name = self.names[index]
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)

        '''
        shape = list(y.shape)
        shape.insert(1, 4)
        shape = tuple(shape)
        y_x = np.zeros(shape)
        y_x[:,0,:,:,:] = (y == 0)
        y_x[:,1,:,:,:] = (y == 1)
        y_x[:,2,:,:,:] = (y == 2)
        y_x[:,3,:,:,:] = (y == 4)
        '''
        y[y==4] =3

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        for attempt in range(100):
            mask = np.random.choice(2, 4, replace=True, p=self.p)
            mask = (mask == 1)
            if np.sum(mask) != 0:
                mask = torch.from_numpy(mask)
                return x, y, mask, name
        mask = np.array([True, True, True, True])
        mask = torch.from_numpy(mask)
        return x, y, mask, name

    def __len__(self):
        return len(self.volpaths)

class Brats_loadall_test(Dataset):
    def __init__(self, transforms='', root=None, settype='train', split='split1'):
        datalist = np.load(os.path.join(root, settype+'_'+split+'.npy'))
        datalist.sort()
        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)

        x, y = torch.from_numpy(x), torch.from_numpy(y)

        return x, y, name

    def __len__(self):
        return len(self.volpaths)

class Brats_loadall_val(Dataset):
    def __init__(self, transforms='', root=None, settype='train', split='split1'):
        datalist = np.load(os.path.join(root, settype+'_'+split+'.npy'))
        datalist.sort()
        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.masks = np.array([[True, False, False, False], [False, True, False, False], [False, False, True, False], [False, False, False, True],
                      [True, True, False, False], [True, False, True, False], [True, False, False, True], [False, True, True, False], [False, True, False, True], [False, False, True, True],
                      [True, True, True, False], [True, True, False, True], [True, False, True, True], [False, True, True, True],
                      [True, True, True, True]])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        mask = self.masks[index%15]
        mask = torch.from_numpy(mask)
        return x, y, mask, name

    def __len__(self):
        return len(self.volpaths)
