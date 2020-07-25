#Inspired by https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/colorization_dataset.py

import numpy as np
import random

import os.path
from os import path


import torch
from torch.utils.data import Dataset

from PIL import Image

from PIL import ImageFile

import torchaudio

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from image_folder import get_images, get_folders

class read_data():
    """
    Dataloader class used to load in data in an image folder.
    Made it so that it performs a fixed set of transformations to a pair of images in different folders
    """

    def __init__(self, opts):
        '''
        :param opts:
        :param img_folder:
        :param attribute_file:
        '''


        self.opts = opts

        self.file_names = get_images(self.opts.dir)
        self.file_len = len(self.file_names)

        if path.exists(self.opts.dict_name_0) and path.exists(self.opts.dict_name_1):
            self.load_dict()
        else:
            self.create_dict()
            self.label_to_name_dict = {v: k for k, v in self.name_to_label_dict.items()}
            self.save_dict()

    def train_test_split(self):
        '''
        Given the data that's loaded above, returns a data that's been split into training and test set.
        :param split: what % of the data should be allocated to the training dataset
        :return: training, test which are list of tuples (Label, Name)
        '''

        p = np.random.permutation(len(self.file_names))

        data = np.asarray(self.file_names)[p]

        cutoff = np.int(len(data) * self.opts.split)

        train, test = data[:cutoff], data[cutoff:]

        return train, test

    def return_class_size(self):

        return len(self.name_to_label_dict)

    def create_dict(self):

        keyList = get_folders(self.opts.dir)
        # initialize dictionary
        self.name_to_label_dict = {}

        # iterating through the elements of list
        for index, key in enumerate(keyList):
            key = key.split('/')[-1]
            if key != '':
                self.name_to_label_dict[key] = index - 1

    def save_dict(self):

        np.save(self.opts.dict_name_0, self.name_to_label_dict)
        np.save(self.opts.dict_name_1, self.label_to_name_dict)

    def load_dict(self):

        self.name_to_label_dict = np.load(self.opts.dict_name_0, allow_pickle='TRUE').item()
        self.label_to_name_dict = np.load(self.opts.dict_name_1, allow_pickle='TRUE').item()

class Audio_Dataloader(Dataset):
    """
    Dataloader class used to load in data in an image folder.
    Made it so that it performs a fixed set of transformations to a pair of images in different folders
    """

    def __init__(self, filenames, opts, dict0, train=True):
        '''
        :param img_folder:
        :param transform:
        :param additional_transform:
        :param final_transformation:
        '''

        self.opts = opts

        self.train = train

        self.file_names = filenames
        self.file_len = len(self.file_names)

        self.name_to_label_dict = dict0
    def __len__(self):
        '''
        return the length of the dataset
        :return:
        '''
        return len(self.file_names)

    def __getitem__(self, index):
        '''
        function that fetches the data and does appropriate preprocessing with it in order to get the data ready
        :param index:
        :return:
        '''

        #
        filetuple = self.file_names[index % self.file_len]
        filename  = filetuple[0]
        filelabel = filetuple[1]

        waveform, sample_rate = torchaudio.load(filename)

        specgram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.opts.SR,
            n_fft=self.opts.N_FFT,
            n_mels=self.opts.N_MELS,
            hop_length=self.opts.HOP_LEN,
            )(waveform)

        #make the length more uniform
        n_sample = specgram.shape[2]
        n_sample_fit = int(self.opts.DURA)

        if n_sample < n_sample_fit:  # if too short
            specgram = torch.cat((specgram, torch.zeros((1, self.opts.N_MELS, self.opts.DURA - n_sample))), dim=2)
        elif n_sample > n_sample_fit:  # if too long
            specgram = specgram[:, :, int((n_sample - n_sample_fit) / 2) : int((n_sample + n_sample_fit) / 2)]


        return specgram, self.name_to_label_dict.__getitem__(filelabel)

