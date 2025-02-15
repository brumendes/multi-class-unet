import os
import SimpleITK as sitk
import tqdm as tq
import numpy as np
import torch
from torch.utils.data import Dataset
from src.dicom_tools import get_dicom_data
import config


class CTDataSet(Dataset):
    """
    Custom Dataset holding ct images and labels.
    Imgs_names and labels_names are Nrrd volumes contained in the main_dir folder.
    The dataset splits the volume into single slices.
    It will also ignore labels and corresponding images that only contain background.
    1: Prostate
    2: Bladder
    3: Rectum
    """
    def __init__(self, imgs_names, labels_names, main_dir, organ_type='Prostate', transform=None, train=False):
        self.imgs_names = imgs_names
        self.labels_names = labels_names
        self.main_dir = main_dir
        self.label_map = self.get_label_map(organ_type)
        self.imgs_list = []
        self.labels_list = []
        self.transform = transform
        if train:
            msg = 'Training'
        else:
            msg = 'Validation'

        progressbar = tq.tqdm(range(len(self.imgs_names)), desc=f'Caching {msg} data...')
        for i, img_name, label_name in zip(progressbar, self.imgs_names, self.labels_names):
            img_path = os.path.join(config.VOLUMES_DIR, img_name)
            label_path = os.path.join(config.VOLUMES_DIR, label_name)
            image_vol = sitk.ReadImage(img_path)
            label_vol = sitk.ReadImage(label_path)
            re_label_vol = sitk.ChangeLabel(label_vol, self.label_map)
            stats = sitk.LabelShapeStatisticsImageFilter()
            _, _, n = image_vol.GetSize()
            # Loop through each slice and ignore if mask has only background.
            for j in range(0, n):
                label = re_label_vol[:, :, j]
                image = image_vol[:, :, j]
                stats.Execute(label)
                if stats.GetNumberOfLabels() > 0:
                    self.imgs_list.append(image)
                    self.labels_list.append(label)
                # n_classes = np.unique(sitk.GetArrayFromImage(label[:, :, j]))
                # if len(n_classes) == self.n_classes:
                    #self.imgs_list.append(image_vol[:, :, j])
                    #self.labels_list.append(label[:, :, j])
                else:
                    pass
        print(f'{len(self.imgs_list)} images loaded. Ignored images with only background.')

    def get_label_map(self, organ_type):
        organ_types = {'Prostate', 'Bladder', 'Rectum', 'All'}
        if organ_type not in organ_types:
            raise ValueError("Organ to be segmented must be one of %r." % organ_types)
        else:
            if organ_type == 'Prostate':
                label_map = {1:1, 2:0, 3:0}
            elif organ_type == 'Bladder':
                label_map = {1:0, 2:1, 3:0}
            elif organ_type == 'Rectum':
                label_map = {1:0, 2:0, 3:1}
            else:
                label_map = {1:1, 2:2, 3:3}
        return label_map

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, idx):
        img = sitk.GetArrayFromImage(self.imgs_list[idx])
        label = sitk.GetArrayFromImage(self.labels_list[idx])
        x = torch.from_numpy(img[np.newaxis, ...])
        y = torch.from_numpy(label)
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x.float(), y.long()
        
    def get_sitk_images(self, idx):
        """
        Utility function to return Sitk images instead of Tensors.
        Usefull for visualization or to apply Sitk functions.
        """
        return self.imgs_list[idx], self.labels_list[idx]


class DicomDataSet(Dataset):
    def __init__(self, imgs_dir):
        self.imgs_dir = imgs_dir
        self.imgs_list = []
        self.labels_list = []
        progressbar = tq.tqdm(range(len(os.listdir(imgs_dir))), desc='Caching data...')
        for study in os.listdir(imgs_dir):
            study_folder = os.path.join(self.imgs_dir, study)
            ct_images, ct_masks = get_dicom_data(study_folder)
            self.imgs_list.extend(ct_images)
            self.labels_list.extend(ct_masks)

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, idx):
        img = sitk.GetArrayFromImage(self.imgs_list[idx])
        label = sitk.GetArrayFromImage(self.labels_list[idx])
        x = torch.from_numpy(img[np.newaxis, ...])
        y = torch.from_numpy(label)
        return x.float(), y.long()