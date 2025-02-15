import os
import re
import numpy as np
import pandas as pd
import tqdm as tqdm
import torch
from torchmetrics.functional import dice_score
from torchmetrics.functional import accuracy
from torchmetrics.functional import iou
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

def get_filenames(vol_dir):
    images = [f for f in os.listdir(vol_dir) if re.match(r'[0-9]*.nrrd', f)]
    labels = [f for f in os.listdir(vol_dir) if re.match(r'[0-9]+-label*.nrrd', f)]
    images_list = sorted(images)
    labels_list = sorted(labels)
    return images_list, labels_list

def gen_train_val_split_file(vol_dir, output_dir, test_size=0.2):
    images_list, labels_list = get_filenames(vol_dir)
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(images_list, labels_list, test_size=test_size, random_state=10)
    data = pd.concat([
        pd.Series(train_imgs, name='train_imgs'), 
        pd.Series(val_imgs, name='val_imgs'), 
        pd.Series(train_labels, name='train_labels'), 
        pd.Series(val_labels, name='val_labels')], 
        axis=1,
        )
    data.to_csv(os.path.join(output_dir, 'train_test_split.csv'))

def get_train_val_split(file_path):
    df = pd.read_csv(file_path)
    train_imgs = df['train_imgs'].to_list()
    val_imgs = df['val_imgs'].dropna().to_list()
    train_labels = df['train_labels'].to_list()
    val_labels = df['val_labels'].dropna().to_list()
    return train_imgs, val_imgs, train_labels, val_labels


def get_class_weights(data_loader):
    """
    Computes class weights for unbalanced datasets.
    These weights will be introduced in the loss function.
    """
    background_count = []
    prostate_count = []
    for _, target in data_loader:
        background_count.append(np.count_nonzero(target==0))
        prostate_count.append(np.count_nonzero(target==1))
        weights = np.array([1/sum(background_count), 1/sum(prostate_count)])
    return torch.from_numpy(weights).float()

def get_class_weights_sk(data_set):
    weights = []
    progressbar = tqdm(range(data_set.__len__()), desc='Computing class weights...')
    for idx in progressbar:
        _, target = data_set.__getitem__(idx)
        target_weight = compute_class_weight('balanced', classes=np.unique(target.numpy()), y=target.numpy().flatten())
        weights.append(target_weight)
    weights_mean = np.mean(np.array(weights), axis=0)
    return torch.from_numpy(weights_mean).float()

def get_class_weights_v1(data_loader):
    weights = []
    for _, target in data_loader:
        class_sample_count = np.unique(target, return_counts=True)[1]
        if class_sample_count.shape[0] == 2:
            class_sample_count = np.pad(class_sample_count, (0, 1), 'constant')
        weights.append(np.divide(1, class_sample_count, where=class_sample_count!=0))
    return torch.from_numpy(np.mean(np.array(weights), axis=0)).float()

def get_class_weights_v2(data_loader):
    background_count = []
    bladder_count = []
    rectum_count = []
    for _, target in data_loader:
        background_count.append(np.count_nonzero(target==0))
        bladder_count.append(np.count_nonzero(target==1))
        rectum_count.append(np.count_nonzero(target==2))
        weights = np.array([1/sum(background_count), 1/sum(bladder_count), 1/sum(rectum_count)])
    return torch.from_numpy(weights).float()

def compute_metrics(probs, target, num_classes=2):
    acc = accuracy(probs, target, num_classes=num_classes)
    dice = dice_score(probs, target)
    int_ou = iou(probs, target, num_classes=num_classes)
    return acc, dice, int_ou
