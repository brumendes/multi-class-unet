import os
import torch

# Project Name
PROJECT_NAME = 'Prostate Segmentations'

# Organ to be segmented
ORGAN_TYPE = 'All'

# Set Cuda or cpu device
print(PROJECT_NAME)
print("torch = {}".format(torch.__version__))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(DEVICE))

# Root Directory
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

# Dataset directory and file names file path
VOLUMES_DIR = r'E:\PRODEB\ProstateSegmentationDataSet\Nrrd'
MODEL_DIR = r'E:\PRODEB\ProstateSegmentationDataSet\Models'
INFO_DIR = os.path.join(ROOT_DIR, 'info')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

NUM_CHANNELS = 1
NUM_CLASSES = 4
BATCH_SIZE = 10
NUM_WORKERS = 10
NUM_EPOCHS = 100

IMG_HEIGHT = 512
IMG_WIDTH = 512

OPT_LR = 1e-3
SCHE_GOAL = 'max'
SCHE_PATIENCE = 2
SCHE_STEP = 16
SCHE_GAMMA = 0.5