from pathlib import Path
import torch
from torchvision.transforms import CenterCrop
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import SimpleITK as sitk
import numpy as np

from model import Unet


# Project Name
PROJECT_NAME = "Prostate Segmentations"

# Organ to be segmented
ORGAN_TYPE = "Prostate"

# Set Cuda or cpu device
print(PROJECT_NAME)
print(f"torch = {torch.__version__}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")

MODEL_PATH = Path("/Volumes/PRODEB/Models/Prostate_model_100epochs_10batch_71pts.pt")

transform = CenterCrop([int(512 / 2), int(512 / 2)])


class CustomDataSet(Dataset):
    """Custom dataset for DICOM images."""

    def __init__(self, imgs_dir):
        self.imgs_dir = imgs_dir
        self.imgs_list = []
        self.patient_ids = []

        for study in imgs_dir.iterdir():
            print(f"Loading study {study.name}")
            rt_path = study / "RT"
            # Skip if RT directory doesn't exist
            if not rt_path.exists():
                print(f"No RT directory found for {study.name}")
                continue
            # Get the exam date directory (assuming there's only one)
            exam_dates = list(rt_path.iterdir())
            if not exam_dates:
                print(f"No exam date directory found for {study.name}")
                continue
            exam_path = exam_dates[0]  # Take the first exam date
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(exam_path))
            if not dicom_names:
                print(f"No DICOM series found in {exam_path}")
                continue
            reader.SetFileNames(dicom_names)
            dicom_series = reader.Execute()
            self.imgs_list.append(dicom_series)
            self.patient_ids.append(study.name)

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        img = sitk.GetArrayFromImage(self.imgs_list[index])
        # Process all slices at once
        img = img.astype(np.float32)  # Convert to float32
        # Reshape to [num_slices, 1, H, W]
        img = img[:, np.newaxis, ...]
        img = torch.from_numpy(img)
        if transform:
            img = transform(img)
        return img

    def get_patient_id(self, index):
        """Get patient ID for a given index."""
        return self.patient_ids[index]


# Load model
model = Unet(2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Load dataset
data_dir = Path("/Volumes/PRODEB/DicomDataset")
dataset = CustomDataSet(data_dir)

# Prepare dataloader
data_loader = DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
)

output_dir = Path("predictions")
output_dir.mkdir(exist_ok=True)

with torch.no_grad():
    for idx, image_batch in enumerate(data_loader):
        # image_batch shape is now [1, num_slices, 1, H, W]
        # Reshape to [num_slices, 1, H, W]
        image_batch = image_batch.squeeze(0)
        image_batch = image_batch.to(DEVICE)

        # Process slices in smaller batches to avoid memory issues
        batch_size = 16
        masks_list = []
        for i in range(0, len(image_batch), batch_size):
            batch = image_batch[i:i + batch_size]
            out = model(batch)
            probs = torch.softmax(out, dim=1)
            masks = torch.argmax(probs, dim=1)
            masks_list.append(masks)

        # Concatenate all batches
        masks = torch.cat(masks_list, dim=0)

        # Get patient ID and create patient directory
        patient_id = dataset.get_patient_id(idx)
        patient_dir = output_dir / patient_id
        patient_dir.mkdir(exist_ok=True)

        # Save each slice
        masks_np = masks.cpu().numpy()
        for slice_idx, slice_data in enumerate(masks_np):
            mask_sitk = sitk.GetImageFromArray(slice_data.astype(np.uint8))
            filename = f"slice_{slice_idx:03d}.png"
            filepath = patient_dir / filename
            sitk.WriteImage(mask_sitk, str(filepath))

        print(f"Saved segmentations for patient {patient_id}")

print(f"Predictions saved to {output_dir}")
