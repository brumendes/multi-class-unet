from pathlib import Path
import torch
from torchvision.transforms import CenterCrop
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import SimpleITK as sitk
from pydicom import dcmread

from src.unet.model import Unet
from src.img_utils import label_overlay


# Project Name
PROJECT_NAME = "Prostate Segmentations"

# Organ to be segmented
ORGAN_TYPE = "Prostate"

# Set Cuda or cpu device
print(PROJECT_NAME)
print(f"torch = {torch.__version__}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")

MODEL_PATH = Path("../Models/Prostate_model_100epochs_10batch_71pts.pt")

transform = CenterCrop([int(512 / 2), int(512 / 2)])


def list_directories(path, level=0, max_level=3):
    exam_paths = []
    # List patient directories
    for patient_dir in path.iterdir():
        if not patient_dir.is_dir():
            continue
        rt_path = patient_dir / "RT"
        # Check RT directory
        if not rt_path.exists():
            print(f"No RT directory for {patient_dir.name}")
            continue
        # Get exam date directories
        for exam_date in rt_path.iterdir():
            if exam_date.is_dir():
                exam_paths.append(exam_date)

    return exam_paths


def get_roi_frame_indices(rt_struct_path, roi_name="Prostate"):
    """
    Extract frame indices where a specific ROI is defined in an RT structure file.
    
    Args:
        rt_struct_path (Path): Path to the RT structure file
        roi_name (str): Name of the ROI to find (default: "Prostate")
        
    Returns:
        list: Sorted list of frame indices where ROI is defined
    """
    struct_file = dcmread(rt_struct_path)
    
    # Get ROI structure index
    roi_number = None
    for roi in struct_file.StructureSetROISequence:
        if roi.ROIName == roi_name:
            roi_number = roi.ROINumber
            break
    
    if roi_number is None:
        print(f"{roi_name} ROI not found in RT structure file")
        return []
    
    # Get contour data for the ROI
    frame_indices = set()
    for contour in struct_file.ROIContourSequence:
        if contour.ReferencedROINumber == roi_number:
            for contour_slice in contour.ContourSequence:
                # Get referenced image UID from contour data
                ref_image = contour_slice.ContourImageSequence[0]
                frame_indices.add(ref_image.ReferencedSOPInstanceUID)
    
    frame_indices = sorted(list(frame_indices))
    print(f"{roi_name} ROI is defined in {len(frame_indices)} frames")
    return frame_indices


class CustomDataSet(Dataset):
    """Custom dataset for a DICOM study."""

    def __init__(self, imgs_dir, frame_indices, transform=None):
        self.transform = transform
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(imgs_dir)
        if not dicom_names:
            print(f"No DICOM series found in {imgs_dir}")
        # Filter DICOM files by frame indices
        dicom_ids = {dcmread(f).SOPInstanceUID: f for f in dicom_names}
        filtered_names = [dicom_ids[uid] for uid in frame_indices if uid in dicom_ids]
        
        if not filtered_names:
            raise ValueError("No matching DICOM files found for given frame indices")
        reader.SetFileNames(filtered_names)
        self.image_vol = reader.Execute()
        print(f"DICOM volume shape: {self.image_vol.GetSize()}")

    def __len__(self):
        return self.image_vol.GetSize()[2]

    def __getitem__(self, index):
        img = sitk.GetArrayFromImage(self.image_vol[:,:,index])
        x = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        if self.transform:
            x = self.transform(x)
        return x.float()


# Load model
model = Unet(2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Load dataset
dicom_dataset_path = Path("../DicomDataset")
dicom_datasets = list_directories(dicom_dataset_path)
output_dir = Path("predictions")
output_dir.mkdir(exist_ok=True)

# Process each patient's dataset
for patient_idx, data_dir in enumerate(dicom_datasets):
    print(f"\nProcessing patient directory: {data_dir}")
    
    # Find RT structure file
    rt_struct_path = next(data_dir.glob("RS*.dcm"), None)
    if rt_struct_path is None:
        print(f"No RT structure file found in {data_dir}")
        continue
        
    # Get frame indices where ROI is defined
    frame_indices = get_roi_frame_indices(rt_struct_path, roi_name="Prostate")
    if not frame_indices:
        print(f"No Prostate ROI found in {rt_struct_path}")
        continue
        
    try:
        # Create dataset and dataloader
        dataset = CustomDataSet(data_dir, frame_indices)
        data_loader = DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True
        )
        
        # Process each slice
        patient_id = data_dir.parent.parent.name
        patient_dir = output_dir / patient_id
        patient_dir.mkdir(exist_ok=True)
        
        with torch.no_grad():
            for slice_idx, image in enumerate(data_loader):
                # Get prediction
                image = image.to(DEVICE)
                out = model(image)
                probs = torch.softmax(out, dim=1)
                full_mask = torch.argmax(probs, dim=1)
                
                # Create and save overlay
                pred_overlay = label_overlay(image.cpu(), full_mask.cpu())
                filename = f"slice_{slice_idx:03d}.png"
                filepath = patient_dir / filename
                sitk.WriteImage(pred_overlay, str(filepath), imageIO="PNGImageIO")
                
            print(f"Saved {slice_idx + 1} segmentations for patient {patient_id}")
            
    except Exception as e:
        print(f"Error processing patient {patient_id}: {str(e)}")
        continue

print("\nProcessing completed!")
