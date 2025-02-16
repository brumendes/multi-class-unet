from pathlib import Path
import SimpleITK as sitk
from rt_utils import RTStructBuilder
import logging
from yolo_utils import create_3d_mask

patient_id = "3677271"
study_date = "20201014"
study_dir = Path(f"C:/Users/admin/Documents/DicomDataset/{patient_id}/RT/{study_date}")
yolo_labels = Path("C:/Users/admin/Documents/datasets/prostate/labels").rglob("*.txt")

# find the txt file int the yolo_labels that matches the patient_id and study_date
# the txt file path is of the form {patient_id}_{study_date}_{frame}.txt
txt_file = None
for txt in yolo_labels:
    if f"{patient_id}_{study_date}" in txt.name:
        txt_file = txt
        break

if txt_file is None:
    raise FileNotFoundError(f"No YOLO label file found for {patient_id} on {study_date}")


# Get number of slices from DICOM series
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(str(study_dir))
reader.SetFileNames(dicom_names)
dicom_image = reader.Execute()
size = dicom_image.GetSize()
print(size)

logging.info(f"DICOM image size: {size}")  # e.g., (512, 512, 168)

# Create 3D mask with correct dimensions
mask_3d = create_3d_mask(txt_file, size[2], (size[0], size[1]))  # Changed order to match DICOM

logging.info(f"Final mask shape: {mask_3d.shape}")
logging.info(f"Expected shape: ({size[2]}, {size[1]}, {size[0]})")

print(mask_3d.shape)
# # Create RT structure file
# rt_struct = RTStructBuilder.create_new(dicom_series_path=study_dir)
# rt_struct.add_roi(mask_3d, name="Prostate")
# rt_struct.save("new_rt_struct.dcm")
