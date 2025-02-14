from pathlib import Path
from ultralytics import YOLO
import numpy as np
import SimpleITK as sitk


# Set paths
DICOM_DIR = Path("../DicomDataset")
DATASET_DIR = Path("../datasets/prostate")
OUTPUT_DIR = DATASET_DIR / "masks"
IMAGES_DIR = DATASET_DIR / "images" / "val"
LABELS_DIR = DATASET_DIR / "labels" / "val"

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# Load YOLO model with the weigths at prostate_segmentation\prostate_seg4\weights\best.pt
yolo = YOLO(model="prostate_segmentation/prostate_seg4/weights/best.pt")

# Process multiple images
for image_path in IMAGES_DIR.glob("*.png"):
    # Run prediction with high confidence threshold
    results = yolo.predict(
        source=image_path,
        conf=0.5,  # Confidence threshold
        iou=0.5,   # NMS IoU threshold
        save=False,
        save_txt=False,
        verbose=False
    )
    
    result = results[0]  # Get first result
    if result.masks is None:
        print(f"No segmentation found for {image_path}")
        continue
    
    # Read original image using SimpleITK
    orig_img = sitk.ReadImage(str(image_path))
    size = orig_img.GetSize()
    
    # Get mask data
    mask = result.masks.data[0].cpu().numpy()
    
    # Convert mask to SimpleITK image
    mask_sitk = sitk.GetImageFromArray(mask)
    
    # Resize mask to match original image
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(orig_img)
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    mask_sitk = resample.Execute(mask_sitk)
    
    # Convert to binary mask
    binary_mask = sitk.BinaryThreshold(
        mask_sitk,
        lowerThreshold=0.5,
        upperThreshold=1.0,
        insideValue=255,
        outsideValue=0
    )

    output_path = OUTPUT_DIR / f"{image_path.stem}_mask.png"
    sitk.WriteImage(binary_mask, str(output_path))
    print(f"Saved mask to {output_path}")

