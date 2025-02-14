from ultralytics import YOLO
from pathlib import Path
import numpy as np
import cv2
import SimpleITK as sitk
from rt_utils import RTStructBuilder


def prepare_yolo_dataset(dicom_dir, output_dir):
    """
    Convert DICOM images and RT structures to YOLO format
    """
    dicom_dir = Path(dicom_dir)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images" / "train"
    labels_dir = output_dir / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for patient_dir in dicom_dir.iterdir():
        if not patient_dir.is_dir():
            continue
            
        rt_dir = patient_dir / "RT"
        if not rt_dir.exists():
            print(f"No RT directory found in {patient_dir}")
            continue
            
        for exam_dir in rt_dir.iterdir():
            if not exam_dir.is_dir():
                continue
                
            print(f"\nProcessing exam: {exam_dir}")
            
            rt_struct_path = next(exam_dir.glob("RS*.dcm"), None)
            if not rt_struct_path:
                print(f"No RT structure file found in {exam_dir}")
                continue

            try:
                # Get image series
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(str(exam_dir))
                if not dicom_names:
                    print(f"No DICOM series found in {exam_dir}")
                    continue
                    
                reader.SetFileNames(dicom_names)
                series = reader.Execute()
                img_array = sitk.GetArrayFromImage(series)  # (slices, H, W)
                
                print(f"Found {len(dicom_names)} DICOM files")
                print(f"Image array shape: {img_array.shape}")
                
                # Get contours from RT structure
                rtstruct = RTStructBuilder.create_from(
                    dicom_series_path=exam_dir,
                    rt_struct_path=rt_struct_path
                )
                
                mask_3d = rtstruct.get_roi_mask_by_name("Prostate")
                # Transpose mask to match image dimensions
                mask_3d = np.transpose(mask_3d, (2, 0, 1))
                print(f"Mask array shape after transpose: {mask_3d.shape}")

                # Process each slice
                slices_with_roi = 0
                for i, (img, mask) in enumerate(zip(img_array, mask_3d)):
                    if not mask.any():  # Skip slices without prostate
                        continue

                    slices_with_roi += 1
                    
                    # Normalize image to [0, 255]
                    img_norm = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                    
                    # Find contours in the mask
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8), 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )

                    if not contours:  # Skip if no contours found
                        continue

                    # Save image
                    img_name = f"{patient_dir.name}_{exam_dir.name}_{i:03d}.png"
                    img_path = images_dir / img_name
                    cv2.imwrite(str(img_path), img_norm)

                    # Create YOLO label
                    label_path = labels_dir / f"{patient_dir.name}_{exam_dir.name}_{i:03d}.txt"
                    height, width = mask.shape
                    with open(label_path, 'w') as f:
                        for contour in contours:
                            points = contour.squeeze()
                            if len(points) < 3:  # Skip invalid contours
                                continue
                            
                            # Normalize coordinates to [0, 1]
                            points_norm = points.astype(float)
                            points_norm[:, 0] /= width  # x coordinates
                            points_norm[:, 1] /= height  # y coordinates
                            
                            # Write points to file
                            points_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in points_norm])
                            f.write(f"0 {points_str}\n")

                print(f"Processed {slices_with_roi} slices with ROI")
                
            except Exception as e:
                print(f"Error processing {exam_dir}: {str(e)}")
                continue

    return images_dir, labels_dir


def create_validation_split(dataset_dir, val_split=0.2):
    """Create validation split from training data"""
    dataset_dir = Path(dataset_dir)
    
    # Create validation directories
    (dataset_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (dataset_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # Get all training images
    train_images = list((dataset_dir / 'images' / 'train').glob('*.png'))
    
    # Randomly select validation images
    val_size = int(len(train_images) * val_split)
    val_images = np.random.choice(train_images, val_size, replace=False)
    
    # Move validation images and their labels
    for img_path in val_images:
        img_name = img_path.name
        label_name = img_path.stem + '.txt'
        
        # Move image
        img_path.rename(dataset_dir / 'images' / 'val' / img_name)
        
        # Move corresponding label
        label_path = dataset_dir / 'labels' / 'train' / label_name
        if label_path.exists():
            label_path.rename(dataset_dir / 'labels' / 'val' / label_name)
    
    print(f"Split {len(val_images)} images for validation")


if __name__ == "__main__":
    # Set paths
    dicom_dir = Path("../DicomDataset")
    dataset_dir = Path("../datasets/prostate")
        
    # Prepare dataset
    images_dir, labels_dir = prepare_yolo_dataset(dicom_dir, dataset_dir)
    
    # Print dataset statistics
    print("\nDataset statistics:")
    print(f"Number of images: {len(list(images_dir.glob('*.png')))}")
    print(f"Number of labels: {len(list(labels_dir.glob('*.txt')))}")
    
    # Create validation split if we have data
    if len(list(images_dir.glob('*.png'))) > 0:
        create_validation_split(dataset_dir)
        create_validation_split(dataset_dir)
        
        # Initialize and train model
        model = YOLO('yolov8n-seg.pt')  # load pretrained model
        results = model.train(
            data='prostate.yml',
            epochs=20,
            imgsz=512,
            batch=16,
            name='prostate_seg',
            device=0,  # GPU device (use -1 for CPU)
            workers=4,
            val=True,
            patience=20,  # early stopping patience
            save=True,  # save checkpoints
            project='prostate_segmentation',  # project name
            save_period=10  # save checkpoint every 10 epochs
        )
    else:
        print("No data was generated. Please check the error messages above.")