import sys
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import cv2
import SimpleITK as sitk
from rt_utils import RTStructBuilder
import logging
from datetime import datetime


# Set paths
DICOM_DIR = Path("../DicomDataset")
DATASET_DIR = Path("../datasets/prostate")
IMAGES_DIR = DATASET_DIR / "images" / "train"
LABELS_DIR = DATASET_DIR / "labels" / "train"


class TeeStream:
    """Stream handler that writes to both file and terminal"""
    def __init__(self, file, terminal):
        self.file = file
        self.terminal = terminal

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.terminal.flush()
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()


def setup_logging(log_dir='logs'):
    """Setup logging to both file and console"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will print to console
        ]
    )
    return log_file

def prepare_yolo_dataset():
    """
    Convert DICOM images and RT structures to YOLO format
    """
    for patient_dir in DICOM_DIR.iterdir():
        if not patient_dir.is_dir():
            continue
            
        rt_dir = patient_dir / "RT"
        if not rt_dir.exists():
            logging.warning(f"No RT directory found in {patient_dir}")
            continue
            
        for exam_dir in rt_dir.iterdir():
            if not exam_dir.is_dir():
                continue
                
            logging.info(f"\nProcessing exam: {exam_dir}")
            
            rt_struct_path = next(exam_dir.glob("RS*.dcm"), None)
            if not rt_struct_path:
                logging.warning(f"No RT structure file found in {exam_dir}")
                continue

            try:
                # Get image series
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(str(exam_dir))
                if not dicom_names:
                    logging.warning(f"No DICOM series found in {exam_dir}")
                    continue
                    
                reader.SetFileNames(dicom_names)
                series = reader.Execute()
                img_array = sitk.GetArrayFromImage(series)  # (slices, H, W)
                
                logging.info(f"Found {len(dicom_names)} DICOM files")
                logging.info(f"Image array shape: {img_array.shape}")
                
                # Get contours from RT structure
                rtstruct = RTStructBuilder.create_from(
                    dicom_series_path=exam_dir,
                    rt_struct_path=rt_struct_path
                )
                
                mask_3d = rtstruct.get_roi_mask_by_name("Prostate")
                # Transpose mask to match image dimensions
                mask_3d = np.transpose(mask_3d, (2, 0, 1))
                logging.info(f"Mask array shape after transpose: {mask_3d.shape}")

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
                    img_path = IMAGES_DIR / img_name
                    cv2.imwrite(str(img_path), img_norm)

                    # Create YOLO label
                    label_path = LABELS_DIR / f"{patient_dir.name}_{exam_dir.name}_{i:03d}.txt"
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

                logging.info(f"Processed {slices_with_roi} slices with ROI")
                
            except Exception as e:
                logging.error(f"Error processing {exam_dir}: {str(e)}")
                continue


def create_validation_split(dataset_dir, val_split=0.2):
    """Create validation split from training data"""
    dataset_dir = Path(dataset_dir)
    
    # Create validation directories
    val_images_dir = dataset_dir / 'images' / 'val'
    val_labels_dir = dataset_dir / 'labels' / 'val'
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all training images
    train_images = list((dataset_dir / 'images' / 'train').glob('*.png'))
    
    # Randomly select validation images
    val_size = int(len(train_images) * val_split)
    val_images = np.random.choice(train_images, val_size, replace=False)
    
    # Move validation images and their labels
    moved_count = 0
    for img_path in val_images:
        img_name = img_path.name
        label_name = img_path.stem + '.txt'
        
        # Define target paths
        val_img_path = val_images_dir / img_name
        val_label_path = val_labels_dir / label_name
        
        try:
            # Move image if it doesn't exist in val
            if not val_img_path.exists():
                img_path.rename(val_img_path)
                
                # Move corresponding label
                label_path = dataset_dir / 'labels' / 'train' / label_name
                if label_path.exists() and not val_label_path.exists():
                    label_path.rename(val_label_path)
                    moved_count += 1
        except Exception as e:
            logging.error(f"Error moving {img_name}: {str(e)}")
            continue
    
    logging.info(f"Split {moved_count} images for validation")


if __name__ == "__main__":
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting YOLO training preparation...")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare dataset
    # prepare_yolo_dataset()

    # Print dataset statistics
    logging.info("\nDataset statistics:")
    logging.info(f"Number of images: {len(list(IMAGES_DIR.glob('*.png')))}")
    logging.info(f"Number of labels: {len(list(LABELS_DIR.glob('*.txt')))}")
    
    # Create validation split if we have data
    if len(list(IMAGES_DIR.glob('*.png'))) > 0:
        # create_validation_split(DATASET_DIR)

        # Setup YOLO logging
        log_dir = Path('logs')
        yolo_log = log_dir / f'yolo_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        # Create tee stream that writes to both file and terminal
        original_stdout = sys.stdout
        with open(yolo_log, 'w') as f:
            tee = TeeStream(f, original_stdout)
            sys.stdout = tee

            try:
                # Initialize and train model
                model = YOLO('yolov8n-seg.pt')  # load pretrained model
                results = model.train(
                    data='prostate.yml',
                    epochs=10,
                    imgsz=512,
                    batch=8,
                    name='prostate_seg',
                    device=0,  # GPU device (use -1 for CPU)
                    workers=4,
                    val=True,
                    patience=20,  # early stopping patience
                    save=True,  # save checkpoints
                    project='prostate_segmentation',  # project name
                    save_period=10,  # save checkpoint every 10 epochs
                    # logger='csv',  # log results to f'runs/train'
                )
            finally:
                # Restore original stdout
                sys.stdout = original_stdout

    logging.info(f"Training completed. Results saved in {results.save_dir}")
    logging.info(f"YOLO terminal output saved to {yolo_log}")