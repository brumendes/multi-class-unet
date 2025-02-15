from pathlib import Path
import SimpleITK as sitk
import logging
from datetime import datetime
from tqdm import tqdm


def setup_logging(log_dir='logs'):
    """Setup logging to both file and console"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'dicom_processing_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            # logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def apply_windowing(image, window_center, window_width):
    """
    Apply intensity windowing to an image
    
    Args:
        image (SimpleITK.Image): Input image
        window_center (float): Window center
        window_width (float): Window width
        
    Returns:
        SimpleITK.Image: Windowed image
    """
    min_intensity = window_center - (window_width / 2)
    max_intensity = window_center + (window_width / 2)
    
    window_params = {
        'windowMinimum': min_intensity,
        'windowMaximum': max_intensity,
        'outputMinimum': 0,
        'outputMaximum': 2^16 - 1
    }
    
    return sitk.IntensityWindowing(image, **window_params)


def process_dicom_to_tiff(dicom_path, output_path):
    """
    Convert DICOM series to TIFF images
    
    Args:
        dicom_path (Path): Path to DICOM directory
        output_path (Path): Path to save TIFF files
    """
    
    try:
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Read DICOM series and metadata
        reader = sitk.ImageSeriesReader()
        reader.AddCommand(sitk.sitkProgressEvent, lambda: print("\rProgress: {0:03.1f}%...".format(100*reader.GetProgress()),end=''))
        reader.AddCommand(sitk.sitkStartEvent, lambda: print("Reading DICOM series..."))
        reader.AddCommand(sitk.sitkEndEvent, lambda: print("Done"))
        reader.SetImageIO("GDCMImageIO")
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_path))
        reader.SetFileNames(dicom_names)
        reader.SetLoadPrivateTags(True)
        reader.MetaDataDictionaryArrayUpdateOn()
        dicom_img = reader.Execute()
        logging.info(f"Image size: {dicom_img.GetSize()}")

        # Get metadata
        patient_id = reader.GetMetaData(0, "0010|0020")
        study_date = reader.GetMetaData(0, "0008|0020")
        spacing = dicom_img.GetSpacing()
        origin = dicom_img.GetOrigin()
        modality = reader.GetMetaData(0, "0008|0060")
        number_of_channels = dicom_img.GetNumberOfComponentsPerPixel()
        photometric_interpretation = reader.GetMetaData(0, "0028|0004")
        logging.info(f'Image Modality: {modality}')
        logging.info(f"Number of channels: {number_of_channels}")
        logging.info(f'Photomertic Interpretation: {photometric_interpretation}')
        logging.info(f"Spacing: {spacing}")
        logging.info(f"Origin: {origin}")

        sitk_image_out = sitk.Cast(dicom_img, sitk.sitkFloat32)

        output_file_names = [output_path / f"{patient_id}_{study_date}_{i:03d}.tiff" for i in range(dicom_img.GetDepth())]

        # Save images to TIFF
        writer = sitk.ImageSeriesWriter()
        writer.AddCommand(sitk.sitkProgressEvent, lambda: print("\rProgress: {0:03.1f}%...".format(100*writer.GetProgress()),end=''))
        writer.AddCommand(sitk.sitkEndEvent, lambda: print("Done"))
        writer.AddCommand(sitk.sitkStartEvent, lambda: logging.info("Exporting to TIFF..."))
        writer.AddCommand(sitk.sitkEndEvent, lambda: logging.info("TIFF export completed"))
        writer.SetFileNames(output_file_names)
        writer.SetUseCompression(True)
        writer.SetImageIO("TIFFImageIO")
        writer.Execute(sitk_image_out)    
        logging.info(f"Successfully saved {dicom_img.GetDepth()} images to {output_path}")
        
    except Exception as e:
        logging.error(f"Error processing DICOM series: {str(e)}")
        raise

if __name__ == "__main__":
    # Set paths
    dicom_path = Path("C:/Users/admin/Documents/DicomDataset/3677271/RT/20201014")
    output_path = Path("C:/Users/admin/Documents/test")
    
    # Setup logging
    setup_logging()
    
    # Process DICOM series
    try:
        process_dicom_to_tiff(dicom_path, output_path)
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")