from pathlib import Path
import SimpleITK as sitk
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pydicom import dcmread


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
        return len(self.image_vol)

    def __getitem__(self, index):
        img = sitk.GetArrayFromImage(self.image_vol[:,:,index])
        x = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        if self.transform:
            x = self.transform(x)
        return x.float()

dicom_series_path = Path(r"C:\Users\admin\Documents\DicomDataset\11122603\RT\20200812")
rt_struct_path = next(dicom_series_path.glob("RS*.dcm"), None)
frame_indices = get_roi_frame_indices(rt_struct_path, roi_name="Prostate")
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(dicom_series_path)

dataset = CustomDataSet(dicom_series_path, frame_indices)
# data_loader = DataLoader(
#     dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
# )

# # get an image from the dataset using the dataloader
# for i, data in enumerate(data_loader):
#     img_array = data.squeeze().numpy()
#     break
# # plot the first slice
# plt.imshow(img_array, cmap='gray')
# plt.show()