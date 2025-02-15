import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pydicom import dcmread
import numpy as np
from matplotlib.path import Path

def get_struct_number(struct_file, structure_name):
    for struct_set in struct_file.StructureSetROISequence:
        if struct_set.ROIName == structure_name:
            return struct_set.ROINumber

def get_slice_contours(struct_file, sop_instance_uid, roi_number):
    for roi_contour in struct_file.ROIContourSequence:
        if roi_contour.ReferencedROINumber == roi_number:
            for contour in roi_contour.ContourSequence:
                for contour_image in contour.ContourImageSequence:
                    if contour_image.ReferencedSOPInstanceUID == sop_instance_uid:
                        return contour.ContourData

def get_mask_from_contour(contour_data, img_size, pixel_spacing, img_origin):
    contour_data = np.array([contour_data[i:i + 3] for i in range(0, len(contour_data), 3)])
    path = Path(contour_data[:,[1,0]] + 256)
    x, y = np.mgrid[:img_size[0], :img_size[1]]
    points = np.hstack((x.reshape(-1, 1), y.reshape(-1,1)))
    mask = path.contains_points(points).reshape(img_size[0], img_size[1])
    sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    sitk_mask.SetSpacing(pixel_spacing)
    sitk_mask.SetOrigin(img_origin)
    sitk_mask = sitk.JoinSeries(sitk_mask)
    return sitk_mask

def get_dicom_data(study_folder):
    ct_images = []
    ct_masks = []
    struct_file = None
    for file_path in os.listdir(study_folder):
        if file_path.startswith('RS'):
            struct_file = dcmread(os.path.join(study_folder, file_path))
        elif file_path.startswith('CT'):
            ct_images.append(sitk.ReadImage(os.path.join(study_folder, file_path)))
        else:
            pass
    structure_number = get_struct_number(struct_file, 'Prostate')
    for image in ct_images:
        sop_instance_uid = image.GetMetaData('0008|0018')
        pixel_spacing = image.GetSpacing()
        img_origin = image.GetOrigin()
        img_size = image.GetSize()
        contour_data = get_slice_contours(struct_file, sop_instance_uid, structure_number)
        if contour_data:
            mask = get_mask_from_contour(contour_data, img_size, pixel_spacing, img_origin)
        else:
            mask = sitk.Image(img_size, sitk.sitkInt8)
            mask.SetSpacing(pixel_spacing)
            mask.SetOrigin(img_origin)
        ct_masks.append(mask)
    return ct_images, ct_masks

# IMGS_DIR = r'D:\ProstateSegmentationDataSet\Dicoms'

# for study in os.listdir(IMGS_DIR):
#     imgs = []
#     labels = []
#     study_folder = os.path.join(IMGS_DIR, study)
#     print(study_folder)
#     ct_images, ct_masks = get_dicom_data(study_folder)
    #imgs.extend(ct_images)
    #labels.extend(ct_masks)

# idx = 65
# image = sitk.GetArrayFromImage(imgs[idx])
# mask = sitk.GetArrayFromImage(labels[idx])

# slice_location = imgs[idx].GetMetaData('0020|0032')
# window_center = int(imgs[idx].GetMetaData('0028|1050'))
# window_width = int(imgs[idx].GetMetaData('0028|1051'))

# vmin = window_center - window_width/2
# vmax = window_center + window_width/2

# fig, axs = plt.subplots()
# fig.suptitle(str(slice_location))
# plt.imshow(image.squeeze(), cmap='gray', vmin=vmin, vmax=vmax)
# plt.imshow(mask.squeeze(), alpha=0.5)
# plt.show()