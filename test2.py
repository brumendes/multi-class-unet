import SimpleITK as sitk
import matplotlib.pyplot as plt

img_path = r'E:\PRODEB\ProstateSegmentationDataSet\Nrrd\88632393.nrrd'
label_path = r'E:\PRODEB\ProstateSegmentationDataSet\Nrrd\88632393-label.nrrd'

img_vol = sitk.ReadImage(img_path)
label_vol = sitk.ReadImage(label_path)
re_label_vol = sitk.ChangeLabel(label_vol, {1:1, 2:2, 3:3})
stats = sitk.LabelShapeStatisticsImageFilter()
# stats.Execute(new_img)
# print(stats.GetNumberOfLabels())
_, _, n = re_label_vol.GetSize()
imgs_list = []
labels_list = []
for j in range(0, n):
    img = img_vol[:, :, j]
    label = re_label_vol[:, :, j]
    stats.Execute(label)
    if stats.GetNumberOfLabels() > 0:
        imgs_list.append(img)
        labels_list.append(label)

img_idx = 23
image = imgs_list[img_idx]
segmentation = labels_list[img_idx]

pink = [255, 105, 180]
green = [0, 255, 0]
gold = [255, 215, 0]

combined2 = sitk.LabelOverlay(
    image=image,
    labelImage=segmentation,
    opacity=0.2,
    backgroundValue=0,
    colormap=pink + green + gold,
)

contour_image = sitk.LabelToRGB(
    sitk.LabelContour(segmentation, fullyConnected=True, backgroundValue=255),
    colormap=pink + green + gold,
    backgroundValue=255,
)

contour_overlaid_image = sitk.LabelMapContourOverlay(
    sitk.Cast(segmentation, sitk.sitkLabelUInt8),
    image,
    opacity=1,
    contourThickness=[2, 2],
    dilationRadius=[3, 3],
    colormap=pink + green + gold,
)

plt.imshow(sitk.GetArrayFromImage(contour_overlaid_image))
plt.show()