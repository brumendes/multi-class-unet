import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def display(display_list):
    """
    Display utility function.
    Takes a list of images as input.
    The first image is the CT image so a viewing window is applied.
    The others are lables (Ground truth or predicted) and converted to RGB image for color display.
    """
    plt.figure(figsize=(12,8), dpi= 100)
    title = ['Input Image', 'Ground Truth', 'Predicted']
    for i, img in enumerate(display_list):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        if i == 0:
          plt.imshow(img, cmap='gray', vmin=-50, vmax=150)
        else:
          plt.imshow(img)
        plt.axis('off')
    plt.show()

def multi_display(imgs_list, title_list):
    plt.figure(figsize=(12,8), dpi= 100)
    for i, (img, title) in enumerate(zip(imgs_list, title_list)):
        plt.subplot(1, len(imgs_list), i+1)
        plt.title(title)
        plt.imshow(img)
        plt.axis('off')
    plt.subplots_adjust(left=0.0, bottom=0.005, right=1.0, top=0.95, wspace=0.0, hspace=0.0)
    plt.show()

def results_display(gt_list, pred_list, scores, type='best'):
    fig, axs = plt.subplots(nrows=2, ncols=5)
    if type=='best':
        fig.suptitle('Best 5 (Highest Dice score)', fontsize=20)
    else:
        fig.suptitle('Worst 5 (Lowest Dice score)', fontsize=20)
    axs[0, 0].set_ylabel('Ground truth', size=18)
    axs[1, 0].set_ylabel('Predicted', size=18)
    plt.setp(axs.flat, xticks=[], yticks=[])
    for i, (gt, pred, score) in enumerate(zip(gt_list, pred_list, scores)):
        axs[0, i].imshow(gt, aspect='equal')
        axs[1, i].imshow(pred, aspect='equal')
        axs[1, i].set_xlabel(round(score, 2), size=18)
    plt.subplots_adjust(left=0.025, bottom=0.0, right=0.995, top=1, wspace=0.0, hspace=0.0)
    plt.show()


def create_color_image(image, nc=3):
    """
    Creates a color image from one channel gray image.
    Usefull to visualize multilabels.
    Adapted to two classes.
    Must add flexibility to more or less or to choose different colors.
    """
    label_colors = np.array([(0, 0, 0), (255,255,0), (165,42,42)])
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def label_to_rgb(label):
    """
    Converts a label (torch tensor) to rgb color sitk image.
    Returns a numpy array.
    """
    label_color = sitk.LabelToRGB(sitk.GetImageFromArray(label.squeeze()), colormap=[255, 255, 0])
    return sitk.GetArrayFromImage(label_color)

def label_overlay(image, label_image):
    overlay = sitk.LabelMapContourOverlay(
        sitk.Cast(sitk.GetImageFromArray(label_image.squeeze()), sitk.sitkLabelUInt8), 
        sitk.Cast(
            sitk.IntensityWindowing(
                sitk.GetImageFromArray(image.squeeze()),
                windowMinimum=-50,
                windowMaximum=150,
                outputMinimum=0,
                outputMaximum=255
                ),
            sitk.sitkUInt8
            ), 
        opacity=1, 
        contourThickness=[2, 2],
        )
    return overlay