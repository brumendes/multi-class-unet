import os
import pandas as pd
import tqdm as tq
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop
import config
from dataset import CTDataSet
from model import Unet
from utils import get_train_val_split, compute_metrics
from img_utils import label_overlay, results_display

# 1: Get train/val split
train_imgs, val_imgs, train_labels, val_labels = get_train_val_split(os.path.join(config.INFO_DIR, 'train_test_split.csv'))

# 2: Prepare dataset
transform = CenterCrop([int(config.IMG_HEIGHT/2), int(config.IMG_WIDTH/2)])
val_set = CTDataSet(val_imgs, val_labels, config.VOLUMES_DIR, config.ORGAN_TYPE, transform)

# 3: Prepare validation dataloader
val_data_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

# 4: Load Unet and state dictionary
model = Unet(config.NUM_CLASSES).to(config.DEVICE)
model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, 'All_model_100epochs_10batch_71pts.pt')))
model.eval()

# 5: Evaluate on the full validation set
data = {'img_idx': [], 'dice': [], 'iou': []}
gt_list = []
pred_list = []
progressbar = tq.tqdm(val_data_loader, desc='Evaluating model...', total=len(val_data_loader))
with torch.no_grad():
    for idx, (image, target) in enumerate(progressbar):
        image, target = image.to(config.DEVICE), target.to(config.DEVICE)
        out = model(image)
        probs = torch.softmax(out, dim=1)
        full_mask = torch.argmax(probs, dim=1)
        acc, dice, int_ou = compute_metrics(probs, target, config.NUM_CLASSES)
        data['img_idx'].append(idx)
        data['dice'].append(dice.item())
        data['iou'].append(int_ou.item())
        gt_overlay = label_overlay(image.cpu(), target.cpu())
        pred_overlay = label_overlay(image.cpu(), full_mask.cpu())
        gt_list.append(gt_overlay)
        pred_list.append(pred_overlay)

# 6: Save results to csv
df = pd.DataFrame.from_dict(data)
df.to_csv(os.path.join(config.RESULTS_DIR, 'multiclass', 'multiclass_results_v1.csv'))

df = df[df.dice != 0]

# 7: Get the 5 best and 5 worst results (based on dice score)
top_5 = df.nlargest(5, 'dice')
bottom_5 = df.nsmallest(5, 'dice')

display_idx_top = top_5['img_idx'].values
display_scores_top = top_5['iou'].values

display_idx_bottom = bottom_5['img_idx'].values
display_scores_bottom = bottom_5['dice'].values

display_gt_top = [gt_list[i] for i in display_idx_top]
display_pred_top = [pred_list[i] for i in display_idx_top]

display_gt_bottom = [gt_list[i] for i in display_idx_bottom]
display_pred_bottom = [pred_list[i] for i in display_idx_bottom]

results_display(display_gt_top, display_pred_top, display_scores_top, type='best')
results_display(display_gt_bottom, display_pred_bottom, display_scores_bottom, type='worst')

