import os
import tqdm as tq
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as opt
from torchinfo import summary
from torchvision.transforms import CenterCrop
import config
from dataset import CTDataSet
from model import Unet
import statistics
import neptune.new as neptune
from utils import gen_train_val_split_file, get_train_val_split, get_class_weights, compute_metrics

# 0: Generate Train/Val split from filenames in volumes directory
# gen_train_val_split_file(config.VOLUMES_DIR, config.INFO_DIR)

# 1: Get train/val split
train_imgs, val_imgs, train_labels, val_labels = get_train_val_split(os.path.join(config.INFO_DIR, 'train_test_split.csv'))

# 2: Setup transform and prepare train and val set
transform = CenterCrop([int(config.IMG_HEIGHT/4), int(config.IMG_WIDTH/4)])
train_set = CTDataSet(train_imgs, train_labels, config.VOLUMES_DIR, config.ORGAN_TYPE, transform, train=True)
val_set = CTDataSet(val_imgs, val_labels, config.VOLUMES_DIR, config.ORGAN_TYPE, transform)

# 3: Prepare train and val dataloaders
train_data_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
val_data_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)

# 3.1: Sanity check: Visualize CT image and label
img_sample, target_sample = next(iter(train_data_loader))
# The batch_size is 2 so the data_loader returns 2 images
display([img_sample.squeeze()[0], target_sample.squeeze()[0]])

# 4: Load Unet
model = Unet(config.NUM_CLASSES)
summary(model, input_size=(config.BATCH_SIZE, config.NUM_CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH))
model.to(config.DEVICE)

# 5: Configure optimizer, scheduler and loss function
optimizer = opt.Adam(model.parameters(), lr=config.OPT_LR)
scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, config.SCHE_GOAL, patience=config.SCHE_PATIENCE)
weights = get_class_weights(train_data_loader)
criterion = nn.CrossEntropyLoss(weight=weights).to(config.DEVICE)

# 6: Connect to Neptune and set parameters
run = neptune.init(
    project="****",
    api_token="*****",
)

params = {
    "optimizer": "Adam", 
    "scheduler": "ReduceLROnPlateau", 
    "scheduler goal": config.SCHE_GOAL,
    "scheduler metric": "dice",
    "scheduler patience": config.SCHE_PATIENCE, 
    "learning_rate": config.OPT_LR,
    "criterion": "CrossEntropyLoss",
    "n_classes": config.NUM_CLASSES,
    "n_epochs": config.NUM_EPOCHS,
    "batch_size": config.BATCH_SIZE
    }
run["parameters"] = params

# 5: Begin training
# # Training Loop
progressbar = tq.trange(config.NUM_EPOCHS, desc='Progress (Epochs)', leave=True)
for p in progressbar:
    
    # Dictionary to hold training and validation losses and metrics
    data = {'train_loss': [], 'val_loss':[], 'acc': [], 'dice': [], 'iou': []}
    
    # Training Round
    model.train()
    train_batch_iter = tq.tqdm(train_data_loader, 'Training', total=len(train_data_loader), leave=False)
    for image, target in train_batch_iter:
        image, target = image.to(config.DEVICE), target.to(config.DEVICE)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        out = model(image)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        data['train_loss'].append(loss.item())
    train_batch_iter.close()
    
    # Validation Round
    model.eval()
    val_batch_iter = tq.tqdm(val_data_loader, 'Validating', total=len(val_data_loader), leave=False)
    for image, target in val_batch_iter:
        image, target = image.to(config.DEVICE), target.to(config.DEVICE)
        with torch.no_grad():
            out = model(image)
            probs = torch.softmax(out, dim=1)
            # full_mask = torch.argmax(probs, dim=1)
            loss = criterion(out, target)
            acc, dice, int_ou = compute_metrics(probs, target)
            data['val_loss'].append(loss.item())
            data['acc'].append(acc.item())
            data['dice'].append(dice.item())
            data['iou'].append(int_ou.item())
    scheduler.step(statistics.mean(data['dice']))
    val_batch_iter.close()
    # Log epoch metrics to neptune
    run['train/loss'].log(statistics.mean(data['train_loss']))
    run['val/loss'].log(statistics.mean(data['val_loss']))
    run['acc'].log(statistics.mean(data['acc']))
    run['dice'].log(statistics.mean(data['dice']))
    run['iou'].log(statistics.mean(data['iou']))
    run['lr'].log(optimizer.param_groups[0]['lr'])

progressbar.close()
run.stop()

save_model = input("Save model (s/n)?: ")

model_name = 'prostate_model_' + str(config.NUM_EPOCHS) + 'epochs_' + str(config.BATCH_SIZE) + 'batch_71pts.pt'

if save_model == "s":
    torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, model_name))
