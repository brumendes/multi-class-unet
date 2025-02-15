import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop


class Unet(nn.Module):
    """
    UNet (almost) original implementation.
    Differences: Applied padding=1 in the down convolutions. The cropping can then be avoided.
    The output size matches the input because of the applied padding and the input size is a multiple of 8.
    Otherwise, the output must be resized for inference with matching label.
    """
    def __init__(self, n_classes):
        super().__init__()
        self.down_conv_1 = self.conv_3x3_relu(1, 64)
        self.down_conv_2 = self.conv_3x3_relu(64, 128)
        self.down_conv_3 = self.conv_3x3_relu(128, 256)
        self.down_conv_4 = self.conv_3x3_relu(256, 512)
        self.down_conv_5 = self.conv_3x3_relu(512, 1024)
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up_trans_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv_1 = self.conv_3x3_relu(1024, 512)
        self.up_trans_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv_2 = self.conv_3x3_relu(512, 256)
        self.up_trans_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv_3 = self.conv_3x3_relu(256, 128)
        self.up_trans_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv_4 = self.conv_3x3_relu(128, 64)
        self.output = nn.Conv2d(64, n_classes, kernel_size=1)

    def conv_3x3_relu(self, in_ch, out_ch):
        block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block

    def crop(self, tensor, tensor_target):
        _, _, H, W = tensor_target.shape
        tensor = CenterCrop([H, W])(tensor)
        return tensor

    def forward(self, image):
        """ Encoder """
        d1 = self.down_conv_1(image) # concat with up_4
        p1 = self.max_pool_2x2(d1)

        d2 = self.down_conv_2(p1) # concat with up_3
        p2 = self.max_pool_2x2(d2)

        d3 = self.down_conv_3(p2) # concat with up_2
        p3 = self.max_pool_2x2(d3)

        d4 = self.down_conv_4(p3) # concat with up_1
        p4 = self.max_pool_2x2(d4)

        d5 = self.down_conv_5(p4)
        """ Decoder """
        up_1 = self.up_trans_1(d5)
        # d4_crop = self.crop(d4, up_1)
        up_1 = self.up_conv_1(torch.cat([up_1, d4], dim=1))

        up_2 = self.up_trans_2(up_1)
        # d3_crop = self.crop(d3, up_2)
        up_2 = self.up_conv_2(torch.cat([up_2, d3], dim=1))

        up_3 = self.up_trans_3(up_2)
        # d2_crop = self.crop(d2, up_3)
        up_3 = self.up_conv_3(torch.cat([up_3, d2], dim=1))

        up_4 = self.up_trans_4(up_3)
        # d1_crop = self.crop(d1, up_4)
        up_4 = self.up_conv_4(torch.cat([up_4, d1], dim=1))

        out = self.output(up_4)
        return out