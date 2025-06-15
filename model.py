import torch
import torch.nn as nn

class SmallUNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_dropout=0.1):
        super(SmallUNet, self).__init__()

        def CBR(in_channels, out_channels, dropout_p=0.0):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=1, bias=False),  # Depthwise separable conv
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=1, bias=False),  # Depthwise separable conv
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Reduced number of filters
        self.enc1 = CBR(in_channels, 32, dropout_p=base_dropout * 1)
        self.enc2 = CBR(32, 64, dropout_p=base_dropout * 1.5)
        self.enc3 = CBR(64, 128, dropout_p=base_dropout * 2)
        self.enc4 = CBR(128, 256, dropout_p=base_dropout * 2.5)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = CBR(256, 512, dropout_p=base_dropout * 3)

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = CBR(512, 256, dropout_p=base_dropout * 2.5)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = CBR(256, 128, dropout_p=base_dropout * 2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = CBR(128, 64, dropout_p=base_dropout * 1.5)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = CBR(64, 32, dropout_p=base_dropout * 1)

        # Final 1x1 convolution for output
        self.conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        # return torch.sigmoid(self.conv(dec1))
        return self.conv(dec1)

class SmallUNetSGM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SmallUNetSGM, self).__init__()

        def CBR(in_channels, out_channels, dropout_p=0.0):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=1, bias=False),  # Depthwise separable conv
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=1, bias=False),  # Depthwise separable conv
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Reduced number of filters
        self.enc1 = CBR(in_channels, 32, dropout_p=0.1)
        self.enc2 = CBR(32, 64, dropout_p=0.2)
        self.enc3 = CBR(64, 128, dropout_p=0.3)
        self.enc4 = CBR(128, 256, dropout_p=0.4)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = CBR(256, 512)

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = CBR(512, 256, dropout_p=0.4)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = CBR(256, 128, dropout_p=0.3)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = CBR(128, 64, dropout_p=0.2)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = CBR(64, 32, dropout_p=0.1)

        # Final 1x1 convolution for output
        self.conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.conv(dec1))
        # return self.conv(dec1)