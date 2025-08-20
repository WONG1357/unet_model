import torch
import torch.nn as nn
from layers import DoubleConv, Down, Up

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Decoder (Upsampling)
        for feature in reversed(features):
            self.ups.append(Up(feature*2, feature))
            self.ups.append(DoubleConv(feature*2, feature))

        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x, skip_connections[idx // 2])
            x = self.ups[idx + 1](x)

        return self.final_conv(x)

if __name__ == "__main__":
    x = torch.randn((1, 3, 572, 572))
    model = UNet(in_channels=3, out_channels=1)
    preds = model(x)
    print(preds.shape)
