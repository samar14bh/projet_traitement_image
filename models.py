import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 7
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


# ─── Custom CNN ────────────────────────────────────────────────────────────────

class CustomCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.block3(self.block2(self.block1(x))))


# ─── VGG-Light ────────────────────────────────────────────────────────────────

class VGGLight(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        def vgg_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )

        self.features = nn.Sequential(
            vgg_block(1, 64), vgg_block(64, 128),
            vgg_block(128, 256), vgg_block(256, 256)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─── ResNet18 Adapté ──────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18Adapted(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(64,  64,  2, stride=1)
        self.layer2 = self._make_layer(64,  128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_ch, out_ch, n, stride):
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, n):
            layers.append(ResidualBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


# ─── Mini-Xception ────────────────────────────────────────────────────────────

class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class MiniXceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.sep1 = SeparableConv2d(in_ch, out_ch)
        self.bn1  = nn.BatchNorm2d(out_ch)
        self.sep2 = SeparableConv2d(out_ch, out_ch)
        self.bn2  = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        res = self.residual(x)
        out = F.relu(self.bn1(self.sep1(x)))
        out = self.pool(self.bn2(self.sep2(out)))
        return F.relu(out + res)


class MiniXception(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
        )
        self.block1 = MiniXceptionBlock(8,  16)
        self.block2 = MiniXceptionBlock(16, 32)
        self.block3 = MiniXceptionBlock(32, 64)
        self.block4 = MiniXceptionBlock(64, 128)
        self.conv_out = nn.Conv2d(128, num_classes, 3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.block4(self.block3(self.block2(self.block1(x))))
        return self.gap(self.conv_out(x)).view(x.size(0), -1)
