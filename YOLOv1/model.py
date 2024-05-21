import torch
import torch.nn as nn

architecture_config = [
    # Tuple = (kernel_size, filters, stride, padding)
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],  # list=(Layer1, layer2, repeat_times)
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, split_size=7, num_boxes=2, num_classes=20):  # in_channels=3 because of RGB images
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.darknet = self.create_conv_layers(self.architecture)
        self.fcs = self.create_fcs(split_size, num_boxes, num_classes)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if isinstance(x, tuple):
                layers.append(CNNBlock(
                    in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3]
                ))
                in_channels = x[1]

            elif isinstance(x, str):
                layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

            elif isinstance(x, list):
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]
                for _ in range(num_repeats):
                    layers.append(CNNBlock(
                        in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3]
                    ))
                    layers.append(CNNBlock(
                        conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3]
                    ))
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        # Assuming the final feature map is 7x7x1024
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),  # according to the original paper it is 4096
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5)),  # reshaped to (S, S, 30)
        )

def test(split_size=7, num_boxes=2, num_classes=20):
    model = Yolov1(in_channels=3, split_size=split_size, num_boxes=num_boxes, num_classes=num_classes)
    x = torch.randn((2, 3, 448, 448))
    print(model(x))

test()
