#####################################################
#
# replace all nn layers to custom triton layers
#
#####################################################
import torch
from TritonMGP import nn as mgp_nn
import time

class BasicBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = mgp_nn.TritonConv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = mgp_nn.TritonBatchNorm2d(out_channels)
        self.relu = mgp_nn.TritonReLU()
        self.conv2 = mgp_nn.TritonConv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = mgp_nn.TritonBatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out

class TritonResNet18(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(TritonResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = mgp_nn.TritonConv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = mgp_nn.TritonBatchNorm2d(64)
        self.relu = mgp_nn.TritonReLU()
        self.maxpool = mgp_nn.TritonMaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = mgp_nn.TritonAvgPool2d(kernel_size=7, stride=1)
        self.fc = mgp_nn.TritonLinear(512, num_classes)
        
        self.done = 0
        self.num = 0 # function number
        self.run = 1 # function number

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = torch.nn.Sequential(
                mgp_nn.TritonConv2d(self.in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                mgp_nn.TritonBatchNorm2d(out_channels),
            )

        layers = []
        layers.append(BasicBlock(self.in_channels,
                      out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return torch.nn.Sequential(*layers)
    
    def print(self, x):
        if self.done:
            return
        
        arr = ["Initial        ",
               "Conv2d         ",
               "BatchNorm2d    ",
               "ReLU           ",
               "MaxPool2d      ",
               "AveragePool2d  ",
               "Linear         "
               ]
        
        print(f"{arr[self.num]}", end="")
        
        try:
            if self.num < 5:
                for i in range(0,5):
                    print(f"{float(x[0][0][i][i]):.5f}", end=" ")
                print("...") 
            elif self.num == 5:
                for i in range(0,5):
                    print(f"{float(x[i][i][0][0]):.5f}", end=" ")
                print("...") 
            else:
                for i in range(0,5):
                    print(f"{float(x[i][i]):.5f}", end=" ")
                print("...") 
        except:
            print("Wrong tensor dimentions!")
            
        self.num += 1

    def forward(self, x):
        print(f"Run {self.run}", flush=True)
        self.run += 1

        def _now():
            # helper to get a timestamp (and sync if needed)
            t = time.perf_counter()
            if x.is_cuda:
                torch.cuda.synchronize()
                t = time.perf_counter()
            return t

        last_ts = _now()

        # ─── conv1 ───────────────────────────────────────────────────────────────
        x = self.conv1(x)
        ts = _now()
        print(f"  conv1   : {(ts - last_ts) * 1000.0:7.2f} ms", flush=True)
        last_ts = ts

        # ─── bn1 ─────────────────────────────────────────────────────────────────
        x = self.bn1(x)
        ts = _now()
        print(f"  bn1     : {(ts - last_ts) * 1000.0:7.2f} ms", flush=True)
        last_ts = ts

        # ─── relu ───────────────────────────────────────────────────────────────
        x = self.relu(x)
        ts = _now()
        print(f"  relu    : {(ts - last_ts) * 1000.0:7.2f} ms", flush=True)
        last_ts = ts

        # ─── maxpool ───────────────────────────────────────────────────────────
        x = self.maxpool(x)
        ts = _now()
        print(f"  maxpool : {(ts - last_ts) * 1000.0:7.2f} ms", flush=True)
        last_ts = ts

        # ─── layer1 ─────────────────────────────────────────────────────────────
        x = self.layer1(x)
        ts = _now()
        print(f"  layer1  : {(ts - last_ts) * 1000.0:7.2f} ms", flush=True)
        last_ts = ts

        # ─── layer2 ─────────────────────────────────────────────────────────────
        x = self.layer2(x)
        ts = _now()
        print(f"  layer2  : {(ts - last_ts) * 1000.0:7.2f} ms", flush=True)
        last_ts = ts

        # ─── layer3 ─────────────────────────────────────────────────────────────
        x = self.layer3(x)
        ts = _now()
        print(f"  layer3  : {(ts - last_ts) * 1000.0:7.2f} ms", flush=True)
        last_ts = ts

        # ─── layer4 ─────────────────────────────────────────────────────────────
        x = self.layer4(x)
        ts = _now()
        print(f"  layer4  : {(ts - last_ts) * 1000.0:7.2f} ms", flush=True)
        last_ts = ts

        # ─── avgpool ─────────────────────────────────────────────────────────────
        x = self.avgpool(x)
        ts = _now()
        print(f"  avgpool : {(ts - last_ts) * 1000.0:7.2f} ms", flush=True)
        last_ts = ts

        # ─── flatten & fc ────────────────────────────────────────────────────────
        x = torch.flatten(x, 1)
        x = self.fc(x)
        ts = _now()
        print(f"  fc      : {(ts - last_ts) * 1000.0:7.2f} ms", flush=True)

        self.done = 1
        return x
