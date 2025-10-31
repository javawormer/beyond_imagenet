from data_loader import *

import torch
import torch.nn as nn


        
class ShuffleNetV2Block(nn.Module):
    def __init__(self, inp, outp, stride):
        super().__init__()
        self.stride = stride
        branch_features = outp // 2

        if stride == 1:
            assert inp == outp

        # First branch for stride > 1
        if stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride=stride, padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                
                nn.Conv2d(inp, branch_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = None

        # Second branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if stride > 1 else inp // 2, branch_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_features),
            
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features, branch_features, 3, stride=stride, padding=1, groups=branch_features, bias=False),
            nn.BatchNorm2d(branch_features),
            
            nn.Conv2d(branch_features, branch_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return out

class ShuffleNet_2M_Model(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        #self.stage_out_channels = [24, 116, 232, 464, 1024] #this match with the official model in torchvision
        self.stage_out_channels = [64, 192, 360, 600, 1024] 

        # Initial conv layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.stage_out_channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[0]),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build stages
        self.stage2 = self._make_stage(self.stage_out_channels[0], self.stage_out_channels[1], 4)
        self.stage3 = self._make_stage(self.stage_out_channels[1], self.stage_out_channels[2], 8)
        self.stage4 = self._make_stage(self.stage_out_channels[2], self.stage_out_channels[3], 4)

        # Final convolution
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.stage_out_channels[3], self.stage_out_channels[4], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[4]),
            nn.ReLU(inplace=True)
        )

        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.stage_out_channels[4], num_classes)

    def _make_stage(self, inp, outp, repeat):
        layers = [ShuffleNetV2Block(inp, outp, stride=2)]
        for _ in range(repeat - 1):
            layers.append(ShuffleNetV2Block(outp, outp, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.globalpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)        

if __name__ == "__main__":
    from torchinfo import summary
    
    model = ShuffleNet_2M_Model(num_classes=10).to(device)
    print("-------------------------My ShuffleNet Model----------------------")
    summary(model, input_size=(1, 3, 160, 160))
    
    """
    from torchvision.models import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
    ref_model = shufflenet_v2_x1_0(num_classes=10, weights=None)  # `weights=None` disables pretrained weights
    print("-------------------------Torchvision ShuffleNet Model----------------------")
    summary(ref_model, input_size=(1, 3, 160, 160))

    """    
    
    count_params_and_shapes(model)
    #count_params(model)
    
    dummy = torch.randn(1, 3, 160, 160).to(device)
    out = model(dummy)
    print(out.shape)  # should be [1, 10]
   
