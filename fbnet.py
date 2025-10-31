from data_loader import *

import timm

from timm import list_models
print(list_models('*fbnet*')) #['fbnetc_100', 'fbnetv3_b', 'fbnetv3_d', 'fbnetv3_g']

def FBNet(num_classes):
    return timm.create_model("fbnetc_100", num_classes=num_classes, pretrained=False).to(device)

if __name__ == "__main__":
    model = FBNet(num_classes=10)
    count_params_and_shapes(model)

