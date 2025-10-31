import time
import copy
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR, OneCycleLR

from data_loader import cutmix
from Imagenette160_loader import * #import imagenette dataset

from progressive_model import *
from conv_cbam_mixer import *
from selfdistill_model import *
from model_trainer import *

#from convmixer_2m import *
#from efficientnet_b0 import *
#from mobilenet_v2 import MobileNetV2 
#from shufflenet_v2 import *
#from ghostnet import *
#from tinynet import *
#from conv_cbam_mixer import *
#from mobileone import *
#from fbnet import *
#from convnext import *
#from convnext_v2 import *
from mobilevit import *
#from starnet import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

dataset="Imagenette160"

patch_size = 5 # 160/5=32 
kernel_size = 5
num_classes = 10 

hidden_dim = 512  # Size of hidden layers
num_layers = 8     # Number of layers
batch_size = 32

epochs = 100
warmup_epochs = 5
lr_max=0.0001
lr_min=0.00001
dropout=0.1

   
############################### TRAINING #############################

train_loader, test_loader = get_dataloaders(data_root="data/imagenette/imagenette2-160", batch_size=batch_size)

from torchvision.models import mobilenet_v2
model_1 = MobileViT_Small(num_classes=num_classes).to(device)
#model_2 = ConvNextModel(num_classes=num_classes).to(device)
count_params_and_shapes(model_1, input_size=(1, 3, 160, 160))

#load_model_wts(model_1, dataset, patch_size, hidden_dim, num_layers)
            
modelWrapper = Cutmix_Model_Wrapper(model_1)
#count_params(modelWrapper)
  
optimizer = optim.Adam(modelWrapper.parameters(), lr=lr_min) #manages trainable params
trainer = ModelTrainer(modelWrapper, optimizer, num_classes, num_layers, lr_min=lr_min, lr_max=lr_max)

for epoch in range(epochs):
    lr = get_up_cos_lr(epochs, epoch, lr_min, lr_max, warmup_epochs)

    #rainer.train_model_by_layer(train_loader, num_layers, lr)               
    # Final fine-tuning (train all layers together after each layer is trained)    
    trainer.fine_tune_all(train_loader, lr=lr)
     
    test_accuracy = trainer.evaluate_model(test_loader, device)
    print(f"{epoch}, Test Accuracy: {test_accuracy:.2f}%")

    torch.save(modelWrapper.teacher.state_dict(), f"progressive_{dataset}_{patch_size}_{hidden_dim}_{num_layers}_weights.pth")
    


