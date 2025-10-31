import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

from data_loader import cutmix


#------------------------------------------------------
"""
Provide two model instance and compete each other 
"""
#Branch out two model instances and pick the winner as the teacher, to distill the student with combo loss  
class SelfDistill_Cutmix_Model_Wrapper(nn.Module):  
    def __init__(self, model_branch1, model_branch2):
        super(SelfDistill_Cutmix_Model_Wrapper, self).__init__()
       
        self.branch1 = model_branch1
        self.branch2 = model_branch2        
        
        self.teacher = self.branch1
                
    def forward(self, x):
        return self.branch1(x)

    def cutmix(self, x, y):
        x_cutmix, targets_a, targets_b, lam = cutmix(x, y, alpha=1.0)
        out1 = self.branch1(x_cutmix)
        out2 = self.branch2(x_cutmix)
        return out1, out2, targets_a, targets_b, lam

    def distillation_loss(self, logits1, logits2, targets_a, targets_b, lam, criterion, temp=2.0):
        targets_a, targets_b = targets_a.to(logits1.device), targets_b.to(logits1.device)

        loss1 = lam * criterion(logits1, targets_a) + (1 - lam) * criterion(logits1, targets_b)
        loss2 = lam * criterion(logits2, targets_a) + (1 - lam) * criterion(logits2, targets_b)

        with torch.no_grad():
            is_branch1_better = loss1 < loss2

        if is_branch1_better:
            teacher_logits, student_logits = logits1, logits2
            teacher_loss, student_loss = loss1, loss2
        else:
            teacher_logits, student_logits = logits2, logits1
            teacher_loss, student_loss = loss2, loss1

        distill_loss = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(student_logits / temp, dim=1),
            F.softmax(teacher_logits / temp, dim=1)
        ) * (temp ** 2)

        total_loss = 0.5 * (student_loss + teacher_loss) + distill_loss
        return total_loss, teacher_logits

    def get_loss_logit(self, x, y, criterion):
        logits1, logits2, targets_a, targets_b, lam = self.cutmix(x, y)
        #print("logits1 shape:", logits1.shape)
        #print("logits2 shape:", logits2.shape)
        #print("targets_a shape:", targets_a.shape)
        #print("targets_b shape:", targets_b.shape)
        #print("lam:", lam)
    
        loss, out = self.distillation_loss(logits1, logits2, targets_a, targets_b, lam, criterion)
        return loss, out
    
class SelfDistill_Model_Wrapper(nn.Module):  
    def __init__(self, model_branch1, model_branch2):
        super(SelfDistill_Model_Wrapper, self).__init__()
       
        self.branch1 = model_branch1
        self.branch2 = model_branch2        
        
        self.teacher = self.branch1
                
    def forward(self, x):
        return self.branch1(x)

    def distillation_loss(self, x, y, criterion, temp=2.0):
        logits1 = self.branch1(x)
        logits2 = self.branch2(x)

        loss1 = criterion(logits1, y) 
        loss2 = criterion(logits2, y) 
    
        if loss1.item() < loss2.item():
            teacher_logits, student_logits = logits1, logits2
            teacher_loss = loss1
            student_loss = loss2
            
            self.teacher = self.branch1
            
        else:
            teacher_logits, student_logits = logits2, logits1            
            teacher_loss = loss2
            student_loss = loss1            
            self.teacher = self.branch2            
        
       
        distill_loss = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(student_logits / temp, dim=1), F.softmax(teacher_logits / temp, dim=1)
        ) * (temp ** 2)
        
        # Final loss: Student's own loss + Distillation loss
        total_loss = 0.5*(student_loss + teacher_loss) + distill_loss
        #print(student_loss.item(), distill_loss.item(), teacher_loss.item())
                    
        return total_loss, teacher_logits    

    def set_trainable_layers_params(self, curr_layer_index=None):
        self.branch1.set_trainable_layers_params(curr_layer_index)
        self.branch2.set_trainable_layers_params(curr_layer_index)

    def get_loss_logit(self, x, y, criterion):
        loss, out = self.distillation_loss(x, y, criterion)
        return loss, out
    
#--------------------------------------------------

#Wrap a model with cutmix
class Cutmix_Model_Wrapper(nn.Module):  
    def __init__(self, model):
        super(Cutmix_Model_Wrapper, self).__init__()
        
        self.teacher = model 
    
    def forward(self, x):
        return self.teacher(x)
        
    def cutmix(self, x, y):
        x_cutmix, self.targets_a, self.targets_b, self.lam = cutmix(x, y, alpha=1.0)

        return self.teacher(x_cutmix)
        
    def set_trainable_layers_params(self, curr_layer_index=None):
        self.teacher.set_trainable_layers_params(curr_layer_index)

    def get_loss_logit(self, x, y, criterion):
        logits = self.cutmix(x, y)

        #print("logits shape:", logits.shape)
        #print("targets_a min/max:", self.targets_a.min().item(), self.targets_a.max().item())
        #print("targets_b min/max:", self.targets_b.min().item(), self.targets_b.max().item())
        #print("n_classes:", logits.shape[1])

        loss = self.lam * criterion(logits, self.targets_a) + (1 - self.lam) * criterion(logits, self.targets_b)
        return loss, logits

#Wrap a model with cutmix
class Model_Wrapper(nn.Module):  
    def __init__(self, model):
        super(Model_Wrapper, self).__init__()
        
        self.teacher = model 
    
    def forward(self, x):
        return self.teacher(x)
        
        
    def set_trainable_layers_params(self, curr_layer_index=None):
        self.teacher.set_trainable_layers_params(curr_layer_index)

    def get_loss_logit(self, x, y, criterion):        
        logits = self.teacher(x)
        loss = criterion(logits, y) 
        return loss, logits
    
