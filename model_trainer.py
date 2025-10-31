import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################## TRAINER ###############################

# A training wrapper that train a SelfDistill model

class ModelTrainer():
    def __init__(self, modelWrapper, optimizer, num_classes, num_layers, epochs=1, lr_min=0.0001, lr_max=0.001):
        self.modelWrapper = modelWrapper

        self.num_classes = num_classes        
        self.num_layers = num_layers        

        self.epochs = epochs
        self.lr_min = lr_min
        self.lr_max = lr_max

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.optimizer = optimizer

    def set_opt_lr(self, lr):
        self.optimizer.zero_grad() #reset the computation graph

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
          
    def train_layer(self, layer_idx, dataloader, lr, epochs=1):        
        self.modelWrapper.set_trainable_layers_params(layer_idx) #only current layer params is for backprop
        self.modelWrapper.train()

        training_pts = len(dataloader.dataset)

        for epoch in range(epochs):
            predicts = 0
            start = time.time()

            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)
                
                loss, out = self.modelWrapper.get_loss_logit(x, y, self.criterion) 
                        
                self.set_opt_lr(lr) #also reset opt grad to 0
                loss.backward()                
                self.optimizer.step()
            
                predicts += (out.max(1)[1] == y).sum().item()

            accu = 100*predicts / training_pts
            print(f"{epoch}_{layer_idx}: {time.time() - start:.1f}, lr:{lr:.5f}, Layer {layer_idx} Loss: {loss.item():.4f} | Acc: {accu:.2f}% per {predicts}")
    
    #Train the model layer by layer across all the layers, then save the model params after its final layer is trained.
    def train_model_by_layer(self, train_loader, num_layers, lr):        
        for i in range(num_layers):
            layer_lr = lr #lr * 0.9**i #decrease LR over each layer
            self.train_layer(i, train_loader, layer_lr, epochs=1)
        
        torch.save(self.modelWrapper.state_dict(), f"progressive_imagenet_{hidden_dim}_{num_layers}_weights.pth")
        
    def fine_tune_all(self, dataloader, epochs=1, lr=1e-4):
        self.turn_on_all_params() #all params are turned on for backprop
        self.modelWrapper.train()

        training_pts = len(dataloader.dataset)
        for epoch in range(epochs):
            predicts = 0
            start = time.time()

            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)
                
                #print(x.shape, y.shape)
                loss, out = self.modelWrapper.get_loss_logit(x, y, self.criterion) 
                                    
                self.set_opt_lr(lr) #also reset opt grad to 0
                loss.backward()                
                self.optimizer.step()                
                
                predicts += (out.max(1)[1] == y).sum().item()


            accu = 100.0*predicts/training_pts
            print(f"LR: {lr:.6f} | Time: {time.time() - start:.1f} | Loss: {loss.item():.4f} | Accu: {accu:.4f}")

    def turn_on_all_params(self):
        # Final training stage: unfreeze everything and train end-to-end
        for param in self.modelWrapper.parameters():
            param.requires_grad = True
                

    def evaluate_model(self, test_loader, device):
        teacher = self.modelWrapper.teacher
        teacher.eval()  # evaluation mode
    
        correct = 0
        total_samples = len(test_loader.dataset)
    
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = teacher(inputs)  # [B, 200]

                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()
    
        accuracy = 100.0 * correct / total_samples
        return accuracy

    #The model predicts on both the original and flipped image for test dataset.
    def tta_predict(self, teacher, image):
        flips = [image, torch.flip(image, dims=[-1])]
        outputs = [teacher(f) for f in flips]
        return torch.stack(outputs).mean(dim=0)
        
