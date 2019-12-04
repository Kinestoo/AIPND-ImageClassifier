import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.vgg import VGG, make_layers, cfg, vgg16
from torchvision.models.densenet import densenet121

from workspace_utils import active_session
from fileutils import saveCheckpoint
from nnutils import accuracy4batch

class FlowerPowerVGG(VGG):
    def __init__(self, num_classes=1000, init_weights=True):
        super().__init__(make_layers(cfg['D']))

        # self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(cat_to_name)),
            nn.LogSoftmax(dim=1))
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

                
def train(model, trainloader, testloader, criterion, optimizer, cp_dir="", epochs=1, print_every=5):

    with active_session():

        steps = 0
        running_loss = 0
        print_every = 5
        
        for epoch in range(epochs):
            model.train()
            for inputs, labels in trainloader:
                steps += 1

                # Move input and label tensors to the default device
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                optimizer.zero_grad()
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    
                    accuracy, test_loss = accuracy4batch(model, testloader, criterion)
                    
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Test loss: {test_loss/len(testloader):.3f}.. "
                          f"Test accuracy: {accuracy:.3f}")
                    
                    running_loss = 0
                    model.train()
                    
    saveCheckpoint(cp_dir, f'checkpoint.pth', model, optimizer)

