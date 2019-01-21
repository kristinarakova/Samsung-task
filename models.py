import torch.nn as nn
from torchvision.models.resnet import resnet18

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
def build_model(device=1):
    '''Сreate a model for learning from the scratch'''
    model =  nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3)),
        nn.PReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3)),
        nn.PReLU(),
        nn.MaxPool2d((2,2)),
        nn.Dropout(0.2),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3)),
        nn.PReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3)),
        nn.PReLU(),
        nn.MaxPool2d((2,2)),
        nn.Dropout(0.2),
        Flatten(),
        nn.Linear(1600, 1024),
        nn.PReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        nn.PReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 2))
    model.cuda(device);
    return model
    
def build_pretrained_model(device=1):
    '''Build pretrained model, using ResNet18'''
    resnet = resnet18(pretrained=True)
    model = nn.Sequential(*list(resnet.children())[:7],
                       Flatten(),
                        nn.Dropout(0.2),
                        nn.Linear(4*256, 512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                       # nn.Linear(1024, 512),
                       # nn.ReLU(),
                       # nn.Dropout(0.2),
                        nn.Linear(512, 2))
    model.cuda(device);
    return model