import os
import errno
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from nnutils import config_pretrained_model

def createDataloaders(data_dir):
    """ create dataloaders for training/validation/test datasets according to std directory structure 

    INPUT:
    data_dir: string. name of base directory

    OUTPUT: 
    dataloaders: dict of torch Dataloaders keys: (training|validation|testing)
    image_folders: dict of ImageFolders keys: (training|validation|testing)
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
        "training" : transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])]),

        "validation" : transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),   


        "testing" : transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),   

    }

    image_datasets = {
        "training" : datasets.ImageFolder(train_dir, transform = data_transforms["training"]),
        "validation" : datasets.ImageFolder(valid_dir, transform = data_transforms["validation"]),
        "testing" : datasets.ImageFolder(test_dir, transform = data_transforms["testing"])
    }

    dataloaders = {
        "training" : torch.utils.data.DataLoader(image_datasets["training"], batch_size=64, shuffle=True),
        "validation" : torch.utils.data.DataLoader(image_datasets["validation"], batch_size=64),
        "testing" : torch.utils.data.DataLoader(image_datasets["testing"], batch_size=64)
    }
    
    return dataloaders, image_datasets
    
def saveCheckpoint(cp_dir, filename, model, optimizer):
    """save a model checkpoint

    INPUT:
    filename: string. name of checkpoint file
    nInput: int. number of inputs
    nOutput: int. number of outputs (classes)
    nInner: int. number of hidden units
    modelArch: str. model architecture
    model: torch.model. model to save
    optimizer: torch.optim. optimizer to resume training

    OUTPUT: 
    no output.
    """
    # capture these layers from the model's classifier
    checkpoint = {'model_arch': model.arch,
                  'model_platform': model.platform,
                  'model_noutput': model.noutput,
                  'model_nhidden': model.nhidden,
                  'model_state_dict': model.state_dict(),
                  'model_classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'optim_state_dict': optimizer.state_dict(),
                  'optim_lr': optimizer.lr
                 # 'epoch': epoch
                 }
    
    # create directory if it doesn't exist 
    os.makedirs(cp_dir, exist_ok=True)
    torch.save(checkpoint, cp_dir + filename)
    
def loadCheckpoint(filename):
    """load a pretrained vgg16 model checkpoint and optimizer to resume training
       features of pretrained model are frozen, only classifier is restored
       model is mapped to device type it was stored in

    INPUT:
    filename: string. name of checkpoint file

    OUTPUT: 
    model: torch.model. model to save
    optimizer: torch.optim. optimizer to resume training
    """
    
    checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)

    model = config_pretrained_model(checkpoint['model_arch'], checkpoint['model_nhidden'], checkpoint['model_noutput'], checkpoint['model_platform'])

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.class_to_idx = checkpoint["class_to_idx"]

    optimizer = optim.Adam(model.classifier.parameters(), checkpoint['optim_lr'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    optimizer.lr = checkpoint['optim_lr']
    
    return model, optimizer #, epoch