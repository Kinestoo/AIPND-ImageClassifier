import torch
from torch import nn, optim
from torchvision import models
from collections import OrderedDict

from imgutils import process_image

def config_pretrained_model(arch, nhidden, noutput, platform):
    # load, freeze and setup the model architecture given by arch
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(1024, nhidden)),
                              ('relu1', nn.ReLU()),
                              ('d_out1',nn.Dropout(0.2)),
                              ('fc2', nn.Linear( nhidden, 1024)),
                              ('relu2', nn.ReLU()),
                              ('d_out2',nn.Dropout(0.2)),
                              ('fc3', nn.Linear(1024, noutput)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    else:
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(nn.Linear(512 * 7 * 7, nhidden),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(nhidden, noutput),
                                      nn.LogSoftmax(dim=1))
    model.classifier = classifier
    model.arch = arch
    model.nhidden = nhidden
    model.noutput = noutput
    model.platform = platform
    model.device = torch.device("cuda" if torch.cuda.is_available() and platform == "gpu" else "cpu")
    model.to(model.device);

    return model
    
def accuracy4batch(model, testloader, criterion):
    """save a model checkpoint

    INPUT:
    model: pytorch nn model. 
    testloader: DataLoader. test data set
    criterion: criterion. loss criterion
    device: torch.device. device on which model/data is based

    OUTPUT: 
    accuracy: float in [0:1]. percenct proportion of correct classifications in testloader
    test_loss: float. absolute error
    """
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    accuracy = accuracy/len(testloader)
    
    return accuracy, test_loss
    
def predict(image_path, model, topk = 1):
    """ predict an image's k most probable classes based on a given model

    INPUT:
    image: 
    model: pytorch nn model. 
    topk: number of classes to predict

    OUTPUT: 
    top_p : array of probabilities
    top_k : array of classes
    """
    # configure model
    model.eval()

    # pass image through model
    
    with torch.no_grad():
        image = process_image(image_path)
        image = image.to(model.device)
        image.unsqueeze_(0)
        outputs = model(image)
        probs, classes = torch.exp(outputs).topk(topk)

        idx_to_class = {value: key for key, value in model.class_to_idx.items()}
        classes = [idx_to_class[idx] for idx in classes[0].tolist()]
        
        return probs[0].tolist(), classes

