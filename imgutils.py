
import numpy as np
import torch
from PIL import Image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # quick'n dirty reverse normalization
    img = np.squeeze(image.numpy())
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = img.transpose((1, 2, 0))

    ax.imshow(img)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image,
        returns a Pytorch tensor
    '''
    img = Image.open(image)
    
    # calculate scaling and cropping from original dimensions
    
    # resize to 255px keeping the form factor of the original
    original_width, original_height = img.size
    aspect_ratio = original_width / original_height

    if original_width > original_height:
        height = 255
        width = int(height * aspect_ratio)
    elif original_width < original_height:
        width = 255
        height = int(width / aspect_ratio)
    else:
        width = height = 255

    img = img.resize((width, height))
    
    # Get the dimensions of the new image size
    width, height = img.size
    
    # center crop of 224 x 224
    left, top, right, bottom = (width - 224)/2, (height - 224)/2, (width + 224)/2, (height + 224)/2
    img = img.crop((left, top, right, bottom))
    
    # Turn image into numpy array
    img = np.array(img)
    
    # Make the color channel dimension first instead of last
    img = img.transpose((2, 0, 1))
    
    # transform to [0..1] and normalize
    img = img/255
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225

    # cast to torch tensor
    image = torch.from_numpy(img)
    image = image.float()
    return image
