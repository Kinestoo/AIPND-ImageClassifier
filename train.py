#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/aipnd-project/train.py
#                                                                             
# PROGRAMMER: Werner Ebert
# DATE CREATED: Nov 18, 2019                                
# REVISED DATE: 
# PURPOSE: train a new network on a dataset and save the model as a checkpoint. 
#     Command Line Arguments:
#     1. Image Folder as --dir with default value 'images'
#     2. CNN Model Architecture as --arch with default value 'vgg'
#     3. Text File with Class-2-Names mapping as --names with default value 'class_to_names.json'
#
##

# Imports python modules
import argparse
import os
import json
import torch
from torch import nn, optim
import numpy as np
import pandas as pd
import seaborn as sb

import flowerpower
from fileutils import createDataloaders, saveCheckpoint, loadCheckpoint
from nnutils import config_pretrained_model, predict

def get_input_args():
    """
    Retrieve command line arguments 
    Command Line Arguments:
      1. data directory as --data_dir with default value './flowers'
      2. directory for saving checkpoints --cp_dir with default value './checkpoints'
      3. name of json file with class to name mappings
      4. network architecture --arch with default value 'vgg16' (other: densenet121)
      5. learning rate --lr. floar. default 0.001
      6. number of hidden layers. int. --nhidden default 1024
      7. number of epochs --nepochs. int. default 2
      8. platform --platform. str. platform to train on (gpu|cpu) default cpu
    returns arguments as an ArgumentParser object.
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type = str, default = 'flowers/', help = 'path to the image folder')
    parser.add_argument('--cp_dir', type = str, default = 'checkpoints/', help = 'path to checkpoint folder')
    parser.add_argument('--mapfile', type = str, default = 'cat_to_name.json', help = 'json file that contains a dictionary')
    parser.add_argument('--arch', type = str, default = 'vgg16', help = '(vgg16|densenet121)')
    parser.add_argument('--lr', type = float, default = 0.001, help = 'learning rate')
    parser.add_argument('--nhidden', type = int, default = 1024, help = 'number of hidden layers')
    parser.add_argument('--nepochs', type = int, default = 2, help = 'number of epochs')
    parser.add_argument('--platform', type = str, default = 'cpu', help = 'device to train on (cpu|gpu)')
    
    return parser.parse_args()

def main():
    args = get_input_args()

    # load training data
    dataloaders, image_datasets = createDataloaders(args.data_dir)
    with open(args.mapfile, 'r') as f:
        cat_to_name = json.load(f)

    # create model & optimizer according to cl parameters
    model = config_pretrained_model(args.arch, args.nhidden, len(cat_to_name), args.platform)
    model.class_to_idx = image_datasets['training'].class_to_idx

    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    optimizer.lr = args.lr
    criterion = nn.NLLLoss()

    flowerpower.train(model, dataloaders["training"], dataloaders["validation"], criterion, optimizer, args.cp_dir, args.nepochs)
    
    print("Prediction from in-mem model")
    top_p, top_class = predict("rose.jpeg", model, 5)
    df = pd.DataFrame(list(zip([cat_to_name[c].capitalize() for c in top_class], top_p)), columns=['Flower','Probability'])
    print(df)
    
    print("Prediction from checkpt model")
    model, _ = loadCheckpoint("checkpoints/checkpoint.pth")
    top_p, top_class = predict("rose.jpeg", model, 5)
    df = pd.DataFrame(list(zip([cat_to_name[c].capitalize() for c in top_class], top_p)), columns=['Flower','Probability'])
    print(df)


if __name__ == '__main__':
    main()

