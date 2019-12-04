#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/aipnd-project/train.py
#                                                                             
# PROGRAMMER: Werner Ebert
# DATE CREATED: Nov 18, 2019                                
# REVISED DATE: 
# PURPOSE: train a new network on a dataset and save the model as a checkpoint. 
#     Command Line Arguments:
#     1. image file. string. --image with default value 'rose.jpeg'
#     2. checkpoint file. string. --checkpoint with default value 'checkpoint.pth'
#
##

# Imports python modules
import argparse
import json
import pandas as pd
import seaborn as sb

from fileutils import loadCheckpoint
from nnutils import predict

def get_input_args():
    """
    Retrieve command line arguments 
    Command Line Arguments:
      1. image file --image with default value 'rose.jpeg'
      2. checkpoint file. string. --checkpoint with default value 'checkpoints/checkpoint.pth'
      3. top_k. integer. --topk with default value 1
      4. name of json file with class->name mappings. string. --cnfile with default value 1
      5. platform --platform. str. platform to evaluate on (gpu|cpu) default cpu
    returns arguments as an ArgumentParser object.
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--image', type = str, default = 'rose.jpeg', help = 'image data file to predict')
    parser.add_argument('--checkpoint', type = str, default = 'checkpoints/checkpoint.pth', help = 'checkpoint file - default checkpoints/checkpoint.pth')
    parser.add_argument('--topk', type = int, default = 1, help = 'number of most probable classes - default 1')
    parser.add_argument('--cnfile', type = str, default = 'cat_to_name.json', help = 'name of json file with class->name mappings - default cat_to_name.json')
    parser.add_argument('--platform', type = str, default = 'cpu', help = 'platform to train on (cpu|gpu)')
    
    return parser.parse_args()

def main():
    args = get_input_args()

    with open(args.cnfile, 'r') as f:
        cat_to_name = json.load(f)
        
    model, _ = loadCheckpoint(args.checkpoint)

    top_p, top_class = predict(args.image, model, args.topk)
    df = pd.DataFrame(list(zip([cat_to_name[c].capitalize() for c in top_class], top_p)), columns=['Flower','Probability'])

    print(df)

if __name__ == '__main__':
    main()
