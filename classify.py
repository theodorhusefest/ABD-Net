from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import argparse
import torch
import pandas as pd

from args import argument_parser, optimizer_kwargs
from torchreid import models
from torchreid.data_manager import SimpleImageDataManager

parser = argument_parser()
args = parser.parse_args()

"""
def argument_parser():
    parser= argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data', type=str, help="directory for yolo-images")


def create_dataset():

"""

def simple_image_dataset_kwargs(): 
    """
    Build kwargs for SimpleImageManager in data_manager.py
    """
    return {
        'target_names': ['pedestrianreid'],
        'root': 'data/milestone3',
    }



def classify_images(model, queryloader, galleryloader, use_gpu):
    """
    Query: Un-processed images/persons
    Gallery: Processed images/persons
    PseudoCode
    1. Load model 
    2. Use queryloader to get next query image
    3. Compare the queryimage with gallery and get a prediction
    5. Select a rank and give same ID to this rank
    """
    model.eval()
    pass
    """
    with torch.no_grad():

        enumerator = enumerate(queryloader[0])

        for batch_idx, package in enumerator:

            imgs, pids, camids, paths = package
            if use_gpu:
                imgs = imgs.cuda()

            features = model(imgs)[0]
            
    """

def main():
    "Started Main"

    use_gpu = True if torch.cuda.is_available() else False

    print("Initializing DataManager")
    dm = SimpleImageDataManager(use_gpu, **simple_image_dataset_kwargs())
    testloader = dm.return_testloader()

    print("Initializing Model")
    model = models.init_model(name= 'resnet50', num_classes = 30, loss= {'xent'}, use_gpu=use_gpu)


if __name__ == "__main__":
    main()