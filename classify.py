from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import argparse
import torch
import pandas as pd
import numpy as np

from args import argument_parser, optimizer_kwargs
from torchreid import models
from torchreid.data_manager import SimpleImageDataManager
from torchreid.utils.reidtools import visualize_ranked_results

parser = argument_parser()
args = parser.parse_args()

"""
def argument_parser():
    parser= argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data', type=str, help="directory for yolo-images")


def create_dataset():

"""

def simple_image_dataset_kwargs(parsed_args): 
    """
    Build kwargs for SimpleImageManager in data_manager.py
    """
    return {
        'target_names': parsed_args.target_names,
        'root': parsed_args.root,
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
    with torch.no_grad():
        qf, q_pids, q_camids, q_paths = [], [], [], []

        enumerator = enumerate(queryloader)
        for batch_idx, package in enumerator:
            imgs, pids, camids, paths = package
            if use_gpu:
                imgs = imgs.cuda()

            features = model(imgs)[0]

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(camids)
            q_paths.extend(paths)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
            
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))


        gf, g_pids, g_camids, g_paths = [], [], [], []

        enumerator = enumerate(galleryloader)
        for batch_idx, package in enumerator:
            imgs, pids, camids, paths = package
            if use_gpu:
                imgs = imgs.cuda()

            features = model(imgs)[0]

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(camids)
            g_paths.extend(paths)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
            
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

        m, n = qf.size(0), gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.numpy()

        return distmat

            

def main():
    "Started Main"
    global args

    use_gpu = True if torch.cuda.is_available() else False

    print("Initializing DataManager")
    dm = SimpleImageDataManager(use_gpu, **simple_image_dataset_kwargs(args))
    testloader_dict = dm.return_testloader()

    print("Initializing Model")
    model = models.init_model(name= 'resnet50', num_classes = 30, loss= {'xent'}, use_gpu=use_gpu, args=vars(args))

    if args.load_weights:
        try:
            checkpoint = torch.load(args.load_weights)
        except Exception as e:
            print(e)
            checkpoint = torch.load(args.load_weights, map_location= torch.device('cpu'))
        
        pretrain_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        print("Loaded pretrained weights from '{}'".format(args.load_weights))

    model = model.cuda()
    print("Staring ReID")
    for name in args.target_names:
        queryloader = testloader_dict[name]['query']
        galleryloader = testloader_dict[name]['gallery']
        distmat = classify_images(model, queryloader, galleryloader, use_gpu)

        visualize_ranked_results(distmat, dm.return_testdataset_by_name(name),
        save_dir= osp.join(args.save_dir, 'ranked_results', name), topk=5)


if __name__ == "__main__":
    main()
