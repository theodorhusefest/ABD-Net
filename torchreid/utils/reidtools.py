from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
import os
import os.path as osp
import shutil

from .iotools import mkdir_if_missing

def choose_images(distmat, dataset, csv = None, save_dir='./output/', video='video_1'):
    """
    Choose images with low enough distance 'max_distance'
    
    Args:
    - distmat: distance matrix
    - dataset: 2-tuple containg (query, gallery), each contains a list of img_path, pid, camid
    - save_dir
    """
    print("\Started sorting images")
    
    if csv:
        datasheet = pd.read_csv(csv)
    else:
        datasheet = pd.read_csv("./data/milestone3/{}/yolov3_outputs/query_list.csv".format(video))
    datasheet['ID'] = -1

    mkdir_if_missing(osp.join(save_dir, video))

    max_distance = distmat.mean()/2
    matched_images = np.argwhere(distmat < max_distance)

    processed_images = set()
    num_images = 0 
    for matched_image in matched_images:
        gallery_img, query_img = matched_image

        if (gallery_img > query_img) or (gallery_img in processed_images) or (query_img in processed_images):
            # This means we are under the diagonal or we have added this image before, will create duplicates
            continue

        if gallery_img == query_img:
            """
            If we find a person that has not been seen, we create a new pid and folder
            """

            pid = gallery_img
            person_dir = osp.join(save_dir, video, "person{}".format(pid))
            mkdir_if_missing(person_dir)

            img_path = dataset[0][gallery_img][0]
            datasheet.loc[datasheet['file_path'] == img_path, 'ID'] = int(pid)

            shutil.copy(img_path, person_dir + "/image{}.png".format(gallery_img))
            num_images += 1

        elif pid == gallery_img:
            """
            If the query image is assosiated with 
            """
            
            person_dir = osp.join(save_dir, video, "person{}".format(pid))
            img_path = dataset[0][query_img][0]
            datasheet.loc[datasheet['file_path'] == img_path, 'ID'] = int(pid)

            shutil.copy(img_path, person_dir + "/image{}.png".format(query_img))

            processed_images.add(query_img)
            num_images += 1
 
    datasheet.to_csv( osp.join(save_dir,video) + "/{}_data.csv".format(video) )
    print("Sorted {} images, {} images in total".format(num_images, len(dataset[0])))
    

def visualize_ranked_results(distmat, dataset, save_dir='log/ranked_results', topk=20):
    """
    Visualize ranked results

    Support both imgreid and vidreid

    Args:
    - distmat: distance matrix of shape (num_query, num_gallery).
    - dataset: a 2-tuple containing (query, gallery), each contains a list of (img_path, pid, camid);
               for imgreid, img_path is a string, while for vidreid, img_path is a tuple containing
               a sequence of strings.
    - save_dir: directory to save output images.
    - topk: int, denoting top-k images in the rank list to be visualized.
    """
    num_q, num_g = distmat.shape

    print("Visualizing top-{} ranks".format(topk))
    print("# query: {}\n# gallery {}".format(num_q, num_g))
    print("Saving images to '{}'".format(save_dir))
    
    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)
    
    indices = np.argsort(distmat, axis=1)
    mkdir_if_missing(save_dir)

    def _cp_img_to(src, dst, rank, prefix):
        """
        - src: image path or tuple (for vidreid)
        - dst: target directory
        - rank: int, denoting ranked position, starting from 1
        - prefix: string
        """
        if isinstance(src, tuple) or isinstance(src, list):
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src))
            shutil.copy(src, dst)

    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = query[q_idx]
        qdir = osp.join(save_dir, osp.basename(qimg_path))
        mkdir_if_missing(qdir)
        _cp_img_to(qimg_path, qdir, rank=0, prefix='query')

        rank_idx = 1
        for g_idx in indices[q_idx,:]:
            gimg_path, gpid, gcamid = gallery[g_idx]
            invalid = (qpid == gpid) & (qcamid == gcamid)
            if not invalid:
                _cp_img_to(gimg_path, qdir, rank=rank_idx, prefix='gallery')
                rank_idx += 1
                if rank_idx > topk:
                    break

    print("Done")
