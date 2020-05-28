from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import os.path as osp
import shutil

from .iotools import mkdir_if_missing

def choose_images(distmat, dataset, save_dir='./log/chosen_results', max_distance=200):
    """
    Choose images with low enough distance 'max_distance'
    
    Args:
    - distmat: distance matrix
    - dataset: 2-tuple containg (query, gallery), each contains a list of img_path, pid, camid
    - save_dir
    - max_distance
    """
    print("Started sorting images")

    mkdir_if_missing(save_dir)

    max_distance = distmat.mean()/2
    candidates = np.argwhere(distmat < max_distance)

    pid_dict = {}
    processed_images = set()
    num_images = 0 
    for cand in candidates:
        
        if (cand[0] > cand[1]) or (cand[0] in processed_images):
            # This means we are under the diagonal or we have added this image before, will create duplicates
            continue

        if cand[0] == cand[1]: # Query = gallery
            pid = cand[0]
            pid_dict[pid] = [pid]
            
            person_dir = osp.join(save_dir, "person{}".format(pid))
            mkdir_if_missing(person_dir)
            img_path = dataset[0][cand[0]][0]
            shutil.copy(img_path, person_dir + "/image{}.png".format(cand[0]))
            num_images += 1

        elif pid == cand[0]: # Candicate in gallery
            pid_dict[pid].append(cand[1])
            person_dir = osp.join(save_dir, "person{}".format(pid))
            img_path = dataset[0][cand[1]][0]
            shutil.copy(img_path, person_dir + "/image{}.png".format(cand[1]))
            processed_images.add(cand[1])
            num_images += 1
 
    assert (num_images == len(dataset[0]))
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
