import pandas as pd
import os.path as osp
from torchreid.datasets.bases import BaseImageDataset

class PedestrianReID(BaseImageDataset):
    
    def __init__(self, root='data/milestone3', video='video_1', verbose=True, **kwargs):
        super(PedestrianReID, self).__init__()
        self.dataset_dir = osp.join(root, video)
        self.images = osp.join(self.dataset_dir, 'yolov3_outputs/pedestrian_images')
        self.query = pd.read_csv(osp.join(self.dataset_dir, 'yolov3_outputs/query_list.csv'))
        self.gallery = pd.read_csv(osp.join(self.dataset_dir, 'yolov3_outputs/gallery_list.csv'))
