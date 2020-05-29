import pandas as pd
import os.path as osp
from torchreid.datasets.bases import BaseImageDataset

class PedestrianReID(BaseImageDataset):
    
    def __init__(self, root='./data/milestone3', video='video_1', verbose=True, **kwargs):
        super(PedestrianReID, self).__init__()
        self.dataset_dir = osp.join(root, video)
        self.images = osp.join(self.dataset_dir, 'yolov3_outputs/pedestrian_images')
        self.query = pd.read_csv(osp.join(self.dataset_dir, 'yolov3_outputs/query_list.csv'))

        query = self._process_dir(self.images, self.query)
        gallery = self._process_dir(self.images, self.query)

        if verbose:
            print("\n=> {} images loaded".format(video))
            print("Loaded {} images".format(len(query)))

        self.query = query
        self.gallery = gallery

    def _process_dir(self, dir_path, datasheet):
        
        dataset = []
        for _, row in datasheet.iterrows():
            img_path, pid, camid = row['file_path'], - 1, row['image_id']
            dataset.append((img_path, pid, camid ))
        return dataset
