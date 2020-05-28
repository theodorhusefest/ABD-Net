import pandas as pd
import os.path as osp
from torchreid.datasets.bases import BaseImageDataset

class PedestrianReID(BaseImageDataset):
    
    def __init__(self, root='./data/milestone3', video='video_1', verbose=True, **kwargs):
        super(PedestrianReID, self).__init__()
        self.dataset_dir = osp.join(root, video)
        self.images = osp.join(self.dataset_dir, 'yolov3_outputs/pedestrian_images')
        self.query = pd.read_csv(osp.join(self.dataset_dir, 'yolov3_outputs/query_list.csv'))
        self.gallery = pd.read_csv(osp.join(self.dataset_dir, 'yolov3_outputs/gallery_list.csv'))

        query = self._process_dir(self.images, self.query)
        gallery = self._process_dir(self.images, self.query)

        if verbose:
            print("=> {} images loaded".format(video))

        self.query = query
        self.gallery = gallery
        print(query[0])

    def _process_dir(self, dir_path, datasheet):
        
        dataset = []
        max_images = 50
        i = 0
        for _, row in datasheet.iterrows():
            img_path, pid, camid = row['file_path'], - 1, row['image_id']
            dataset.append((img_path, pid, camid ))
            i-=-1
            if i > max_images:
                return dataset
        return dataset
