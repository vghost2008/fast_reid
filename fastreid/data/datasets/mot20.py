# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class MOT20(ImageDataset):
    """MOT20.

    Reference:
        Dendorfer, P., Rezatofighi, H., Milan, A., Shi, J., Cremers, D., Reid, I., Roth, S., Schindler, K. & Leal-Taixé, L. MOT20: A benchmark for multi object tracking in crowded scenes. arXiv:2003.09003[cs], 2020., (arXiv: 2003.09003).

    URL: `<https://motchallenge.net/data/MOT20/>`_

    Dataset statistics:
        - identities: ?
        - images: ?
    """
    _junk_pids = [0, -1]
    dataset_dir = 'MOT20'
    dataset_url = ''  # 'https://motchallenge.net/data/MOT20.zip'
    dataset_name = "MOT20"

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        print(f"Root: {self.root}")
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.train_dir1 = osp.join(self.data_dir, 'bounding_box_train1')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.extra_gallery = False

        required_files = [
            self.data_dir,
            self.train_dir,
            # self.query_dir,
            # self.gallery_dir,
        ]

        self.check_before_run(required_files)

        if osp.exists(self.train_dir1):
            train = lambda: self.process_dir(self.train_dir)+self.process_dir(self.train_dir1)
        else:
            train = lambda: self.process_dir(self.train_dir)
        query = lambda: self.process_dir(self.query_dir, is_train=False)
        gallery = lambda: self.process_dir(self.gallery_dir, is_train=False) + \
                          (self.process_dir(self.extra_gallery_dir, is_train=False) if self.extra_gallery else [])

        super(MOT20, self).__init__(train, query, gallery, **kwargs)
        print(f"Total find {len(self.data)} datas")

    '''def process_dir(self, dir_path, is_train=True):

        print(f"Process dir {dir_path}")
        img_paths = glob.glob(osp.join(dir_path, '*.bmp'))

        data = []
        for i,img_path in enumerate(img_paths):
            t_data = os.path.basename(img_path).split("_")
            pid = int(t_data[0])


            if t_data[-4][:1] != "c":
                print(f"ERROR,{img_path}")
                exit(0)

            camid = int(t_data[-4][1:])

            if i%1000==0:
                print(i,img_path)
                print(i,pid,camid)

            if pid == -1:
                continue  # junk images are just ignored
            # assert 0 <= pid   # pid == 0 means background
            # assert 1 <= camid <= 5
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))
        print(f"Total find {len(data)} datas in dir {dir_path}")
        return data'''

    def process_dir(self, dir_path, is_train=True):
        print(f"Process dir {dir_path}")
        img_paths_ = glob.glob(osp.join(dir_path, '*.jpg'))  # TODO: Concat
        img_paths = glob.glob(osp.join(dir_path, '*.bmp'))

        pattern = re.compile(r'([-\d]+)_MOT20-0(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid  ,f"Error:{img_path}" # pid == 0 means background
            assert 1 <= camid <= 10000,f"Error:{img_path}"
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        print(f"Total find {len(data)} datas in dir {dir_path}")
        return data
