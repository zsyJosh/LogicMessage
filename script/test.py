import os
import sys
from torch.utils import data as torch_data
from tqdm import tqdm
import util
import torch

from torchdrug import core, data, utils
from torchdrug.core import Registry as R

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

@R.register("datasets.kinship")
class kinship(data.KnowledgeGraphDataset):
    """
    Subset of Freebase knowledge base for knowledge graph reasoning.

    Statistics:
        - #Entity:
        - #Relation:
        - #Triplet:

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
    """

    urls = [
        "https://github.com/Colinasda/KGdatasets/tree/main/Kinship/train.txt",
        "https://github.com/Colinasda/KGdatasets/tree/main/Kinship/valid.txt",
        "https://github.com/Colinasda/KGdatasets/tree/main/Kinship/test.txt",
    ]

    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        txt_files = ['/Users/zhaoshiyu/Desktop/transfer/kinship_train.txt',
                     '/Users/zhaoshiyu/Desktop/transfer/kinship_valid.txt',
                     '/Users/zhaoshiyu/Desktop/transfer/kinship_test.txt']

        self.load_tsvs(txt_files, verbose=verbose)

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)
    dataset = core.Configurable.load_config_dict(cfg.dataset)
    print('dataset', dataset)
    train, valid, test = dataset.split()

    h_list = []
    t_list = []
    r_list = []
    for i in range(len(train)):
        h_list.append(train[i][0])
        t_list.append(train[i][1])
        r_list.append(train[i][2])
    h_list = torch.tensor(h_list)
    t_list = torch.tensor(t_list)
    r_list = torch.tensor(r_list)


