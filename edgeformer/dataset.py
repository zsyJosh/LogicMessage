import os
from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.core import Registry as R

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

        txt_files = []
        for url in self.urls:
            save_file = "kinship_%s" % os.path.basename(url)
            txt_file = os.path.join(path, save_file)
            assert os.path.exists(txt_file)
            print("%s exists" % (txt_file))
            txt_files.append(txt_file)

        self.load_tsvs(txt_files, verbose=verbose)

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits