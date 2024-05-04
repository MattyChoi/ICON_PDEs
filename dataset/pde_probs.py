import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, default_collate
from tfrecord.torch.dataset import TFRecordDataset
from tfrecord import reader
from tfrecord import iterator_utils
import typing
    


def create_prompt(examples):
    grid, conds, qois = examples['grid'], examples['conditions'], examples['qois']
    grid, conds, qois = torch.Tensor(grid), torch.Tensor(conds), torch.Tensor(qois)
    
    # get the number of examples and grid_size
    grid_size = grid.size(0)
    num_examples = conds.size(0) // grid_size 
    
    # reshape the conditions and qois back to their 2d array forms
    conds = conds.reshape(num_examples, grid_size)
    qois = qois.reshape(num_examples, grid_size)

    # create the label
    label = torch.hstack([conds, qois])
    
    # shuffle the label so that we are predicting different qois each time
    shuffle = torch.randperm(num_examples)
    label = label[shuffle]
    
    prompt = label.clone().detach()
    assert grid_size == prompt.size(1) // 2
    prompt[-1, grid_size:] = 0
    
    return prompt, label


# create a pytorch dataset of pde problems
class PDEProblems(Dataset):
    """
    Dataset of pde problems
    """
    def __init__(
        self, 
        data_dir="./data/finite_diff_gsize_101.tfrecord", 
        transform=None,
    ):
        super(PDEProblems, self).__init__()

        # load the iterable dataset from the tfrecord file
        print("loading tfrecord dataset")
        self.data = list(TFRecordDataset(data_dir, index_path=None))
        print("tfrecord dataset loaded")

    def __getitem__(self, index):
        # create a single prompt, using a set of examples from one operator from our dataset
        examples = self.data[index]
        return create_prompt(examples)

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        prompts, labels = list(zip(*batch))

        prompts = torch.stack(prompts, dim=0)
        labels = torch.stack(labels, dim=0)

        return prompts, labels 


# create an iterable pytorch dataset of pde problems
class PDEProblemsIter(IterableDataset):
    """Parse (generic) TFRecords dataset into `IterableDataset` object,
    which contain `np.ndarrays`s. By default (when `sequence_description`
    is None), it treats the TFRecords as containing `tf.Example`.
    Otherwise, it assumes it is a `tf.SequenceExample`.

    Params:
    -------
    data_path: str
        The path to the tfrecords file.

    index_path: str or None
        The path to the index file.

    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.

    shuffle_queue_size: int, optional, default=None
        Length of buffer. Determines how many records are queued to
        sample from.

    transform : a callable, default = None
        A function that takes in the input `features` i.e the dict
        provided in the description, transforms it and returns a
        desirable output.

    sequence_description: list or dict of str, optional, default=None
        Similar to `description`, but refers to the sequence features
        within a `SequenceExample`. When this field is `None`, then it
        is assumed that an `Example` is being read otherwise, a
        `SequenceExample` is read. If an empty list or dictionary is
        passed, then all features contained in the file are extracted.

    compression_type: str, optional, default=None
        The type of compression used for the tfrecord. Choose either
        'gzip' or None.

    """

    def __init__(self,
        data_dir: str,
        index_path: typing.Union[str, None],
        description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
        shuffle_queue_size: typing.Optional[int] = None,
        transform: typing.Callable[[dict], typing.Any] = None,
        sequence_description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
        compression_type: typing.Optional[str] = None,
    ) -> None:
        super(PDEProblemsIter, self).__init__()
        self.data_path = data_dir
        self.index_path = index_path
        self.description = description
        self.sequence_description = sequence_description
        self.shuffle_queue_size = shuffle_queue_size
        self.transform = transform
        self.compression_type = compression_type

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shard = worker_info.id, worker_info.num_workers
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        else:
            shard = None
        it = reader.tfrecord_loader(data_path=self.data_path,
                                    index_path=self.index_path,
                                    description=self.description,
                                    shard=shard,
                                    sequence_description=self.sequence_description,
                                    compression_type=self.compression_type)
        if self.shuffle_queue_size:
            it = iterator_utils.shuffle_iterator(it, self.shuffle_queue_size)
        if self.transform:
            it = map(self.transform, it)
        return map(create_prompt, it)

        
    def collate_fn(self, batch):
        prompts, labels = list(zip(*batch))

        prompts = torch.stack(prompts, dim=0)
        labels = torch.stack(labels, dim=0)

        return prompts, labels 




# if __name__ == "__main__":
#     data = PDEProblemsIter()
    
#     import pdb
#     pdb.set_trace()
    