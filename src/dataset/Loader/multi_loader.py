import logging
from typing import List

import numpy as np
import torch

LOG = logging.getLogger(__name__)



class MultiLoader:
     
    last_task_index = None
    

    def __init__(self, loaders: List[torch.utils.data.DataLoader], 
                 weights=None,  n_batches=None):
                 
        self.loaders = loaders
        self._weights = weights

        if self._weights is None:
            self._weights = [1.0 / len(loaders) for _ in range(len(loaders))]
        elif len(self._weights) == len(loaders) - 1:
            self._weights.append(1.0 - sum(self._weights))
        elif len(self._weights) == len(loaders):
            pass
        else:
            raise Exception('invalid dataset weights: {}'.format(self._weights))
        assert all(w > 0.0 for w in self._weights)
        sum_w = sum(self._weights)
        # normalize weights between datasets
        self._weights = [w / sum_w for w in self._weights]
        LOG.info('dataset weights: %s', self._weights)
        # set the total number of batches in one epoch
        self.n_batches = int(min(len(l) / w for l, w in zip(loaders, self._weights)))
        if n_batches is not None:
            self.n_batches = min(self.n_batches, n_batches)

    def __iter__(self):
        loader_iters = [iter(l) for l in self.loaders]
        # counter of loaded batches for each dataset
        n_loaded = [0 for _ in self.loaders]
        while True:
            # select loader for one iteration
            loader_index = int(np.argmin([n / w for n, w in zip(n_loaded, self._weights)]))
            next_batch = next(loader_iters[loader_index], None)
            if next_batch is None:
                break
            n_loaded[loader_index] += 1
            MultiLoader.last_task_index = loader_index
            # generator
            yield next_batch
            
            if sum(n_loaded) >= self.n_batches:
                break

    def __len__(self):
        return self.n_batches