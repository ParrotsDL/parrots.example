from torch import multiprocessing
from torch.utils.data.dataloader import (DataLoader, _DataLoaderIter,
                                         default_collate)


class _PoolDataLoaderIter(_DataLoaderIter):
    def __init__(self, loader):
        self._initialized = False
        self.workers_exist = len(loader.workers) > 0
        self.loader_workers = loader.workers
        self.loader_worker_memory = loader.worker_memory
        self.loader_index_queues = loader.index_queues
        self.loader_worker_result_queue = loader.worker_result_queue
        self.loader_done_event = loader.done_event
        super(_PoolDataLoaderIter, self).__init__(loader)
        self._initialized = True

    def start_worker(self, i):
        if self._initialized:
            super(_PoolDataLoaderIter, self).start_worker(i)
        else:
            if not self.workers:
                self.workers = self.loader_workers
                self.worker_memory = self.loader_worker_memory
                self.index_queues = self.loader_index_queues
                self.worker_result_queue = self.loader_worker_result_queue
                self.done_event = self.loader_done_event
            if not self.workers_exist:
                super(_PoolDataLoaderIter, self).start_worker(i)

    def _shutdown_workers(self):
        pass


class PoolDataLoader(DataLoader):
    __initialized = False

    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 batch_sampler=None,
                 num_workers=0,
                 collate_fn=default_collate,
                 pin_memory=False,
                 drop_last=False,
                 timeout=0,
                 worker_init_fn=None,
                 mode='shm',
                 prefetch_num=8,
                 allow_fail_times=4):
        assert not pin_memory, ("PoolDataLoader do not support"
                                + "pin_memory=True now")
        super(PoolDataLoader,
              self).__init__(dataset, batch_size, shuffle, sampler,
                             batch_sampler, num_workers, collate_fn,
                             pin_memory, drop_last, timeout,
                             worker_init_fn, mode, prefetch_num,
                             allow_fail_times)
        if num_workers > 0:
            self.workers = []
            self.worker_memory = []
            self.index_queues = []
            self.worker_result_queue = multiprocessing.Queue(mode=mode)
            self.done_event = multiprocessing.Event()

    def __iter__(self):
        if self.num_workers > 0:
            return _PoolDataLoaderIter(self)
        return super(PoolDataLoader, self).__iter__()

    def __del__(self):
        if getattr(self, "num_workers", 0) > 0:
            self._shutdown_workers()

    def _shutdown_workers(self):
        # removes pids from the C side data structure first so worker
        # termination afterwards won't trigger false positive error report.
        self.done_event.set()
        # Workers can't be waiting to put be cause their output queue
        # is a multiprocessing.Queue and its .put is non-blocking.
        # They can only be waiting to get, so we put `None` here.
        for q in self.index_queues:
            q.put(None)
        for w in self.workers:
            w.join()
