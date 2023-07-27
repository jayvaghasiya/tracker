import multiprocessing as mp
import time


class MonitoredQueue:
    def __init__(self, maxsize=0):
        self.queue = mp.Queue(maxsize=maxsize)
        self.last_put_time = time.time()

    def put(self, item, *args, **kwargs):
        self.queue.put(item, *args, **kwargs)
        self.last_put_time = time.time()

    def get(self, *args, **kwargs):
        return self.queue.get(*args, **kwargs)

    def qsize(self, *args, **kwargs):
        return self.queue.qsize(*args, **kwargs)

    def empty(self):
        return self.queue.empty()
