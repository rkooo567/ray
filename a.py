import ray
from ray.util.metrics import Histogram
import time

ray.init()

DEFAULT_LATENCY_BUCKET_MS = [
    i for i in range(1, 1000)
]

@ray.remote
class Actor:
    def __init__(self):
        self.hists = [
            Histogram(
                    f"{i}",
                    description=("my histogram "),
                    boundaries=DEFAULT_LATENCY_BUCKET_MS,
                    tag_keys=())
            for i in range(100)
        ]

    def record(self):
        for hist in self.hists:
            for i in DEFAULT_LATENCY_BUCKET_MS:
                hist.observe(i)

while True:
    actors = [Actor.remote() for _ in range(40)]
    ray.get([actor.record.remote() for actor in actors])
    time.sleep(5)
    del actors
