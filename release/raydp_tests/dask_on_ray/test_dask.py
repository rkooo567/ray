import glob
import ray
import dask
import dask.dataframe as dd
import json
import pandas as pd
import numpy as np
from ray.util.dask import ray_dask_get_sync
import os.path
import csv
import fastparquet
import sys

from dask.distributed import Client
from dask.distributed import wait
import cProfile, pstats, io
from pstats import SortKey
from tqdm import tqdm

import time

DATA_DIR = "~/dask-on-ray-data"
# DATA_DIR = "/obj-data"


def load_dataset(nbytes, npartitions, sort):
    num_bytes_per_partition = nbytes // npartitions
    filenames = []

    @ray.remote
    def foo(i):
        filename = "df-{}-{}-{}.parquet.gzip".format(
            "sort" if sort else "groupby", num_bytes_per_partition, i)
        filename = os.path.join(DATA_DIR, filename)
        # print("Partition file", filename)
        if not os.path.exists(filename):
            if sort:
                nrows = num_bytes_per_partition // 8
                # print("Allocating dataset with {} rows".format(nrows))
                dataset = pd.DataFrame(
                    np.random.randint(
                        0,
                        np.iinfo(np.int64).max,
                        size=(nrows, 1),
                        dtype=np.int64),
                    columns=['a'])
            else:
                nrows = num_bytes_per_partition // (8 * 2)
                # print("Allocating dataset with {} rows".format(nrows))
                dataset = pd.DataFrame(
                    np.random.randint(0, 100, size=(nrows, 2), dtype=np.int64),
                    columns=['a', 'b'])
            # print("Done allocating")
            dataset.to_parquet(filename, compression='gzip')
            # print("Done writing to disk")
        return filename

    for i in range(npartitions):
        filenames.append(foo.remote(i))
    results = []
    pbar = tqdm(total=len(filenames))
    ready, unready = ray.wait(filenames)
    while unready:
        results.extend(ready)
        ready, unready = ray.wait(unready)
        pbar.update(len(ready))
    del filenames
    filenames = ray.get(results)

    df = dd.read_parquet(filenames)
    return df


def trial(nbytes, n_partitions, sort, generate_only):
    from ray.util.dask import RayDaskCallback

    @ray.remote
    class CounterActor:
        def __init__(self):
            self.cnt = 0
            self.object_num = 0
        def increment(self, n, object_num):
            self.cnt += n
            self.object_num += object_num
        def get(self):
            return self.cnt
        def get_object_num(self):
            return self.object_num

    c = CounterActor.remote()

    class CountRefs(RayDaskCallback):
        def __init__(self, cnt_actor):
            self.cnt_actor = cnt_actor

        def _ray_pretask(self, key, consumed_objs):
            total = 0
            object_num = 0
            for obj in consumed_objs:
                if sys.getsizeof(obj) > 1024 * 100:
                    print(type(obj))
                    object_num += 1
                total += sys.getsizeof(obj)
            self.cnt_actor.increment.remote(total, object_num)

        def _ray_postsubmit_all(self, object_refs, dsk):
            pass
            # print(f"dask DAG: {dsk.dicts}")

        def _ray_finish(self, result):
            if isinstance(result, (list, tuple)):
                result = [result]
            total = 0
            object_num = 0
            for obj in result:
                if sys.getsizeof(obj) > 1024 * 100:
                    object_num += 1
                total += sys.getsizeof(obj)
            self.cnt_actor.increment.remote(total, object_num)


    count = CountRefs(c)
    df = load_dataset(nbytes, n_partitions, sort)
    # pr = cProfile.Profile()
    if generate_only:
        return

    times = []
    # df.visualize(filename=f'a-{i}.svg')
    start = time.time()
    for i in range(1):
        print("Trial {} start".format(i))
        trial_start = time.time()

        if sort:
            # pr.enable()
            with count:
                a = df.set_index('a', shuffle='tasks', max_branch=10 ** 9)
                a = a.head(10, npartitions=-1, compute=False)
                # a.visualize(filename=f'a-{i}.svg')
                a.compute()
            # pr.disable()
        else:
            df.groupby('b').a.mean().compute()

        trial_end = time.time()
        duration = trial_end - trial_start
        times.append(duration)
        print("Trial {} done after {}".format(i, duration))
        print(f"num obj ref bytes used: {ray.get(c.get.remote()) / 1024 / 1024}")
        print(f"num objs: {ray.get(c.get_object_num.remote())}")
        # sortby = SortKey.CUMULATIVE
        # ps = pstats.Stats(pr).sort_stats(sortby)
        # ps.dump_stats("ray_profile_data")

        if time.time() - start > 60 and i > 0:
            break
    return times


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--nbytes", type=int, default=1_000_000)
    parser.add_argument("--npartitions", type=int, default=100, required=False)
    # Max partition size is 1GB.
    parser.add_argument(
        "--max-partition-size", type=int, default=1000_000_000, required=False)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--sort", action="store_true")
    parser.add_argument("--timeline", action="store_true")
    parser.add_argument("--dask", action="store_true")
    parser.add_argument("--ray", action="store_true")
    parser.add_argument("--dask-tasks", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--clear-old-data", action="store_true")
    args = parser.parse_args()

    if args.clear_old_data:
        print(f"Clearing old data from {DATA_DIR}.")
        files = glob.glob(os.path.join(DATA_DIR, "*.parquet.gzip"))
        for f in files:
            os.remove(f)

    if args.ray:
        args.dask_tasks = True

    if args.dask_tasks:
        print("Using task-based Dask shuffle")
        dask.config.set(shuffle='tasks')
    else:
        print("Using disk-based Dask shuffle")

    if args.dask:
        client = Client('127.0.0.1:8786')
        ray.init(address='auto')
    if args.ray:
        ray.init(address="auto")
        # ray.init(
        #     num_cpus=16,
        #     _system_config={
        #         "max_io_workers": 1,
        #         "object_spilling_config": json.dumps(
        #             {
        #                 "type": "filesystem",
        #                 "params": {
        #                     "directory_path": "/tmp/spill"
        #                 }
        #             },
        #             separators=(",", ":"))
        #     })
        dask.config.set(scheduler=ray_dask_get_sync)

    system = "dask" if args.dask else "ray"

    # print(system, trial(1000, 10, args.sort, args.generate_only))
    # print("WARMUP DONE")

    npartitions = args.npartitions
    if args.nbytes // npartitions > args.max_partition_size:
        npartitions = args.nbytes // args.max_partition_size

    output = trial(args.nbytes, npartitions, args.sort, args.generate_only)
    print("{} mean over {} trials: {} +- {}".format(system, len(output),
                                                    np.mean(output),
                                                    np.std(output)))

    write_header = not os.path.exists("output.csv") or os.path.getsize(
        "output.csv") == 0
    with open("output.csv", "a+") as csvfile:
        fieldnames = [
            "system", "operation", "num_nodes", "nbytes", "npartitions",
            "duration"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row = {
            "operation": "sort" if args.sort else "groupby",
            "num_nodes": args.num_nodes,
            "nbytes": args.nbytes,
            "npartitions": npartitions,
        }
        for output in output:
            row["system"] = system
            row["duration"] = output
            writer.writerow(row)

    if args.timeline:
        time.sleep(1)
        ray.timeline(filename="dask.json")