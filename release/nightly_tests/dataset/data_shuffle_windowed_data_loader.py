import time
import ray
import argparse
import os
from ray.data.dataset_pipeline import DatasetPipeline

DUMMY_ROW = {
    "key": 0,
    "embeddings_name0": 1527,
    "embeddings_name1": 175,
    "embeddings_name2": 8,
    "embeddings_name3": 5,
    "embeddings_name4": 5,
    "embeddings_name5": 687,
    "embeddings_name6": 165,
    "embeddings_name7": 10,
    "embeddings_name8": 137,
    "embeddings_name9": 597,
    "embeddings_name10": 1574,
    "embeddings_name11": 78522,
    "embeddings_name12": 283941,
    "embeddings_name13": 3171,
    "embeddings_name14": 39560,
    "embeddings_name15": 718571,
    "embeddings_name16": 73699,
    "one_hot0": 1,
    "one_hot1": 30,
    "labels": 0.3801173847541949,
    "__index_level_0__": 0
}


def create_parser():
    parser = argparse.ArgumentParser(description="Eric Example")
    parser.add_argument("--address")
    parser.add_argument("--num-rows", type=int, default=1 * (10**9))
    parser.add_argument("--num-files", type=int, default=200)
    parser.add_argument("--read-cache", action="store_true", default=False)
    parser.add_argument(
        "--data-dir",
        type=str,
        default="s3://shuffling-data-loader-benchmarks/data/")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=250000,
        metavar="N",
        help="input batch size for training (default: 64)")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


# def run_shuffle(args, data_dir):
#     ds = ray.data.read_parquet(data_dir)
#     res = ds.random_shuffle().take(10)
#     print(res)
def create_shuffled_dataset(args, data_dir):
    if data_dir:
        ds = ray.data.read_parquet(data_dir)
    else:
        ds = ray.data.range(args.num_rows).map(lambda i: DUMMY_ROW)
    pipeline = ds.repeat().random_shuffle()
    splits = pipeline.split(args.num_workers, equal=True)
    return splits


def create_windowed_dataset(args, data_dir, windows=10, epochs=2):
    if data_dir:
        ds = ray.data.read_parquet(data_dir)
    else:
        ds = ray.data.range(args.num_rows).map(lambda i: DUMMY_ROW)

    splits = ds.split(windows)

    def loop_over(datasets):
        return [(lambda ds=ds: ds) for i in range(epochs) for ds in datasets]

    pipe = DatasetPipeline.from_iterable(loop_over(splits))
    return pipe.split(args.num_workers, equal=True)


def run_consume(args, data_dir=None):
    splits = create_windowed_dataset(args, data_dir)

    @ray.remote(num_gpus=1)
    def consume(split, rank=None, batch_size=None):
        for i, x in enumerate(split.iter_rows()):
            time.sleep(1)
            if i % 10 == 0:
                print(i)
        return

    tasks = [
        consume.remote(split, rank=idx, batch_size=args.batch_size)
        for idx, split in enumerate(splits)
    ]
    ray.get(tasks)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print("Connecting to Ray cluster...")
    ray.init(address=args.address)

    # data_dir = None
    data_dir = os.path.join(args.data_dir,
                            f"r{args.num_rows:_}-f{args.num_files}/")
    run_consume(args, data_dir=data_dir)
    # run_shuffle(args)
