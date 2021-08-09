import ray
import pytest
import sys

from ray.experimental import shuffle


def test_shuffle():
    try:
        shuffle.main()
    finally:
        ray.shutdown()


# https://github.com/ray-project/ray/pull/16408
@pytest.mark.skipif(sys.platform == "win32", reason="Failing on Windows.")
def test_shuffle_hang():
    try:
        shuffle.main(
            object_store_memory=1e9, num_partitions=200, partition_size=10e6)
    finally:
        ray.shutdown()


def test_shuffle_no_streaming():
    try:
        shuffle.main(no_streaming=True)
    finally:
        ray.shutdown()


def test_shuffle_multi_node(ray_start_cluster):
    cluster = ray_start_cluster
    for _ in range(4):
        cluster.add_node(num_cpus=2, object_store_memory=1e9)

    shuffle.main(ray_address="auto", num_partitions=200, partition_size=10e6)


def test_shuffle_multi_node_no_streaming(ray_start_cluster):
    cluster = ray_start_cluster
    for _ in range(4):
        cluster.add_node(num_cpus=2, object_store_memory=1e9)

    shuffle.main(
        ray_address="auto",
        num_partitions=200,
        partition_size=10e6,
        no_streaming=True)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
