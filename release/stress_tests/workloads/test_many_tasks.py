#!/usr/bin/env python

from collections import defaultdict
import numpy as np
import logging
import time

import ray
from ray.cluster_utils import Cluster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# These numbers need to correspond with the autoscaler config file.
# The number of remote nodes in the autoscaler should upper bound
# these because sometimes nodes fail to update.
num_remote_nodes = 5
head_node_cpus = 4
num_remote_cpus = num_remote_nodes * head_node_cpus

cluster = Cluster()
for _ in range(num_remote_nodes + 1):
    cluster.add_node(num_cpus=head_node_cpus, num_gpus=1)
ray.init(address=cluster.address)

# Wait until the expected number of nodes have joined the cluster.
while True:
    num_nodes = len(ray.nodes())
    logger.info("Waiting for nodes {}/{}".format(num_nodes,
                                                 num_remote_nodes + 1))
    if num_nodes >= num_remote_nodes + 1:
        break
    time.sleep(5)
logger.info("Nodes have all joined. There are %s resources.",
            ray.cluster_resources())


# Require 1 GPU to force the tasks to be on remote machines.
@ray.remote(num_gpus=1)
def f(size, *xs):
    return np.ones(size, dtype=np.uint8)


# Require 1 GPU to force the actors to be on remote machines.
@ray.remote(num_cpus=1)
class Actor(object):
    def method(self, size, *xs):
        return np.ones(size, dtype=np.uint8)


# Stage 0: Submit a bunch of small tasks with large returns.
stage_0_iterations = []
start_time = time.time()
logger.info("Submitting many tasks with large returns.")
for i in range(10):
    iteration_start = time.time()
    logger.info("Iteration %s", i)
    ray.get([f.remote(1000000) for _ in range(100)])
    stage_0_iterations.append(time.time() - iteration_start)

stage_0_time = time.time() - start_time
logger.info("Finished stage 0 after %s seconds.", stage_0_time)

# Stage 1: Launch a bunch of tasks.
stage_1_iterations = []
start_time = time.time()
logger.info("Submitting many tasks.")
for i in range(10):
    iteration_start = time.time()
    logger.info("Iteration %s", i)
    ray.get([f.remote(0) for _ in range(1000)])
    stage_1_iterations.append(time.time() - iteration_start)

stage_1_time = time.time() - start_time
logger.info("Finished stage 1 after %s seconds.", stage_1_time)

# Launch a bunch of tasks, each with a bunch of dependencies. TODO(rkn): This
# test starts to fail if we increase the number of tasks in the inner loop from
# 500 to 1000. (approximately 615 seconds)
stage_2_iterations = []
start_time = time.time()
logger.info("Submitting tasks with many dependencies.")
x_ids = []
for _ in range(5):
    iteration_start = time.time()
    for i in range(20):
        logger.info("Iteration %s. Cumulative time %s seconds", i,
                    time.time() - start_time)
        x_ids = [f.remote(0, *x_ids) for _ in range(50)]
    ray.get(x_ids)
    stage_2_iterations.append(time.time() - iteration_start)
    logger.info("Finished after %s seconds.", time.time() - start_time)

stage_2_time = time.time() - start_time
logger.info("Finished stage 2 after %s seconds.", stage_2_time)

# Create a bunch of actors.
start_time = time.time()
logger.info("Creating %s actors.", num_remote_cpus)
actors = [Actor.remote() for _ in range(num_remote_cpus)]
stage_3_creation_time = time.time() - start_time
logger.info("Finished stage 3 actor creation in %s seconds.",
            stage_3_creation_time)

# Submit a bunch of small tasks to each actor. (approximately 1070 seconds)
start_time = time.time()
logger.info("Submitting many small actor tasks.")
for N in [1000, 1100]:
    x_ids = []
    for i in range(N):
        x_ids = [a.method.remote(0) for a in actors]
        if i % 100 == 0:
            logger.info("Submitted {}".format(i * len(actors)))
    ray.get(x_ids)
stage_3_time = time.time() - start_time
logger.info("Finished stage 3 in %s seconds.", stage_3_time)

del actors

# This tests https://github.com/ray-project/ray/issues/10150. The only way to
# integration test this is via performance. The goal is to fill up the cluster
# so that all tasks can be run, but spillback is required. Since the driver
# submits all these tasks it should easily be able to schedule each task in
# O(1) iterative spillback queries. If spillback behavior is incorrect, each
# task will require O(N) queries. Since we limit the number of inflight
# requests, we will run into head of line blocking and we should be able to
# measure this timing.
num_tasks = int(ray.cluster_resources()["GPU"])
logger.info(f"Scheduling many tasks for spillback.")


@ray.remote(num_gpus=1)
def func(t):
    if t % 100 == 0:
        logger.info(f"[spillback test] {t}/{num_tasks}")
    start = time.perf_counter()
    time.sleep(1)
    end = time.perf_counter()
    return start, end, ray.worker.global_worker.node.unique_id


results = ray.get([func.remote(i) for i in range(num_tasks)])

host_to_start_times = defaultdict(list)
for start, end, host in results:
    host_to_start_times[host].append(start)

spreads = []
for host in host_to_start_times:
    last = max(host_to_start_times[host])
    first = min(host_to_start_times[host])
    spread = last - first
    spreads.append(spread)
    logger.info(f"Spread: {last - first}\tLast: {last}\tFirst: {first}")

# avg_spread ~ 115 with Ray 1.0 scheduler. ~695 with (buggy) 0.8.7 scheduler.
avg_spread = sum(spreads) / len(spreads)
logger.info(f"Avg spread: {sum(spreads)/len(spreads)}")

print("Stage 0 results:")
print("\tTotal time: {}".format(stage_0_time))

print("Stage 1 results:")
print("\tTotal time: {}".format(stage_1_time))
print("\tAverage iteration time: {}".format(
    sum(stage_1_iterations) / len(stage_1_iterations)))
print("\tMax iteration time: {}".format(max(stage_1_iterations)))
print("\tMin iteration time: {}".format(min(stage_1_iterations)))

print("Stage 2 results:")
print("\tTotal time: {}".format(stage_2_time))
print("\tAverage iteration time: {}".format(
    sum(stage_2_iterations) / len(stage_2_iterations)))
print("\tMax iteration time: {}".format(max(stage_2_iterations)))
print("\tMin iteration time: {}".format(min(stage_2_iterations)))

print("Stage 3 results:")
print("\tActor creation time: {}".format(stage_3_creation_time))
print("\tTotal time: {}".format(stage_3_time))

print("Stage 4 results:")
print(f"\tScheduling spread: {avg_spread}.")

# TODO(rkn): The test below is commented out because it currently does not
# pass.
# # Submit a bunch of actor tasks with all-to-all communication.
# start_time = time.time()
# logger.info("Submitting actor tasks with all-to-all communication.")
# x_ids = []
# for _ in range(50):
#     for size_exponent in [0, 1, 2, 3, 4, 5, 6]:
#         x_ids = [a.method.remote(10**size_exponent, *x_ids) for a in actors]
# ray.get(x_ids)
# logger.info("Finished after %s seconds.", time.time() - start_time)
