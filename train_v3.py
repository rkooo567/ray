"""
1F1B Pipeline Parallel with static tensor shapes.
"""
import argparse
import ray
import ray.cluster_utils
from ray.experimental.channel.torch_tensor_type import TorchTensorType
from typing import Optional
from ray.dag.compiled_dag_node import CompiledDAG
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import time

BATCH_SIZE = 4
# FEATURE_SIZE = 8192
FEATURE_SIZE = 24576
FORWARD_SHAPE = (BATCH_SIZE, FEATURE_SIZE)
BACKWARD_SHAPE = (BATCH_SIZE, FEATURE_SIZE)

def cifar_trainset(dl_path="/tmp/cifar10-data"):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda x: x.view(-1)),  # Flatten the tensor
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=dl_path, train=True, download=True, transform=transform
    )
    return trainset


@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, n_features, pp_rank, pp_size, micro_batch_size, is_output):
        self.pp_rank = pp_rank
        self.rank = pp_rank
        self.pp_size = pp_size

        self.trace = []
        layers = []
        for i in range(1, len(n_features)):
            in_features, out_features = n_features[i - 1], n_features[i]
            layers.append(nn.Linear(in_features, out_features))

            if not is_output or i < len(n_features) - 1:
                layers.append(nn.ReLU(inplace=True))

        self.module = nn.Sequential(*layers).cuda()
        self.optimizer = optim.SGD(self.module.parameters(), lr=1e-3)
        self.loss = nn.CrossEntropyLoss()

        # if pp_rank == 0:
        self.initialize_dataloader(micro_batch_size=micro_batch_size)

        self.input_activations = dict()
        self.output_activations = dict()

        self.max_memory_allocated = 0
        self.max_memory_reserved = 0

        self.fwd_batch_id = 0
        self.bwd_batch_id = 0

    def initialize_dataloader(self, micro_batch_size):
        trainset = cifar_trainset()
        sampler = torch.utils.data.distributed.DistributedSampler(
            trainset, num_replicas=1, rank=0, shuffle=False
        )
        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=micro_batch_size, shuffle=False, sampler=sampler, pin_memory=True
        )
        self.traindata_iter = iter(self.trainloader)

        # Ste
        input_activation, targets = next(self.traindata_iter)
        self.cache_input_activation = input_activation.cuda()
        self.cache_targets = targets.cuda()

    def fwd(self, fwd_inputs):
        # self.trace.append(("FWD", self.pp_rank))

        input_activation = fwd_inputs
        targets = self.cache_targets
        batch_id = self.fwd_batch_id
        self.fwd_batch_id += 1

        if self.pp_rank > 0:
            input_activation.requires_grad = True
            input_activation.retain_grad()

        # Fetch input batches from dataloader
        if self.pp_rank == 0:
            input_activation = self.cache_input_activation

        # Forward Pass
        self.input_activations[batch_id] = input_activation
        output_activation = self.module(input_activation)
        
        if self.pp_rank == self.pp_size - 1:
            loss = self.loss(input_activation, targets)
            self.output_activations[batch_id] = loss
            return None
        else:
            self.output_activations[batch_id] = output_activation
            return output_activation

    def bwd(self, bwd_inputs):
        # self.trace.append(("BWD", self.pp_rank))

        gradients = bwd_inputs
        batch_id = self.bwd_batch_id
        self.bwd_batch_id += 1

        # Backward Pass
        self.output_activations[batch_id].backward(gradients)
        bwd_gradients = self.input_activations[batch_id].grad

        # Clear cache to free GRAM
        self.input_activations.pop(batch_id)
        self.output_activations.pop(batch_id)

        # Return None to avoid Actor-Driver Comm
        if self.pp_rank == 0:
            return None
        else:
            return bwd_gradients

    def pop_trace(self):
        trace = self.trace
        self.trace = []
        return trace

    def read_input(self, inp):
        # Placeholder: (batch_id, activations, targets)
        self.batch_id = 0
        return (None, None, None)

    def get_memory_logs(self):
        return [self.max_memory_allocated, self.max_memory_reserved]
    
    def get_logs(self):
        return self.logs


def generate_feature_dim(pp_size):
    input_size = 3 * 32 * 32
    feature_size = FEATURE_SIZE
    feature_dim = []

    feature_dim.append([input_size, feature_size, feature_size, feature_size])

    for _ in range(pp_size - 2):
        feature_dim.append([feature_size, feature_size, feature_size])

    feature_dim.append([feature_size, feature_size, feature_size, 10])
    return feature_dim


def generate_1f1b_dag(
    num_workers: int, num_microbatches: int, micro_batch_size: int, overlap = False
) -> CompiledDAG:
    pp_size = num_workers
    num_lead_microbatches = num_workers
    feature_dim_list = generate_feature_dim(num_workers)

    workers = [
        Worker.remote(
            n_features, pp_rank, pp_size, micro_batch_size, bool(pp_rank == pp_size - 1)
        )
        for pp_rank, n_features in enumerate(feature_dim_list)
    ]

    with ray.dag.InputNode() as inp:
        fwd_queues = [[] for _ in range(num_workers)]
        bwd_queues = [[] for _ in range(num_workers)]
        # Once a worker's counter reaches 0, it cannot execute another fwd until it
        # executes a bwd first.
        fwd_counter = [num_lead_microbatches - i for i in range(num_workers)]
        # All of the done batches.
        done = []

        # FWD on worker 0.
        input_data = workers[0].read_input.bind(inp)
        for i in range(num_microbatches):
            fwd_queues[0].append(input_data)

        while len(done) < num_microbatches:
            for i, worker in enumerate(workers):
                if fwd_counter[i] > 0 and fwd_queues[i]:
                    b = fwd_queues[i].pop(0)
                    b = worker.fwd.bind(b)
                    if i < num_workers - 1:
                        fwd_queues[i + 1].append(b)
                        # Use NCCL channel for communication between workers.
                        # b.with_type_hint(
                        #     TorchTensorType(transport=TorchTensorType.NCCL)
                        # )
                        b.with_type_hint(
                            TorchTensorType(transport=TorchTensorType.NCCL, _shape=FORWARD_SHAPE, _dtype=torch.float32, _direct_return=True)
                        )
                    else:
                        bwd_queues[i].append(b)
                    fwd_counter[i] -= 1
                elif bwd_queues[i]:
                    b = bwd_queues[i].pop(0)
                    b = worker.bwd.bind(b)
                    if i > 0:
                        bwd_queues[i - 1].append(b)
                        # Use NCCL channel for communication between workers.
                        # b.with_type_hint(
                        #     TorchTensorType(transport=TorchTensorType.NCCL)
                        # )
                        b.with_type_hint(
                            TorchTensorType(transport=TorchTensorType.NCCL, _shape=BACKWARD_SHAPE, _dtype=torch.float32, _direct_return=True)
                        )
                    else:
                        done.append(b)
                    fwd_counter[i] += 1
        dag = ray.dag.MultiOutputNode(done)
    compiled_dag = dag.experimental_compile(_overlap_gpu_communication=overlap, _execution_timeout=1000)
    return compiled_dag, workers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlap", action="store_true")
    args = parser.parse_args()
    
    # RAY_ADAG_ENABLE_DETECT_DEADLOCK=0 python train_v2.py
    dag, workers = generate_1f1b_dag(num_workers=2, num_microbatches=2, micro_batch_size=BATCH_SIZE, overlap=args.overlap)

    s = time.time()
    for i in range(100):
        print(f"Step {i}:", ray.get(dag.execute(1), timeout=1000))
    e = time.time()

    print(f"Total Training Time: {e - s}s")
    # dag.teardown()

    # records = []
    # for rank, worker in enumerate(workers):
    #     logs = ray.get(worker.get_logs.remote())
    #     records.extend(logs)
    
    # import pandas as pd
    # df = pd.DataFrame.from_records(records)
    # df["op"] = df['task'] + "-" + df['bind_index'].astype(str) + "-" + df['operation']
    # df.to_csv("dag_pp.csv")

    # memory_logs = {}
    # for rank, worker in enumerate(workers):
    #     mem_log = ray.get(worker.get_memory_logs.remote())
    #     memory_logs[rank] = mem_log

    # for rank, mem_log in memory_logs.items():
    #     print("\nOverall CUDA Memory Statistics:")
    #     print(
    #         f"RANK[{rank}] Peak CUDA memory allocated: {mem_log[0] / (1024**3):.2f} GB"
    #     )
    #     print(
    #         f"RANK[{rank}] Peak CUDA memory reserved: {mem_log[1] / (1024**3):.2f} GB"
    #     )