import ray
import ray.cluster_utils
from ray.experimental.channel.torch_tensor_type import TorchTensorType
from ray.dag import InputNode, MultiOutputNode
from typing import Optional
from ray.dag.compiled_dag_node import CompiledDAG
import time
from dataclasses import dataclass, asdict
from collections import defaultdict
from argparse import ArgumentError, ArgumentParser


@ray.remote(num_cpus=0, num_gpus=1)
class DummyWorker:
    def __init__(self, rank: Optional[int] = None):
        self.rank = rank
        self.trace = []

    def fwd(self, value):
        # self.trace.append(("FWD", self.rank))
        self.trace.append("F")
        return value

    def bwd(self, value):
        # self.trace.append(("BWD", self.rank))
        self.trace.append("B")
        return value

    def w(self, value):
        # self.trace.append(("W", self.rank))
        if self.trace[-1] == "W":
            assert False
        self.trace.append("W")
        return None

    def echo(self, value):
        return value

    def pop_trace(self):
        trace = self.trace
        self.trace = []
        return trace

    def read_input(self, input):
        return input

    def no_op(self, value):
        return value

    def no_op_two(self, value1, value2):
        return value1, value2


import torchvision
import torchvision.transforms as transforms
import time
import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_lightning import seed_everything

BATCH_SIZE = 4
# FEATURE_SIZE = 8192
FEATURE_SIZE = 24576

IMAGE_SIZE = 3072
FORWARD_SHAPE = (BATCH_SIZE, FEATURE_SIZE)
BACKWARD_SHAPE = (BATCH_SIZE, FEATURE_SIZE)
INPUT_SIZE = (BATCH_SIZE, IMAGE_SIZE)
LABEL_SIZE = (BATCH_SIZE,)


def cifar_trainset(dl_path="/mnt/local_storage/cifar10-data"):
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


class WrappedLinear(nn.Module):
    def __init__(self, linear) -> None:
        super().__init__()
        self.linear = linear
        self.output_activations = []
        self.output_grads = []
        self.call_counter = 0

    def forward(self, x):
        output = self.linear(x)
        output.register_hook(self.save_gradients)
        self.output_activations.append(output)
        return output

    def w(self):
        output_activation = self.output_activations.pop(0)
        output_grad = self.output_grads.pop(0)
        output_activation.backward(output_grad, inputs=list(self.parameters()))

    def save_gradients(self, grad):
        self.output_grads.append(grad)
        self.call_counter = +1
        return grad


def wrap_linear_layer(linear):
    wrapped_linear = WrappedLinear(linear)
    # wrapped_linear = linear
    # wrapped_linear.register_full_backward_hook(save_gradients_hook)
    return wrapped_linear


@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, n_features, pp_rank, pp_size, micro_batch_size, is_output):
        seed_everything(420)
        torch.manual_seed(420)
        self.pp_rank = pp_rank
        self.rank = pp_rank
        self.pp_size = pp_size

        self.trace = []
        self.timer = defaultdict(list)

        layers = []
        for i in range(1, len(n_features)):
            in_features, out_features = n_features[i - 1], n_features[i]
            linear = nn.Linear(in_features, out_features)
            layers.append(wrap_linear_layer(linear))

            if not is_output or i < len(n_features) - 1:
                layers.append(nn.ReLU(inplace=False))

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
        self.w_batch_id = 0

    def initialize_dataloader(self, micro_batch_size):
        self.cache_input_activation = torch.randn(INPUT_SIZE).cuda()
        self.cache_targets = torch.randint(0, 10, LABEL_SIZE).cuda()

        # trainset = cifar_trainset()
        # sampler = torch.utils.data.distributed.DistributedSampler(
        #     trainset, num_replicas=1, rank=0, shuffle=False
        # )
        # self.trainloader = torch.utils.data.DataLoader(
        #     trainset,
        #     batch_size=micro_batch_size,
        #     shuffle=False,
        #     sampler=sampler,
        #     pin_memory=True,
        # )
        # self.traindata_iter = iter(self.trainloader)

        # # Ste
        # input_activation, targets = next(self.traindata_iter)
        # self.input_shape = input_activation.shape
        # self.target_shape = targets.shape

        # fout = f"/mnt/cluster_storage/dag_check/inputs_{self.pp_rank}.pkl"
        # torch.save(
        #     {
        #         "input": self.cache_input_activation.cpu(),
        #         "target": self.cache_targets.cpu(),
        #     },
        #     fout,
        # )
        # self.cache_input_activation = input_activation.cuda()
        # self.cache_targets = targets.cuda()

    def fwd(self, activations):
        # s = time.perf_counter()
        self.trace.append("F")

        if self.pp_rank == 0:
            # input_activations = torch.randn(*self.input_shape).cuda()
            input_activations = self.cache_input_activation
        else:
            input_activations = activations

        input_activations.requires_grad = True
        input_activations.retain_grad()

        self.input_activations[self.fwd_batch_id] = input_activations

        output_activation = self.module(input_activations)

        if self.pp_rank == self.pp_size - 1:
            loss = output_activation.sum()
            self.output_activations[self.fwd_batch_id] = loss
            self.fwd_batch_id += 1
            return None
        else:
            self.output_activations[self.fwd_batch_id] = output_activation
            self.fwd_batch_id += 1
            return output_activation

    def bwd(self, gradient):
        # s = time.perf_counter()
        self.trace.append("B")

        if self.pp_rank == self.pp_size - 1:
            gradient = None

        output_activation = self.output_activations.pop(self.bwd_batch_id)
        input_activation = self.input_activations.pop(self.bwd_batch_id)

        for param in self.module.parameters():
            param.requires_grad = False
        input_activation.requires_grad = True

        output_activation.backward(
            gradient, retain_graph=True, inputs=[input_activation]
        )

        # for layer in self.module:
        #     if isinstance(layer, WrappedLinear):
        #         assert len(layer.output_activations) > 0
        #         assert len(layer.output_grads) > 0
        #         assert layer.call_counter > 0

        if self.pp_rank == 0:
            self.bwd_batch_id += 1
            return None
        else:
            self.bwd_batch_id += 1
            return input_activation.grad

    def w(self, x):
        # s = time.perf_counter()
        self.trace.append("W")

        try:

            for layer in self.module:
                if isinstance(layer, WrappedLinear):
                    for param in self.module.parameters():
                        param.requires_grad = True
                    
                    layer.w()
                
        except Exception as e:
            print("Error", e)

        return None

    def pop_trace(self):
        trace = self.trace
        self.trace = []
        return trace

    def pop_timer(self):
        return self.timer

    def read_input(self, inp):
        # Placeholder: (batch_id, activations, targets)
        self.batch_id = 0
        return (None, None, None)

    def get_memory_logs(self):
        return [self.max_memory_allocated, self.max_memory_reserved]

    def get_events(self):
        events = getattr(self, "__ray_adag_events", [])
        events_list = [asdict(event) for event in events]
        print(events_list)
        return events_list

    def echo(self, value):
        return value


def generate_feature_dim(pp_size):
    input_size = 3 * 32 * 32
    feature_size = FEATURE_SIZE
    feature_dim = []

    feature_dim.append([input_size, feature_size, feature_size, feature_size])

    for _ in range(pp_size - 2):
        feature_dim.append([feature_size, feature_size, feature_size])

    feature_dim.append([feature_size, feature_size, feature_size, 10])
    return feature_dim


def generate_zbh1_dag(
    num_workers: int,
    num_microbatches: int,
    num_lead_microbatches: int,
    use_dummy: bool,
    zb=True,
    overlap=False
):
    if use_dummy:
        workers = [
            DummyWorker.options(name=f"worker-{rank}").remote(rank)
            for rank in range(num_workers)
        ]
    else:
        pp_size = num_workers
        num_lead_microbatches = num_workers
        feature_dim_list = generate_feature_dim(num_workers)

        workers = [
            Worker.options(name=f"worker-{pp_rank}").remote(
                n_features, pp_rank, pp_size, BATCH_SIZE, bool(pp_rank == pp_size - 1)
            )
            for pp_rank, n_features in enumerate(feature_dim_list)
        ]

    with InputNode() as inp:
        fwd_queues = [[] for _ in range(num_workers)]
        bwd_queues = [[] for _ in range(num_workers)]
        # Once a worker's counter reaches 0, it cannot execute another fwd until it
        # executes a bwd first.
        fwd_counter = [num_lead_microbatches - i for i in range(num_workers)]
        bwd_counter = [0 for i in range(num_workers)]
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
                        b.with_type_hint(
                            TorchTensorType(
                                transport=TorchTensorType.NCCL,
                                _shape=BACKWARD_SHAPE,
                                _dtype=torch.float32,
                                _direct_return=True,
                            )
                        )
                        # b.with_type_hint(
                        #     TorchTensorType(transport=TorchTensorType.NCCL)
                        # )
                    else:
                        bwd_queues[i].append(b)
                    fwd_counter[i] -= 1
                elif bwd_queues[i]:
                    b = bwd_queues[i].pop(0)
                    b2 = worker.bwd.bind(b)

                    # Code change for Zero Bubble PP
                    # ++++++++++++++++++++++++++++++++++++++++++++++++
                    bwd_counter[i] += 1

                    if bwd_counter[i] > i:
                        echo_b2 = worker.echo.bind(b2)
                        w2 = worker.w.bind(b2)

                        if bwd_counter[i] == num_microbatches:
                            for _ in range(i):
                                w2 = worker.w.bind(w2)
                    else:
                        echo_b2 = None

                    if echo_b2:
                        b2 = echo_b2
                    # ++++++++++++++++++++++++++++++++++++++++++++++++

                    if i > 0:
                        bwd_queues[i - 1].append(b2)
                        # Use NCCL channel for communication between workers.
                        b2.with_type_hint(
                            TorchTensorType(
                                transport=TorchTensorType.NCCL,
                                _shape=BACKWARD_SHAPE,
                                _dtype=torch.float32,
                                _direct_return=True,
                            )
                        )
                        # b2.with_type_hint(
                        #     TorchTensorType(transport=TorchTensorType.NCCL)
                        # )
                    else:
                        done.append(b2)

                    fwd_counter[i] += 1

        dag = MultiOutputNode(done)
    compiled_dag = dag.experimental_compile(_overlap_gpu_communication=overlap)
    return compiled_dag, workers


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--use_dummy", action="store_true")
    parser.add_argument("--zb", action="store_true")
    parser.add_argument("--overlap", action="store_true")
    args = parser.parse_args()

    dag, workers = generate_zbh1_dag(
        num_workers=4,
        num_lead_microbatches=4,
        num_microbatches=8,
        use_dummy=args.use_dummy,
        overlap=args.overlap
    )

    s = 0
    for i in range(101):
        if i == 1:
            s = time.perf_counter()
        if i % 10 == 0:
            print("step", i)
        ray.get(dag.execute(1))
    e = time.perf_counter()

    print(f"E2E Time: {e - s} s")

    # import time
    # time.sleep(5)

    # for i, worker in enumerate(workers):
    #     timer = ray.get(worker.pop_timer.remote())
    #     for k, v in timer.items():
    #         v = sorted(v)
    #         print(f"Worker {i} TIMER-{k}")
    #         print("AVG =", sum(v)/len(v) * 1000)
    #         for percent in [0.1, 0.5, 0.8, 0.9, 0.95, 0.99]:
    #             print(percent, v[int(percent * len(v))] * 1000)

    # print(f"Schedule of {'dummy' if args.use_dummy else 'normal'} workers:")
    from collections import Counter
    for worker in workers:
        trace = ray.get(worker.pop_trace.remote())
        print(Counter(trace))

    # import pandas as pd
    # records = []
    # for i, worker in enumerate(workers):
    #     print(f"Worker {i}")
    #     events = ray.get(worker.get_events.remote())
    #     records.extend(events)

    # df = pd.DataFrame.from_records(records)
    # df.to_csv(f"events_zbh1.csv")
