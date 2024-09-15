import ray
from ray.dag import InputNode
from ray.experimental.channel.torch_tensor_type import TorchTensorType
import torch
ray.init()

@ray.remote(num_gpus=1)
class A:
    @ray.method(num_returns=2)
    def f(self, a):
        return torch.tensor([1], device="cuda"), a


@ray.remote(num_gpus=1)
class B:
    def f(self, a, b):
        print(a, b)
        return a


a = A.remote()
b = B.remote()

with InputNode() as inp:
    x, y = a.f.bind(inp)
    x.with_type_hint(TorchTensorType(transport="nccl"))
    dag = b.f.bind(x, y)

dag = dag.experimental_compile()
ray.get(dag.execute(1))
