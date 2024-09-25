import torch
import ray
from ray.experimental.channel.torch_tensor_type import TorchTensorType
from ray.dag import InputNode, MultiOutputNode

ray.init()
@ray.remote(num_gpus=1)
class W:
    def send(self, inp):
        return torch.tensor([inp], device="cuda", dtype=torch.float16)

    def recv(self, tensor):
        assert tensor.device.type == "cuda"
        return tensor
    
    def finish(self, tensor):
        return None

w1 = W.remote()
w2 = W.remote()
w3 = W.remote()
with InputNode() as inp:
    tensors = [w1.send.bind(inp) for _ in range(10)]
    for tensor in tensors:
        tensor.with_type_hint(TorchTensorType(_direct_return=True, _shape=(1,), _dtype=torch.float16, transport="nccl"))
    tensors_2 = [w2.recv.bind(tensor) for tensor in tensors]
    for tensor in tensors_2:
        tensor.with_type_hint(TorchTensorType(_direct_return=True, _shape=(1,), _dtype=torch.float16, transport="nccl"))
    dag = MultiOutputNode([w3.recv.bind(tensor) for tensor in tensors_2])

adag = dag.experimental_compile(_overlap_gpu_communication=True)
# for _ in range(10):
ray.get(adag.execute(1))
