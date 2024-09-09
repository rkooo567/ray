import random
import ray
import os
import sys
import time
import pytest
from ray.dag import InputNode, MultiOutputNode
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.dag import DAGContext
from ray.tests.conftest import *  # noqa

if sys.platform != "linux" and sys.platform != "darwin":
    pytest.skip("Skipping, requires Linux or Mac.", allow_module_level=True)


@ray.remote
class Actor:
    def __init__(self, init_value, fail_after=None, sys_exit=False):
        self.i = init_value
        self.fail_after = fail_after
        self.sys_exit = sys_exit

        self.count = 0

    def _fail_if_needed(self):
        if self.fail_after and self.count > self.fail_after:
            # Randomize the failures to better cover multi actor scenarios.
            if random.random() > 0.5:
                if self.sys_exit:
                    os._exit(1)
                else:
                    raise RuntimeError("injected fault")

    def inc(self, x):
        self.i += x
        self.count += 1
        self._fail_if_needed()
        return self.i

    def double_and_inc(self, x):
        self.i *= 2
        self.i += x
        return self.i

    def echo(self, x):
        print("ECHO!")
        self.count += 1
        self._fail_if_needed()
        return x

    def append_to(self, lst):
        lst.append(self.i)
        return lst

    def inc_two(self, x, y):
        self.i += x
        self.i += y
        return self.i

    def sleep(self, x):
        time.sleep(x)
        return x

    @ray.method(num_returns=2)
    def return_two(self, x):
        return x, x + 1


def test_readers_on_different_nodes(ray_start_cluster):
    cluster = ray_start_cluster
    # This node is for the driver (including the CompiledDAG.DAGDriverProxyActor) and
    # one of the readers.
    first_node_handle = cluster.add_node(num_cpus=2)
    # This node is for the other reader.
    second_node_handle = cluster.add_node(num_cpus=1)
    ray.init(address=cluster.address)
    cluster.wait_for_nodes()

    nodes = [first_node_handle.node_id, second_node_handle.node_id]
    # We want to check that the readers are on different nodes. Thus, we convert `nodes`
    # to a set and then back to a list to remove duplicates. Then we check that the
    # length of `nodes` is 2.
    nodes = list(set(nodes))
    assert len(nodes) == 2

    def create_actor(node):
        return Actor.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node, soft=False)
        ).remote(0)

    a = create_actor(nodes[0])
    b = create_actor(nodes[1])
    actors = [a, b]

    def _get_node_id(self) -> "ray.NodeID":
        return ray.get_runtime_context().get_node_id()

    nodes_check = ray.get([act.__ray_call__.remote(_get_node_id) for act in actors])
    a_node = nodes_check[0]
    b_node = nodes_check[1]
    assert a_node != b_node

    with InputNode() as inp:
        x = a.inc.bind(inp)
        y = b.inc.bind(inp)
        dag = MultiOutputNode([x, y])

    with pytest.raises(
        ValueError,
        match="All reader actors must be on the same node.*",
    ):
        dag.experimental_compile()


def test_bunch_readers_on_different_nodes(ray_start_cluster):
    cluster = ray_start_cluster
    # This node is for the driver (including the CompiledDAG.DAGDriverProxyActor) and
    # two of the readers.
    first_node_handle = cluster.add_node(num_cpus=3)
    # This node is for the other two readers.
    second_node_handle = cluster.add_node(num_cpus=2)
    ray.init(address=cluster.address)
    cluster.wait_for_nodes()

    nodes = [first_node_handle.node_id, second_node_handle.node_id]
    # We want to check that the readers are on different nodes. Thus, we convert `nodes`
    # to a set and then back to a list to remove duplicates. Then we check that the
    # length of `nodes` is 2.
    nodes = list(set(nodes))
    assert len(nodes) == 2

    def create_actor(node):
        return Actor.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node, soft=False)
        ).remote(0)

    a = create_actor(nodes[0])
    b = create_actor(nodes[0])
    c = create_actor(nodes[1])
    d = create_actor(nodes[1])
    actors = [a, b, c, d]

    def _get_node_id(self) -> "ray.NodeID":
        return ray.get_runtime_context().get_node_id()

    nodes_check = ray.get([act.__ray_call__.remote(_get_node_id) for act in actors])
    a_node = nodes_check[0]
    b_node = nodes_check[1]
    c_node = nodes_check[2]
    d_node = nodes_check[3]
    assert a_node == b_node
    assert b_node != c_node
    assert c_node == d_node

    with InputNode() as inp:
        w = a.inc.bind(inp)
        x = b.inc.bind(inp)
        y = c.inc.bind(inp)
        z = d.inc.bind(inp)
        dag = MultiOutputNode([w, x, y, z])

    with pytest.raises(
        ValueError,
        match="All reader actors must be on the same node.*",
    ):
        dag.experimental_compile()


def test_pp(ray_start_cluster):
    cluster = ray_start_cluster
    # This node is for the driver.
    cluster.add_node(num_cpus=0)
    ray.init(address=cluster.address)
    TP = 2
    # This node is for the PP stage 1.
    cluster.add_node(resources={"pp1": TP})
    # This node is for the PP stage 2.
    cluster.add_node(resources={"pp2": TP})

    @ray.remote
    class Worker:
        def __init__(self):
            pass

        def execute_model(self, val):
            return val

    pp1_workers = [
        Worker.options(num_cpus=0, resources={"pp1": 1}).remote() for _ in range(TP)
    ]
    pp2_workers = [
        Worker.options(num_cpus=0, resources={"pp2": 1}).remote() for _ in range(TP)
    ]

    with InputNode() as inp:
        outputs = [inp for _ in range(TP)]
        outputs = [pp1_workers[i].execute_model.bind(outputs[i]) for i in range(TP)]
        outputs = [pp2_workers[i].execute_model.bind(outputs[i]) for i in range(TP)]
        dag = MultiOutputNode(outputs)

    compiled_dag = dag.experimental_compile()
    ref = compiled_dag.execute(1)
    assert ray.get(ref) == [1] * TP

    # So that raylets' error messages are printed to the driver
    time.sleep(2)

    compiled_dag.teardown()


def test_payload_large(ray_start_cluster, monkeypatch):
    GRPC_MAX_SIZE = 1024 * 1024 * 5
    monkeypatch.setenv("RAY_max_grpc_message_size", str(GRPC_MAX_SIZE))
    cluster = ray_start_cluster
    # This node is for the driver (including the CompiledDAG.DAGDriverProxyActor).
    first_node_handle = cluster.add_node(num_cpus=1)
    # This node is for the reader.
    second_node_handle = cluster.add_node(num_cpus=1)
    ray.init(address=cluster.address)
    cluster.wait_for_nodes()

    nodes = [first_node_handle.node_id, second_node_handle.node_id]
    # We want to check that there are two nodes. Thus, we convert `nodes` to a set and
    # then back to a list to remove duplicates. Then we check that the length of `nodes`
    # is 2.
    nodes = list(set(nodes))
    assert len(nodes) == 2

    def create_actor(node):
        return Actor.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node, soft=False)
        ).remote(0)

    def get_node_id(self):
        return ray.get_runtime_context().get_node_id()

    driver_node = get_node_id(None)
    nodes.remove(driver_node)

    a = create_actor(nodes[0])
    a_node = ray.get(a.__ray_call__.remote(get_node_id))
    assert a_node == nodes[0]
    # Check that the driver and actor are on different nodes.
    assert driver_node != a_node

    with InputNode() as i:
        dag = a.echo.bind(i)

    compiled_dag = dag.experimental_compile()

    # Ray sets the gRPC payload max size to 512 MiB. We choose a size in this test that
    # is a bit larger.
    size = GRPC_MAX_SIZE + (1024 * 1024 * 2)
    val = b"x" * size

    for i in range(3):
        print(f"{i} iteration")
        ref = compiled_dag.execute(val)
        result = ray.get(ref)
        assert result == val

    # Note: must teardown before starting a new Ray session, otherwise you'll get
    # a segfault from the dangling monitor thread upon the new Ray init.
    compiled_dag.teardown()


if __name__ == "__main__":
    if os.environ.get("PARALLEL_CI"):
        sys.exit(pytest.main(["-n", "auto", "--boxed", "-vs", __file__]))
    else:
        sys.exit(pytest.main(["-sv", __file__]))
