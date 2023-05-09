import asyncio
import pytest
import numpy as np
import sys
import time

import ray
from ray.util.client.ray_client_helpers import (
    ray_start_client_server_for_address,
)
from ray._private.client_mode_hook import enable_client_mode
from ray.tests.conftest import call_ray_start_context
from ray._private.test_utils import wait_for_condition
from ray.experimental.state.api import list_tasks
from ray._raylet import StreamingObjectRefGeneratorV2


@pytest.mark.skipif(
    sys.platform != "linux" and sys.platform != "linux2",
    reason="memory monitor only on linux currently",
)
def test_generator_oom(ray_start_regular):
    @ray.remote(max_retries=0)
    def large_values(num_returns):
        return [
            np.random.randint(
                np.iinfo(np.int8).max, size=(100_000_000, 1), dtype=np.int8
            )
            for _ in range(num_returns)
        ]

    @ray.remote(max_retries=0)
    def large_values_generator(num_returns):
        for _ in range(num_returns):
            yield np.random.randint(
                np.iinfo(np.int8).max, size=(100_000_000, 1), dtype=np.int8
            )

    num_returns = 100
    try:
        # Worker may OOM using normal returns.
        ray.get(large_values.options(num_returns=num_returns).remote(num_returns)[0])
    except ray.exceptions.WorkerCrashedError:
        pass

    # Using a generator will allow the worker to finish.
    ray.get(
        large_values_generator.options(num_returns=num_returns).remote(num_returns)[0]
    )


def test_generator_basic(shutdown_only):
    ray.init(num_cpus=1)

    # """Basic cases"""
    @ray.remote
    def f():
        for i in range(5):
            yield i

    gen = f.options(num_returns="dynamic").remote()
    i = 0
    for ref in gen:
        print(ray.get(ref))
        assert i == ray.get(ref)
        del ref
        i += 1

    """Exceptions"""

    @ray.remote
    def f():
        for i in range(5):
            if i == 2:
                raise ValueError
            yield i

    gen = f.options(num_returns="dynamic").remote()
    ray.get(next(gen))
    ray.get(next(gen))
    with pytest.raises(ray.exceptions.RayTaskError) as e:
        ray.get(next(gen))
    print(str(e.value))
    with pytest.raises(StopIteration):
        ray.get(next(gen))
    with pytest.raises(StopIteration):
        ray.get(next(gen))

    """Generator Task failure"""

    @ray.remote
    class A:
        def getpid(self):
            import os

            return os.getpid()

        def f(self):
            for i in range(5):
                import time

                time.sleep(0.1)
                yield i

    a = A.remote()
    i = 0
    gen = a.f.options(num_returns="dynamic").remote()
    i = 0
    for ref in gen:
        if i == 2:
            ray.kill(a)
        if i == 3:
            with pytest.raises(ray.exceptions.RayActorError) as e:
                ray.get(ref)
            assert "The actor is dead because it was killed by `ray.kill`" in str(
                e.value
            )
            break
        assert i == ray.get(ref)
        del ref
        i += 1
    for _ in range(10):
        with pytest.raises(StopIteration):
            next(gen)

    """Retry exceptions"""

    @ray.remote
    class Actor:
        def __init__(self):
            self.should_kill = True

        def should_kill(self):
            return self.should_kill

        async def set(self, wait_s):
            await asyncio.sleep(wait_s)
            self.should_kill = False

    @ray.remote(retry_exceptions=[ValueError], max_retries=10)
    def f(a):
        for i in range(5):
            should_kill = ray.get(a.should_kill.remote())
            if i == 3 and should_kill:
                raise ValueError
            yield i

    a = Actor.remote()
    gen = f.options(num_returns="dynamic").remote(a)
    assert ray.get(next(gen)) == 0
    assert ray.get(next(gen)) == 1
    assert ray.get(next(gen)) == 2
    a.set.remote(3)
    assert ray.get(next(gen)) == 3
    assert ray.get(next(gen)) == 4
    with pytest.raises(StopIteration):
        ray.get(next(gen))

    """Cancel"""

    @ray.remote
    def f():
        for i in range(5):
            time.sleep(5)
            yield i

    gen = f.options(num_returns="dynamic").remote()
    assert ray.get(next(gen)) == 0
    ray.cancel(gen)
    with pytest.raises(ray.exceptions.RayTaskError) as e:
        assert ray.get(next(gen)) == 1
    assert "was cancelled" in str(e.value)
    with pytest.raises(StopIteration):
        next(gen)


@pytest.mark.parametrize("use_actors", [False, True])
@pytest.mark.parametrize("store_in_plasma", [False, True])
def test_generator_streaming(shutdown_only, use_actors, store_in_plasma):
    """Verify the generator is working in a streaming fashion."""
    ray.init()
    remote_generator_fn = None
    if use_actors:

        @ray.remote
        class Generator:
            def __init__(self):
                pass

            def generator(self, num_returns, store_in_plasma):
                for i in range(num_returns):
                    if store_in_plasma:
                        yield np.ones(1_000_000, dtype=np.int8) * i
                    else:
                        yield [i]

        g = Generator.remote()
        remote_generator_fn = g.generator
    else:

        @ray.remote(max_retries=0)
        def generator(num_returns, store_in_plasma):
            for i in range(num_returns):
                if store_in_plasma:
                    yield np.ones(1_000_000, dtype=np.int8) * i
                else:
                    yield [i]

        remote_generator_fn = generator

    """Verify num_returns="dynamic" is streaming"""
    gen = remote_generator_fn.options(num_returns="dynamic").remote(3, store_in_plasma)
    for ref in gen:
        id = ref.hex()
        print(ray.get(ref))
        del ref
        from ray.experimental.state.api import list_objects

        wait_for_condition(
            lambda: len(list_objects(filters=[("object_id", "=", id)])) == 0
        )


@pytest.mark.parametrize("use_actors", [False, True])
@pytest.mark.parametrize("store_in_plasma", [False, True])
def test_generator_returns(ray_start_regular, use_actors, store_in_plasma):
    remote_generator_fn = None
    if use_actors:

        @ray.remote
        class Generator:
            def __init__(self):
                pass

            def generator(self, num_returns, store_in_plasma):
                for i in range(num_returns):
                    if store_in_plasma:
                        yield np.ones(1_000_000, dtype=np.int8) * i
                    else:
                        yield [i]

        g = Generator.remote()
        remote_generator_fn = g.generator
    else:

        @ray.remote(max_retries=0)
        def generator(num_returns, store_in_plasma):
            for i in range(num_returns):
                if store_in_plasma:
                    yield np.ones(1_000_000, dtype=np.int8) * i
                else:
                    yield [i]

        remote_generator_fn = generator

    # Check cases when num_returns does not match the number of values returned
    # by the generator.
    num_returns = 3

    try:
        ray.get(
            remote_generator_fn.options(num_returns=num_returns).remote(
                num_returns - 1, store_in_plasma
            )
        )
        assert False
    except ray.exceptions.RayTaskError as e:
        assert isinstance(e.as_instanceof_cause(), ValueError)

    # TODO(swang): When generators return more values than expected, we log an
    # error but the exception is not thrown to the application.
    # https://github.com/ray-project/ray/issues/28689.
    ray.get(
        remote_generator_fn.options(num_returns=num_returns).remote(
            num_returns + 1, store_in_plasma
        )
    )

    # Check return values.
    [
        x[0]
        for x in ray.get(
            remote_generator_fn.options(num_returns=num_returns).remote(
                num_returns, store_in_plasma
            )
        )
    ] == list(range(num_returns))
    # Works for num_returns=1 if generator returns a single value.
    assert (
        ray.get(remote_generator_fn.options(num_returns=1).remote(1, store_in_plasma))[
            0
        ]
        == 0
    )


@pytest.mark.parametrize("use_actors", [False, True])
@pytest.mark.parametrize("store_in_plasma", [False, True])
def test_generator_errors(ray_start_regular, use_actors, store_in_plasma):
    remote_generator_fn = None
    if use_actors:

        @ray.remote
        class Generator:
            def __init__(self):
                pass

            def generator(self, num_returns, store_in_plasma):
                for i in range(num_returns - 2):
                    if store_in_plasma:
                        yield np.ones(1_000_000, dtype=np.int8) * i
                    else:
                        yield [i]
                raise Exception("error")

        g = Generator.remote()
        remote_generator_fn = g.generator
    else:

        @ray.remote(max_retries=0)
        def generator(num_returns, store_in_plasma):
            for i in range(num_returns - 2):
                if store_in_plasma:
                    yield np.ones(1_000_000, dtype=np.int8) * i
                else:
                    yield [i]
            raise Exception("error")

        remote_generator_fn = generator

    ref1, ref2, ref3 = remote_generator_fn.options(num_returns=3).remote(
        3, store_in_plasma
    )
    ray.get(ref1)
    with pytest.raises(ray.exceptions.RayTaskError):
        ray.get(ref2)
    with pytest.raises(ray.exceptions.RayTaskError):
        ray.get(ref3)

    dynamic_ref = remote_generator_fn.options(num_returns="dynamic").remote(
        3, store_in_plasma
    )
    ref1, ref2 = ray.get(dynamic_ref)
    ray.get(ref1)
    with pytest.raises(ray.exceptions.RayTaskError):
        ray.get(ref2)


@pytest.mark.parametrize("store_in_plasma", [False, True])
def test_dynamic_generator_retry_exception(ray_start_regular, store_in_plasma):
    class CustomException(Exception):
        pass

    @ray.remote(num_cpus=0)
    class ExecutionCounter:
        def __init__(self):
            self.count = 0

        def inc(self):
            self.count += 1
            return self.count

        def get_count(self):
            return self.count

        def reset(self):
            self.count = 0

    @ray.remote(max_retries=1)
    def generator(num_returns, store_in_plasma, counter):
        for i in range(num_returns):
            if store_in_plasma:
                yield np.ones(1_000_000, dtype=np.int8) * i
            else:
                yield [i]

            # Fail on first execution, succeed on next.
            if ray.get(counter.inc.remote()) == 1:
                raise CustomException("error")

    counter = ExecutionCounter.remote()
    dynamic_ref = generator.options(num_returns="dynamic").remote(
        3, store_in_plasma, counter
    )
    ref1, ref2 = ray.get(dynamic_ref)
    ray.get(ref1)
    with pytest.raises(ray.exceptions.RayTaskError):
        ray.get(ref2)

    ray.get(counter.reset.remote())
    dynamic_ref = generator.options(
        num_returns="dynamic", retry_exceptions=[CustomException]
    ).remote(3, store_in_plasma, counter)
    for i, ref in enumerate(ray.get(dynamic_ref)):
        assert ray.get(ref)[0] == i


@pytest.mark.parametrize("use_actors", [False, True])
@pytest.mark.parametrize("store_in_plasma", [False, True])
def test_dynamic_generator(ray_start_regular, use_actors, store_in_plasma):
    if not use_actors:

        @ray.remote(num_returns="dynamic")
        def dynamic_generator(num_returns, store_in_plasma):
            for i in range(num_returns):
                if store_in_plasma:
                    yield np.ones(1_000_000, dtype=np.int8) * i
                else:
                    yield [i]

        remote_generator_fn = dynamic_generator

    else:

        @ray.remote
        class Generator:
            def __init__(self):
                pass

            def generator(self, num_returns, store_in_plasma):
                for i in range(num_returns):
                    if store_in_plasma:
                        yield np.ones(1_000_000, dtype=np.int8) * i
                    else:
                        yield [i]

        g = Generator.remote()
        remote_generator_fn = g.generator

    @ray.remote
    def read(gen):
        for i, ref in enumerate(gen):
            if ray.get(ref)[0] != i:
                return False
        return True

    gen = ray.get(
        remote_generator_fn.options(num_returns="dynamic").remote(10, store_in_plasma)
    )
    for i, ref in enumerate(gen):
        print(ray.get(ref))
        print(i, ref)
        assert ray.get(ref)[0] == i

    # Test empty generator.
    gen = ray.get(
        remote_generator_fn.options(num_returns="dynamic").remote(0, store_in_plasma)
    )
    with pytest.raises(StopIteration):
        next(gen)
    gen = ray.get(
        remote_generator_fn.options(num_returns="dynamic").remote(0, store_in_plasma)
    )
    assert len(list(gen)) == 0

    # Check that passing as task arg.
    # SANG-TODO This is not allowed.
    # gen = remote_generator_fn.options(num_returns="dynamic").remote(10, store_in_plasma)
    # assert ray.get(read.remote(gen))
    # assert ray.get(read.remote(ray.get(gen)))

    # Also works if we override num_returns with a static value.
    # ray.get(
    #     read.remote(
    #         remote_generator_fn.options(num_returns=10).remote(10, store_in_plasma)
    #     )
    # )

    # Normal remote functions don't work with num_returns="dynamic".
    @ray.remote(num_returns="dynamic")
    def static(num_returns):
        return list(range(num_returns))

    with pytest.raises(ray.exceptions.RayTaskError):
        gen = ray.get(static.remote(3))
        ref = next(gen)
        ray.get(ref)


def test_dynamic_generator_distributed(ray_start_cluster):
    cluster = ray_start_cluster
    # Head node with no resources.
    cluster.add_node(num_cpus=0, object_store_memory=1e9)
    ray.init(address=cluster.address)
    cluster.add_node(num_cpus=1, object_store_memory=1e9)
    cluster.wait_for_nodes()

    @ray.remote(num_returns="dynamic")
    def dynamic_generator(num_returns):
        for i in range(num_returns):
            yield np.ones(1_000_000, dtype=np.int8) * i
            time.sleep(0.1)

    gen = ray.get(dynamic_generator.remote(3))
    for i, ref in enumerate(gen):
        print(ref.hex())
        # Check that we can fetch the values from a different node.
        assert ray.get(ref)[0] == i


def test_dynamic_generator_reconstruction(ray_start_cluster):
    config = {
        "health_check_failure_threshold": 10,
        "health_check_period_ms": 100,
        "health_check_initial_delay_ms": 0,
        "max_direct_call_object_size": 100,
        "task_retry_delay_ms": 100,
        "object_timeout_milliseconds": 200,
        "fetch_warn_timeout_milliseconds": 1000,
    }
    cluster = ray_start_cluster
    # Head node with no resources.
    cluster.add_node(
        num_cpus=0, _system_config=config, enable_object_reconstruction=True
    )
    ray.init(address=cluster.address)
    # Node to place the initial object.
    node_to_kill = cluster.add_node(num_cpus=1, object_store_memory=10**8)
    cluster.wait_for_nodes()

    @ray.remote(num_returns="dynamic")
    def dynamic_generator(num_returns):
        for i in range(num_returns):
            # Random ray.put to make sure it's okay to interleave these with
            # the dynamic returns.
            if np.random.randint(2) == 1:
                ray.put(np.ones(1_000_000, dtype=np.int8) * np.random.randint(100))
            yield np.ones(1_000_000, dtype=np.int8) * i

    @ray.remote
    def fetch(x):
        return x[0]

    # Test recovery of all dynamic objects through re-execution.
    gen = ray.get(dynamic_generator.remote(10))
    cluster.remove_node(node_to_kill, allow_graceful=False)
    node_to_kill = cluster.add_node(num_cpus=1, object_store_memory=10**8)
    refs = list(gen)
    for i, ref in enumerate(refs):
        assert ray.get(fetch.remote(ref)) == i

    cluster.add_node(num_cpus=1, resources={"node2": 1}, object_store_memory=10**8)

    # Fetch one of the ObjectRefs to another node. We should try to reuse this
    # copy during recovery.
    ray.get(fetch.options(resources={"node2": 1}).remote(refs[-1]))
    cluster.remove_node(node_to_kill, allow_graceful=False)
    for i, ref in enumerate(refs):
        assert ray.get(fetch.remote(ref)) == i


@pytest.mark.parametrize("too_many_returns", [False, True])
def test_dynamic_generator_reconstruction_nondeterministic(
    ray_start_cluster, too_many_returns
):
    config = {
        "health_check_failure_threshold": 10,
        "health_check_period_ms": 100,
        "health_check_initial_delay_ms": 0,
        "max_direct_call_object_size": 100,
        "task_retry_delay_ms": 100,
        "object_timeout_milliseconds": 200,
        "fetch_warn_timeout_milliseconds": 1000,
    }
    cluster = ray_start_cluster
    # Head node with no resources.
    cluster.add_node(
        num_cpus=1,
        _system_config=config,
        enable_object_reconstruction=True,
        resources={"head": 1},
    )
    ray.init(address=cluster.address)
    # Node to place the initial object.
    node_to_kill = cluster.add_node(num_cpus=1, object_store_memory=10**8)
    cluster.wait_for_nodes()

    @ray.remote(num_cpus=1, resources={"head": 1})
    class FailureSignal:
        def __init__(self):
            return

        def ping(self):
            return

    @ray.remote(num_returns="dynamic")
    def dynamic_generator(failure_signal):
        num_returns = 10
        try:
            ray.get(failure_signal.ping.remote())
        except ray.exceptions.RayActorError:
            if too_many_returns:
                num_returns += 1
            else:
                num_returns -= 1
        for i in range(num_returns):
            yield np.ones(1_000_000, dtype=np.int8) * i

    @ray.remote
    def fetch(x):
        return

    failure_signal = FailureSignal.remote()
    gen = ray.get(dynamic_generator.remote(failure_signal))
    cluster.remove_node(node_to_kill, allow_graceful=False)
    ray.kill(failure_signal)
    refs = list(gen)
    if too_many_returns:
        for i, ref in enumerate(refs):
            assert np.array_equal(np.ones(1_000_000, dtype=np.int8) * i, ray.get(ref))
    else:
        for i, ref in enumerate(refs):
            assert np.array_equal(np.ones(1_000_000, dtype=np.int8) * i, ray.get(ref))
    # TODO(swang): If the re-executed task returns a different number of
    # objects, we should throw an error for every return value.
    # for ref in refs:
    #     with pytest.raises(ray.exceptions.RayTaskError):
    #         ray.get(ref)


def test_dynamic_generator_reconstruction_fails(ray_start_cluster):
    config = {
        "health_check_failure_threshold": 10,
        "health_check_period_ms": 100,
        "health_check_initial_delay_ms": 0,
        "max_direct_call_object_size": 100,
        "task_retry_delay_ms": 100,
        "object_timeout_milliseconds": 200,
        "fetch_warn_timeout_milliseconds": 1000,
    }
    cluster = ray_start_cluster
    cluster.add_node(
        num_cpus=1,
        _system_config=config,
        enable_object_reconstruction=True,
        resources={"head": 1},
    )
    ray.init(address=cluster.address)
    # Node to place the initial object.
    node_to_kill = cluster.add_node(num_cpus=1, object_store_memory=10**8)
    cluster.wait_for_nodes()

    @ray.remote(num_cpus=1, resources={"head": 1})
    class FailureSignal:
        def __init__(self):
            return

        def ping(self):
            return

    @ray.remote(num_returns="dynamic")
    def dynamic_generator(failure_signal):
        num_returns = 10
        for i in range(num_returns):
            yield np.ones(1_000_000, dtype=np.int8) * i
            if i == num_returns // 2:
                # If this is the re-execution, fail the worker after partial yield.
                try:
                    ray.get(failure_signal.ping.remote())
                except ray.exceptions.RayActorError:
                    sys.exit(-1)

    @ray.remote
    def fetch(*refs):
        pass

    failure_signal = FailureSignal.remote()
    gen = ray.get(dynamic_generator.remote(failure_signal))
    refs = list(gen)
    ray.get(fetch.remote(*refs))

    cluster.remove_node(node_to_kill, allow_graceful=False)
    done = fetch.remote(*refs)

    ray.kill(failure_signal)
    # Make sure we can get the error.
    with pytest.raises(ray.exceptions.WorkerCrashedError):
        for ref in refs:
            ray.get(ref)
    # Make sure other tasks can also get the error.
    with pytest.raises(ray.exceptions.RayTaskError):
        ray.get(done)


def test_dynamic_empty_generator_reconstruction_nondeterministic(ray_start_cluster):
    config = {
        "health_check_failure_threshold": 10,
        "health_check_period_ms": 100,
        "health_check_initial_delay_ms": 0,
        "max_direct_call_object_size": 100,
        "task_retry_delay_ms": 100,
        "object_timeout_milliseconds": 200,
        "fetch_warn_timeout_milliseconds": 1000,
    }
    cluster = ray_start_cluster
    # Head node with no resources.
    cluster.add_node(
        num_cpus=0,
        _system_config=config,
        enable_object_reconstruction=True,
        resources={"head": 1},
    )
    ray.init(address=cluster.address)
    # Node to place the initial object.
    node_to_kill = cluster.add_node(num_cpus=1, object_store_memory=10**8)
    cluster.wait_for_nodes()

    @ray.remote(num_cpus=0, resources={"head": 1})
    class ExecutionCounter:
        def __init__(self):
            self.count = 0

        def inc(self):
            self.count += 1
            return self.count

        def get_count(self):
            return self.count

    @ray.remote(num_returns="dynamic")
    def maybe_empty_generator(exec_counter):
        if ray.get(exec_counter.inc.remote()) > 1:
            for i in range(3):
                yield np.ones(1_000_000, dtype=np.int8) * i

    @ray.remote
    def check(empty_generator):
        return len(empty_generator) == 0

    exec_counter = ExecutionCounter.remote()
    gen = maybe_empty_generator.remote(exec_counter)
    refs = list(gen)
    assert ray.get(check.remote(refs))
    cluster.remove_node(node_to_kill, allow_graceful=False)
    node_to_kill = cluster.add_node(num_cpus=1, object_store_memory=10**8)
    assert ray.get(check.remote(refs))

    # We should never reconstruct an empty generator.
    assert ray.get(exec_counter.get_count.remote()) == 1


# # Client server port of the shared Ray instance
# SHARED_CLIENT_SERVER_PORT = 25555


# @pytest.fixture(scope="module")
# def call_ray_start_shared(request):
#     request = Mock()
#     request.param = (
#         "ray start --head --min-worker-port=0 --max-worker-port=0 --port 0 "
#         f"--ray-client-server-port={SHARED_CLIENT_SERVER_PORT}"
#     )
#     with call_ray_start_context(request) as address:
#         yield address


# @pytest.mark.parametrize("store_in_plasma", [False, True])
# def test_ray_client(call_ray_start_shared, store_in_plasma):
#     with ray_start_client_server_for_address(call_ray_start_shared):
#         enable_client_mode()

#         @ray.remote(max_retries=0)
#         def generator(num_returns, store_in_plasma):
#             for i in range(num_returns):
#                 if store_in_plasma:
#                     yield np.ones(1_000_000, dtype=np.int8) * i
#                 else:
#                     yield [i]

#         # TODO(swang): When generators return more values than expected, we log an
#         # error but the exception is not thrown to the application.
#         # https://github.com/ray-project/ray/issues/28689.
#         num_returns = 3
#         ray.get(
#             generator.options(num_returns=num_returns).remote(
#                 num_returns + 1, store_in_plasma
#             )
#         )

#         # Check return values.
#         [
#             x[0]
#             for x in ray.get(
#                 generator.options(num_returns=num_returns).remote(
#                     num_returns, store_in_plasma
#                 )
#             )
#         ] == list(range(num_returns))
#         # Works for num_returns=1 if generator returns a single value.
#         assert (
#             ray.get(generator.options(num_returns=1).remote(1, store_in_plasma))[0] == 0
#         )

#         gen = ray.get(
#             generator.options(num_returns="dynamic").remote(3, store_in_plasma)
#         )
#         for i, ref in enumerate(gen):
#             assert ray.get(ref)[0] == i


@pytest.mark.parametrize("store_in_plasma", [False])
def test_actor_streaming_generator(shutdown_only, store_in_plasma):
    ray.init()

    @ray.remote
    class Actor:
        def f(self, ref):
            for i in range(3):
                yield i

        async def async_f(self, ref):
            for i in range(3):
                await asyncio.sleep(0.1)
                yield i

        def g(self):
            return 3

    a = Actor.remote()
    if store_in_plasma:
        arr = np.random.rand(5 * 1024 * 1024)
    else:
        arr = 3

    def verify_sync_task_executor():
        generator = a.f.options(num_returns="dynamic").remote(ray.put(arr))
        # Verify it works with next.
        assert isinstance(generator, StreamingObjectRefGeneratorV2)
        assert ray.get(next(generator)) == 0
        assert ray.get(next(generator)) == 1
        assert ray.get(next(generator)) == 2
        with pytest.raises(StopIteration):
            ray.get(next(generator))

        # Verify it works with for.
        generator = a.f.options(num_returns="dynamic").remote(ray.put(3))
        for index, ref in enumerate(generator):
            assert index == ray.get(ref)

    def verify_async_task_executor():
        # Verify it works with next.
        generator = a.async_f.options(num_returns="dynamic").remote(ray.put(arr))
        assert isinstance(generator, StreamingObjectRefGeneratorV2)
        assert ray.get(next(generator)) == 0
        assert ray.get(next(generator)) == 1
        assert ray.get(next(generator)) == 2

        # Verify it works with for.
        generator = a.f.options(num_returns="dynamic").remote(ray.put(3))
        for index, ref in enumerate(generator):
            assert index == ray.get(ref)

    async def verify_sync_task_async_generator():
        # Verify anext
        async_generator = a.f.options(num_returns="dynamic").remote(ray.put(arr))
        assert isinstance(async_generator, StreamingObjectRefGeneratorV2)
        for expected in range(3):
            ref = await async_generator.__anext__()
            assert await ref == expected
        with pytest.raises(StopAsyncIteration):
            await async_generator.__anext__()

        # Verify async for.
        async_generator = a.f.options(num_returns="dynamic").remote(ray.put(arr))
        expected = 0
        async for ref in async_generator:
            value = await ref
            assert value == value
            expected += 1

    async def verify_async_task_async_generator():
        async_generator = a.async_f.options(num_returns="dynamic").remote(ray.put(arr))
        assert isinstance(async_generator, StreamingObjectRefGeneratorV2)
        for expected in range(3):
            ref = await async_generator.__anext__()
            assert await ref == expected
        with pytest.raises(StopAsyncIteration):
            await async_generator.__anext__()

        # Verify async for.
        async_generator = a.async_f.options(num_returns="dynamic").remote(ray.put(arr))
        expected = 0
        async for value in async_generator:
            value = await ref
            assert value == value
            expected += 1

    verify_sync_task_executor()
    verify_async_task_executor()
    asyncio.run(verify_sync_task_async_generator())
    asyncio.run(verify_async_task_async_generator())


def test_streaming_generator_exception(shutdown_only):
    # Verify the exceptions are correctly raised.
    # Also verify the followup next will raise StopIteration.
    ray.init()

    @ray.remote
    class Actor:
        def f(self):
            raise ValueError
            yield 1  # noqa

        async def async_f(self):
            raise ValueError
            yield 1  # noqa

    a = Actor.remote()
    g = a.f.options(num_returns="dynamic").remote()
    with pytest.raises(ValueError):
        ray.get(next(g))

    with pytest.raises(StopIteration):
        ray.get(next(g))

    with pytest.raises(StopIteration):
        ray.get(next(g))

    g = a.async_f.options(num_returns="dynamic").remote()
    with pytest.raises(ValueError):
        ray.get(next(g))

    with pytest.raises(StopIteration):
        ray.get(next(g))

    with pytest.raises(StopIteration):
        ray.get(next(g))


def test_threaded_actor_generator(shutdown_only):
    ray.init()

    @ray.remote(max_concurrency=10)
    class Actor:
        def f(self):
            for i in range(30):
                time.sleep(0.1)
                yield np.ones(1024 * 1024) * i

    @ray.remote(max_concurrency=20)
    class AsyncActor:
        async def f(self):
            for i in range(30):
                await asyncio.sleep(0.1)
                yield np.ones(1024 * 1024) * i

    async def main():
        a = Actor.remote()
        asy = AsyncActor.remote()

        async def run():
            i = 0
            async for ref in a.f.options(num_returns="dynamic").remote():
                val = ray.get(ref)
                print(val)
                print(ref)
                assert np.array_equal(val, np.ones(1024 * 1024) * i)
                i += 1
                del ref

        async def run2():
            i = 0
            async for ref in asy.f.options(num_returns="dynamic").remote():
                val = await ref
                print(ref)
                print(val)
                assert np.array_equal(val, np.ones(1024 * 1024) * i), ref
                i += 1
                del ref

        coroutines = [run() for _ in range(10)]
        coroutines = [run2() for _ in range(20)]

        await asyncio.gather(*coroutines)

    asyncio.run(main())


def test_generator_dist_chain(ray_start_cluster):
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=0, object_store_memory=1 * 1024 * 1024 * 1024)
    ray.init()
    cluster.add_node(num_cpus=1)
    cluster.add_node(num_cpus=1)
    cluster.add_node(num_cpus=1)
    cluster.add_node(num_cpus=1)

    @ray.remote
    class ChainActor:
        def __init__(self, child=None):
            self.child = child

        def get_data(self):
            if not self.child:
                for _ in range(10):
                    time.sleep(0.1)
                    yield np.ones(5 * 1024 * 1024)
            else:
                for data in self.child.get_data.options(num_returns="dynamic").remote():
                    yield ray.get(data)

    chain_actor = ChainActor.remote()
    chain_actor_2 = ChainActor.remote(chain_actor)
    chain_actor_3 = ChainActor.remote(chain_actor_2)
    chain_actor_4 = ChainActor.remote(chain_actor_3)

    for ref in chain_actor_4.get_data.options(num_returns="dynamic").remote():
        print(ref)
        assert np.array_equal(np.ones(5 * 1024 * 1024), ray.get(ref))
        del ref
    summary = ray._private.internal_api.memory_summary(stats_only=True)
    # SANG-TODO
    # assert "Spilled" not in summary, summary


def test_generator_dist_all_gather(ray_start_cluster):
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=0, object_store_memory=1 * 1024 * 1024 * 1024)
    ray.init()
    cluster.add_node(num_cpus=1)
    cluster.add_node(num_cpus=1)
    cluster.add_node(num_cpus=1)
    cluster.add_node(num_cpus=1)

    @ray.remote(num_cpus=1)
    class Actor:
        def __init__(self, child=None):
            self.child = child

        def get_data(self):
            for _ in range(10):
                time.sleep(0.1)
                yield np.ones(5 * 1024 * 1024)

    async def all_gather():
        actor = Actor.remote()
        async for ref in actor.get_data.options(num_returns="dynamic").remote():
            val = await ref
            assert np.array_equal(np.ones(5 * 1024 * 1024), val)
            del ref

    async def main():
        await asyncio.gather(all_gather(), all_gather(), all_gather(), all_gather())

    asyncio.run(main())
    summary = ray._private.internal_api.memory_summary(stats_only=True)
    print(summary)
    # assert "Spilled" not in summary, summary


if __name__ == "__main__":
    import os

    if os.environ.get("PARALLEL_CI"):
        sys.exit(pytest.main(["-n", "auto", "--boxed", "-vs", __file__]))
    else:
        sys.exit(pytest.main(["-sv", __file__]))
