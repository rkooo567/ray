import ray
import numpy as np
import time

ray.init()


def read_and_release(ref, use_bytes, val=None):
    arr = ray.get(ref)
    #arr = worker.core_worker.get_if_local(object_refs)
    if val is not None:
        if use_bytes:
            assert int.from_bytes(arr, "little") == val
        else:
            assert arr[0] == val, (arr, val)

    # Signal to writer that they can write again.
    ray.release(ref)


@ray.remote
class Reader:
    def __init__(self, refs, use_bytes):
        self.ref = refs[0]
        # Keep the plasma object pinned.
        # TODO(swang): Pin the object properly in plasma store.
        self.pinned = ray.get(self.ref)
        print("Object ref:", self.ref)
        self.use_bytes = use_bytes

        read_and_release(self.ref, self.use_bytes)


    def read(self, num_trials):
        for _ in range(num_trials):
            for i in range(10_000):
                read_and_release(self.ref, self.use_bytes, val=i)


def run(num_trials=3, use_bytes=True, reuse_object_ref=False, read_local=False, read_remote=False):
    max_readers = -1
    if reuse_object_ref:
        if read_local or read_remote:
            max_readers = 1
        else:
            max_readers = 0

    if use_bytes:
        arr = b"binary"
    else:
        arr = np.random.rand(1)

    ref = ray.put(arr, max_readers=max_readers)

    if use_bytes:
        assert ray.get(ref) == arr
    else:
        assert np.array_equal(ray.get(ref), arr)

    if reuse_object_ref:
        # Keep the plasma object pinned.
        # TODO(swang): Pin the object properly in plasma store.
        pinned = ray.get(ref)
        print("Object ref:", ref)
    else:
        assert not read_remote

    remote_read_done = None
    if read_local:
        ray.release(ref)
    elif read_remote:
        reader = Reader.remote([ref], use_bytes)
        remote_read_done = reader.read.remote(num_trials)

    print("starting...")

    for _ in range(num_trials):
        start = time.time()
        for i in range(10_000):
            if use_bytes:
                arr = i.to_bytes(8, "little")
            else:
                arr[0] = i

            if reuse_object_ref:
                ray.worker.global_worker.put_object(arr,
                        object_ref=ref, max_readers=max_readers)
            else:
                ref = ray.put(arr, max_readers=max_readers)

            if read_local:
                read_and_release(self.ref, use_bytes, val=i)

        end = time.time()
        print(f"done, tput: {10_000 / (end - start)} puts/s")

    if remote_read_done is not None:
        ray.get(remote_read_done)


if __name__ == "__main__":
    run_local = True

    if not run_local:
        remote_run = ray.remote(run)
        def run_fn(*args, **kwargs):
            return ray.get(remote_run.remote(*args, **kwargs))
        run = run_fn

    print("Dynamic ray.put")
    run()

    print("Reuse ray.put buffer")
    run(reuse_object_ref=True)

    print("Reuse ray.put buffer + local read+release (numpy)")
    # TODO(swang): ray.get doesn't work on bytes? Getting deserialization
    # error.
    run(use_bytes=False, reuse_object_ref=True, read_local=True)

    print("Reuse ray.put buffer + remote read+release (numpy)")
    run(use_bytes=False, reuse_object_ref=True, read_remote=True)
