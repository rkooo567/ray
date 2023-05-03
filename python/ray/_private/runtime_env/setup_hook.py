import traceback
import logging
import base64
import os

from typing import Dict, Any, Callable, Union, Tuple, Optional

import ray
import ray._private.ray_constants as ray_constants
import ray.cloudpickle as pickle
from ray.runtime_env import RuntimeEnv

logger = logging.getLogger(__name__)


def get_import_export_timeout():
    return int(
        os.environ.get(ray_constants.RAY_WORKER_SETUP_HOOK_LOAD_TIMEOUT_KEY, "60")
    )


def _decode_function_key(key: bytes) -> str:
    return base64.b64encode(key).decode()


def _encode_function_key(key: str) -> bytes:
    return base64.b64decode(key)


def upload_worker_setup_hook_if_needed(
    runtime_env: Union[Dict[str, Any], RuntimeEnv],
    worker: "ray.Worker",
) -> Dict[str, Any]:
    """Uploads the worker_setup_hook to GCS with a key.

    runtime_env["worker_setup_hook"] is converted to a decoded key
    that can load the worker setup hook function from GCS.
    I.e., you can use internalKV.Get(runtime_env["worker_setup_hook])
    to access the worker setup hook from GCS.

    Args:
        runtime_env: The runtime_env. The value will be modified
            when returned.
        worker: ray.worker instance.
        decoder: GCS requires the function key to be bytes. However,
            we cannot json serialize (which is required to serialize
            runtime env) the bytes. So the key should be decoded to
            a string. The given decoder is used to decode the function
            key.
    """
    setup_func = runtime_env.get("worker_setup_hook")
    if setup_func is None:
        return runtime_env

    if not isinstance(setup_func, Callable):
        raise TypeError(
            "worker_setup_hook must be a function, " f"got {type(setup_func)}."
        )
    # TODO(sang): Support modules.

    try:
        key = worker.function_actor_manager.export_setup_func(
            setup_func, timeout=get_import_export_timeout()
        )
    except Exception as e:
        raise ray.exceptions.RuntimeEnvSetupError(
            "Failed to export the setup function."
        ) from e
    env_vars = runtime_env.get("env_vars", {})
    assert ray_constants.WORKER_SETUP_HOOK_KEY not in env_vars, (
        f"The env var, {ray_constants.WORKER_SETUP_HOOK_KEY}, "
        "is not permitted because it is reserved for the internal use."
    )
    env_vars[ray_constants.WORKER_SETUP_HOOK_KEY] = _decode_function_key(key)
    runtime_env["env_vars"] = env_vars
    # Note: This field is no-op. We don't have a plugin for the setup hook
    # because we can implement it simply using an env var.
    # This field is just for the observability purpose, so we store
    # the name of the method.
    runtime_env["worker_setup_hook"] = setup_func.__name__
    return runtime_env


def load_and_execute_setup_hook(
    worker_setup_hook_key: str,
) -> Tuple[bool, Optional[str]]:
    """Load the setup hook from a given key and execute.

    Args:
        worker_setup_hook_key: The key to import the setup hook
            from GCS.
    Returns:
        A pair of (success, error_message)
    """
    assert worker_setup_hook_key is not None
    worker = ray._private.worker.global_worker
    assert worker.connected

    func_manager = worker.function_actor_manager
    try:
        worker_setup_func_info = func_manager.fetch_registsered_method(
            _encode_function_key(worker_setup_hook_key),
            timeout=get_import_export_timeout(),
        )
    except Exception:
        error_message = (
            "Failed to import setup hook within "
            f"{get_import_export_timeout()} seconds.\n"
            f"{traceback.format_exc()}"
        )
        return False, error_message

    try:
        setup_func = pickle.loads(worker_setup_func_info.function)
    except Exception:
        error_message = (
            "Failed to deserialize the setup hook method.\n" f"{traceback.format_exc()}"
        )
        return False, error_message

    try:
        setup_func()
    except Exception:
        error_message = (
            f"Failed to execute the setup hook method. Function name:"
            f"{worker_setup_func_info.function_name}\n"
            f"{traceback.format_exc()}"
        )
        return False, error_message

    return True, None
