import asyncio
from collections import defaultdict
from dataclasses import dataclass
import json
import logging
import os
import time
from typing import Dict, Set

from ray.core.generated import runtime_env_agent_pb2
from ray.core.generated import runtime_env_agent_pb2_grpc
from ray.core.generated import agent_manager_pb2
from ray.experimental.internal_kv import (_initialize_internal_kv,
                                          _internal_kv_initialized)
import ray.new_dashboard.utils as dashboard_utils
import ray.new_dashboard.modules.runtime_env.runtime_env_consts \
    as runtime_env_consts
from ray._private.ray_logging import setup_component_logger
from ray._private.runtime_env.conda import CondaManager
from ray._private.runtime_env.working_dir import WorkingDirManager
from ray._private.runtime_env import RuntimeEnvContext

logger = logging.getLogger(__name__)

# TODO(edoakes): this is used for unit tests. We should replace it with a
# better pluggability mechanism once available.
SLEEP_FOR_TESTING_S = os.environ.get("RAY_RUNTIME_ENV_SLEEP_FOR_TESTING_S")


@dataclass
class CreatedEnvResult:
    # Whether or not the env was installed correctly.
    success: bool
    # If success is True, will be a serialized RuntimeEnvContext
    # If success is False, will be an error message.
    result: str


class RuntimeEnvAgent(dashboard_utils.DashboardAgentModule,
                      runtime_env_agent_pb2_grpc.RuntimeEnvServiceServicer):
    """An RPC server to create and delete runtime envs.

    Attributes:
        dashboard_agent: The DashboardAgent object contains global config.
    """

    def __init__(self, dashboard_agent):
        super().__init__(dashboard_agent)
        self._runtime_env_dir = dashboard_agent.runtime_env_dir
        self._logging_params = dashboard_agent.logging_params
        self._per_job_logger_cache = dict()
        # Cache the results of creating envs to avoid repeatedly calling into
        # conda and other slow calls.
        self._env_cache: Dict[str, CreatedEnvResult] = dict()
        # Maps a serialized runtime env to a lock that is used
        # to prevent multiple concurrent installs of the same env.
        self._env_locks: Dict[str, asyncio.Lock] = dict()
        # Keeps track of the URIs contained within each env so we can
        # invalidate the env cache when a URI is deleted.
        # This is a temporary mechanism until we have per-URI caching.
        self._working_dir_uri_to_envs: Dict[str, Set[str]] = defaultdict(set)

        # Initialize internal KV to be used by the working_dir setup code.
        _initialize_internal_kv(self._dashboard_agent.gcs_client)
        assert _internal_kv_initialized()

        self._conda_manager = CondaManager(self._runtime_env_dir)
        self._working_dir_manager = WorkingDirManager(self._runtime_env_dir)

    def get_or_create_logger(self, job_id: bytes):
        job_id = job_id.decode()
        if job_id not in self._per_job_logger_cache:
            params = self._logging_params.copy()
            params["filename"] = f"runtime_env_setup-{job_id}.log"
            params["logger_name"] = f"runtime_env_{job_id}"
            per_job_logger = setup_component_logger(**params)
            self._per_job_logger_cache[job_id] = per_job_logger
        return self._per_job_logger_cache[job_id]

    async def CreateRuntimeEnv(self, request, context):
        async def _setup_runtime_env(serialized_runtime_env):
            # This function will be ran inside a thread
            def run_setup_with_logger():
                runtime_env: dict = json.loads(serialized_runtime_env or "{}")

                # Use a separate logger for each job.
                per_job_logger = self.get_or_create_logger(request.job_id)
                context = RuntimeEnvContext(
                    env_vars=runtime_env.get("env_vars"))
                self._conda_manager.setup(
                    runtime_env, context, logger=per_job_logger)
                self._working_dir_manager.setup(
                    runtime_env, context, logger=per_job_logger)

                # Add the mapping of URIs -> the serialized environment to be
                # used for cache invalidation.
                for uri in runtime_env.get("uris", []):
                    self._working_dir_uri_to_envs[uri].add(
                        serialized_runtime_env)

                return context

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, run_setup_with_logger)

        serialized_env = request.serialized_runtime_env

        if serialized_env not in self._env_locks:
            # async lock to prevent the same env being concurrently installed
            self._env_locks[serialized_env] = asyncio.Lock()

        async with self._env_locks[serialized_env]:
            if serialized_env in self._env_cache:
                serialized_context = self._env_cache[serialized_env]
                result = self._env_cache[serialized_env]
                if result.success:
                    context = result.result
                    logger.info("Runtime env already created successfully. "
                                f"Env: {serialized_env}, context: {context}")
                    return runtime_env_agent_pb2.CreateRuntimeEnvReply(
                        status=agent_manager_pb2.AGENT_RPC_STATUS_OK,
                        serialized_runtime_env_context=context)
                else:
                    error_message = result.result
                    logger.info("Runtime env already failed. "
                                f"Env: {serialized_env}, err: {error_message}")
                    return runtime_env_agent_pb2.CreateRuntimeEnvReply(
                        status=agent_manager_pb2.AGENT_RPC_STATUS_FAILED,
                        error_message=error_message)

            if SLEEP_FOR_TESTING_S:
                logger.info(f"Sleeping for {SLEEP_FOR_TESTING_S}s.")
                time.sleep(int(SLEEP_FOR_TESTING_S))

            logger.info(f"Creating runtime env: {serialized_env}")
            runtime_env_context: RuntimeEnvContext = None
            error_message = None
            for _ in range(runtime_env_consts.RUNTIME_ENV_RETRY_TIMES):
                try:
                    runtime_env_context = await _setup_runtime_env(
                        serialized_env)
                    break
                except Exception as ex:
                    logger.exception("Runtime env creation failed.")
                    error_message = str(ex)
                    await asyncio.sleep(
                        runtime_env_consts.RUNTIME_ENV_RETRY_INTERVAL_MS / 1000
                    )
            if error_message:
                logger.error(
                    "Runtime env creation failed for %d times, "
                    "don't retry any more.",
                    runtime_env_consts.RUNTIME_ENV_RETRY_TIMES)
                self._env_cache[serialized_env] = CreatedEnvResult(
                    False, error_message)
                return runtime_env_agent_pb2.CreateRuntimeEnvReply(
                    status=agent_manager_pb2.AGENT_RPC_STATUS_FAILED,
                    error_message=error_message)

            serialized_context = runtime_env_context.serialize()
            self._env_cache[serialized_env] = CreatedEnvResult(
                True, serialized_context)
            logger.info(
                "Successfully created runtime env: %s, the context: %s",
                serialized_env, serialized_context)
            return runtime_env_agent_pb2.CreateRuntimeEnvReply(
                status=agent_manager_pb2.AGENT_RPC_STATUS_OK,
                serialized_runtime_env_context=serialized_context)

    async def DeleteURIs(self, request, context):
        logger.info(f"Got request to delete URIS: {request.uris}.")

        # Only a single URI is currently supported.
        assert len(request.uris) == 1

        uri = request.uris[0]

        # Invalidate the env cache for any environments that contain this URI.
        for env in self._working_dir_uri_to_envs.get(uri, []):
            if env in self._env_cache:
                del self._env_cache[env]

        if self._working_dir_manager.delete_uri(uri):
            return runtime_env_agent_pb2.DeleteURIsReply(
                status=agent_manager_pb2.AGENT_RPC_STATUS_OK)
        else:
            return runtime_env_agent_pb2.DeleteURIsReply(
                status=agent_manager_pb2.AGENT_RPC_STATUS_FAILED,
                error_message=f"Local file for URI {uri} not found.")

    async def run(self, server):
        runtime_env_agent_pb2_grpc.add_RuntimeEnvServiceServicer_to_server(
            self, server)
