from typing import List
from dataclasses import dataclass, fields, is_dataclass
import asyncio
import aiohttp.web

from ray.dashboard.datacenter import DataSource
from ray.dashboard.modules.log.log_grpc_client import LogsGrpcClient
from ray import ray_constants


@dataclass(init=True)
class NodeIdentifiers:
    node_id: str
    node_ip: str


@dataclass(init=True)
class FileIdentifiers:
    log_file_name: str
    actor_id: str
    pid: str


@dataclass(init=True)
class LogIdentifiers:
    file: FileIdentifiers
    node: NodeIdentifiers


@dataclass(init=True)
class LogStreamOptions:
    media_type: str
    lines: int
    interval: float


def to_schema(req: aiohttp.web.Request, Dataclass):
    """Converts a aiohttp Request into the given dataclass

    Args:
        req: the request to extract the fields from
        Dataclass: the dataclass to instantiate from the request
    """
    kwargs = {}
    for field in fields(Dataclass):
        if is_dataclass(field.type):
            # recursively resolve the dataclass
            kwargs[field.name] = to_schema(req, field.type)
        else:
            value = req.query.get(field.name)
            if value is None:
                value = req.match_info.get(field.name)
            # Note: This is brittle as the class needs to be
            # instantiable from the str
            kwargs[field.name] = field.type(value) if value else None
    return Dataclass(**kwargs)


class LogsManager:
    def __init__(self):
        self.client = LogsGrpcClient()

    def wait_until_client_initializes(func):
        async def wait_wrapper(self, *args, **kwargs):
            await self.client.wait_until_initialized()
            return await func(self, *args, **kwargs)

        return wait_wrapper

    @wait_until_client_initializes
    async def resolve_node_id(self, args: NodeIdentifiers):
        node_id = args.node_id
        if node_id is None:
            ip = args.node_ip
            if ip is not None:
                if ip not in self._ip_to_node_id:
                    raise ValueError(f"node_ip: {ip} not found")
                node_id = self._ip_to_node_id[ip]
        elif node_id not in self.client.get_all_registered_nodes():
            raise ValueError(f"node_id: {node_id} not found")
        return node_id

    @wait_until_client_initializes
    async def list_logs(self, node_id_query: str, filters: List[str]):
        """
        Helper function to list the logs by querying each agent
        on each cluster via gRPC.
        """
        response = {}
        tasks = []
        for node_id in self.client.get_all_registered_nodes():
            if node_id_query is None or node_id_query == node_id:

                async def coro():
                    reply = await self.client.list_logs(node_id)
                    response[node_id] = self._list_logs_single_node(
                        reply.log_files, filters
                    )

                tasks.append(coro())
        await asyncio.gather(*tasks)
        return response

    async def _resolve_file_and_node(
        self,
        args: LogIdentifiers
    ):
        node_id = await self.resolve_node_id(args.node)
        log_file_name = args.file.log_file_name

        # If `log_file_name` is not provided, check if we can get the
        # corresponding `log_file_name` if an `actor_id` is provided.
        if log_file_name is None:
            if args.file.actor_id is not None:
                actor_data = DataSource.actors.get(args.file.actor_id)
                if actor_data is None:
                    raise ValueError("Actor ID {actor_id} not found.")
                worker_id = actor_data["address"].get("workerId")
                if worker_id is None:
                    raise ValueError("Worker ID for Actor ID {actor_id} not found.")

                index = await self.list_logs(node_id, [worker_id])
                for node in index:
                    for file in index[node]["worker_outs"]:
                        if file.split(".")[0].split("-")[1] == worker_id:
                            log_file_name = file
                            if node_id is None:
                                node_id = node
                            break

        # If `log_file_name` cannot be resolved, check if we can get the
        # corresponding `log_file_name` if a `pid` is provided.
        if log_file_name is None:
            pid = args.file.pid
            if pid is not None:
                if node_id is None:
                    raise ValueError("Node identifiers (node_ip, node_id) not provided "
                                     f"with pid: {pid}. "
                                     f"Available: {self._ip_to_node_id}")
                index = await self.list_logs(node_id, [pid])
                for file in index[node_id]["worker_outs"]:
                    if file.split(".")[0].split("-")[3] == pid:
                        log_file_name = file
                        break
                if log_file_name is None:
                    raise ValueError(
                        f"Worker with pid {pid} not found on node {node_id}")

        # node_id and log_file_name need to be known by this point
        if node_id is None or node_id not in self.client.get_all_registered_nodes():
            raise ValueError(f"node_id {node_id} not found")
        if log_file_name is None:
            raise ValueError("Could not resolve file identifiers to a file name.")

        return log_file_name, node_id

    @wait_until_client_initializes
    async def create_log_stream(
        self,
        identifiers: LogIdentifiers,
        stream_options: LogStreamOptions,
    ):
        log_file_name, node_id = await self._resolve_file_and_node(identifiers)

        if stream_options.media_type == "stream":
            keep_alive = True
        elif stream_options.media_type == "file":
            keep_alive = False
        else:
            raise ValueError("Invalid media type: {media_type}")

        return await self.client.stream_log(
            node_id=node_id,
            log_file_name=log_file_name,
            keep_alive=keep_alive,
            lines=stream_options.lines,
            interval=stream_options.interval,
        )

    @staticmethod
    def _list_logs_single_node(log_files: List[str], filters: List[str]):
        """
        Returns a JSON file mapping a category of log component to a list of filenames,
        on the given node.
        """
        filters = [] if filters == [""] else filters

        def contains_all_filters(log_file_name):
            return all(f in log_file_name for f in filters)

        filtered = list(filter(contains_all_filters, log_files))
        logs = {}
        logs["worker_errors"] = list(
            filter(lambda s: "worker" in s and s.endswith(".err"), filtered)
        )
        logs["worker_outs"] = list(
            filter(lambda s: "worker" in s and s.endswith(".out"), filtered)
        )
        for lang in ray_constants.LANGUAGE_WORKER_TYPES:
            logs[f"{lang}_core_worker_logs"] = list(
                filter(
                    lambda s: f"{lang}-core-worker" in s and s.endswith(".log"),
                    filtered,
                )
            )
            logs[f"{lang}_driver_logs"] = list(
                filter(
                    lambda s: f"{lang}-core-driver" in s and s.endswith(".log"),
                    filtered,
                )
            )
        logs["dashboard"] = list(filter(lambda s: "dashboard" in s, filtered))
        logs["raylet"] = list(filter(lambda s: "raylet" in s, filtered))
        logs["gcs_server"] = list(filter(lambda s: "gcs" in s, filtered))
        logs["ray_client"] = list(filter(lambda s: "ray_client" in s, filtered))
        logs["autoscaler"] = list(
            filter(lambda s: "monitor" in s and "log_monitor" not in s, filtered)
        )
        logs["runtime_env"] = list(filter(lambda s: "runtime_env" in s, filtered))
        logs["folders"] = list(filter(lambda s: "." not in s, filtered))
        logs["misc"] = list(
            filter(lambda s: all([s not in logs[k] for k in logs]), filtered)
        )
        return logs
