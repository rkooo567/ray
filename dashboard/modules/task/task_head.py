import asyncio
import logging
import aiohttp.web
import ray._private.utils

from collections import defaultdict

import ray.dashboard.utils as dashboard_utils
import ray.dashboard.optional_utils as dashboard_optional_utils
from ray.dashboard.optional_utils import rest_response
from ray.core.generated import node_manager_pb2_grpc
from ray.core.generated import node_manager_pb2
from ray.dashboard.datacenter import DataSource
from ray._raylet import TaskID

logger = logging.getLogger(__name__)
routes = dashboard_optional_utils.ClassMethodRouteTable


class TaskHead(dashboard_utils.DashboardHeadModule):
    def __init__(self, dashboard_head):
        super().__init__(dashboard_head)
        self._stubs = {}
        # ActorInfoGcsService
        self._gcs_actor_info_stub = None
        DataSource.nodes.signal.append(self._update_stubs)

    async def _update_stubs(self, change):
        if change.old:
            node_id, node_info = change.old
            self._stubs.pop(node_id)
        if change.new:
            # TODO(fyrestone): Handle exceptions.
            node_id, node_info = change.new
            address = "{}:{}".format(
                node_info["nodeManagerAddress"], int(node_info["nodeManagerPort"])
            )
            options = (("grpc.enable_http_proxy", 0),)
            channel = ray._private.utils.init_grpc_channel(
                address, options, asynchronous=True
            )
            stub = node_manager_pb2_grpc.NodeManagerServiceStub(channel)
            self._stubs[node_id] = stub

    @routes.get("/api/v0/tasks")
    @dashboard_optional_utils.aiohttp_cache
    async def get_tasks(self, req) -> aiohttp.web.Response:
        # Need to wait until at least the head node stub is available.
        stub_ready = await dashboard_utils.wait_for_stub(self._stubs)
        if not stub_ready:
            rest_response(success=False, message="Node information is not available.")

        async def get_task_info(stub):
            reply = await stub.GetTasksInfo(
                node_manager_pb2.GetTasksInfoRequest(),
                timeout=10,
            )
            return reply

        replies = await asyncio.gather(
            *[get_task_info(stub) for stub in self._stubs.values()]
        )

        result = defaultdict(dict)
        for reply in replies:
            tasks = reply.task_info_entries
            for task in tasks:
                data = dashboard_utils.message_to_dict(
                    task,
                    ["task_id"],
                    including_default_value_fields=True,
                    preserving_proto_field_name=True,
                )
                result[data["task_id"]] = data

        return rest_response(success=True, message="", result=result)

    async def run(self, server):
        pass

    @staticmethod
    def is_minimal_module():
        return False
