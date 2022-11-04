import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from typing import Union

import ray._private.ray_constants as ray_constants
import ray._private.utils as utils
import ray.dashboard.consts as dashboard_consts
import ray.dashboard.utils as dashboard_utils
from ray.core.generated import event_pb2, event_pb2_grpc
from ray.dashboard.modules.event import event_consts
from ray.dashboard.modules.event.event_utils import monitor_events
from ray.dashboard.utils import async_loop_forever, create_task

logger = logging.getLogger(__name__)


class EventAgent(dashboard_utils.DashboardAgentModule):
    def __init__(self, dashboard_agent):
        super().__init__(dashboard_agent)
        self._event_dir = os.path.join(self._dashboard_agent.log_dir, "events")
        os.makedirs(self._event_dir, exist_ok=True)
        self._monitor: Union[asyncio.Task, None] = None
        self._stub: Union[event_pb2_grpc.ReportEventServiceStub, None] = None
        self._cached_events = asyncio.Queue(event_consts.EVENT_AGENT_CACHE_SIZE)
        self._gcs_aio_client = dashboard_agent.gcs_aio_client
        self.monitor_thread_pool_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="event_monitor"
        )
        # Total number of event created from this agent.
        self.total_event_reported = 0
        # Total number of event report request sent.
        self.total_request_sent = 0
        self.module_started = time.monotonic()

        logger.info("Event agent cache buffer size: %s", self._cached_events.maxsize)

    async def _connect_to_dashboard(self):
        """Connect to the dashboard. If the dashboard is not started, then
        this method will never returns.

        Returns:
            The ReportEventServiceStub object.
        """
        while True:
            try:
                dashboard_rpc_address = await self._gcs_aio_client.internal_kv_get(
                    dashboard_consts.DASHBOARD_RPC_ADDRESS.encode(),
                    namespace=ray_constants.KV_NAMESPACE_DASHBOARD,
                    timeout=1,
                )
                dashboard_rpc_address = dashboard_rpc_address.decode()
                if dashboard_rpc_address:
                    logger.info("Report events to %s", dashboard_rpc_address)
                    options = ray_constants.GLOBAL_GRPC_OPTIONS
                    channel = utils.init_grpc_channel(
                        dashboard_rpc_address, options=options, asynchronous=True
                    )
                    return event_pb2_grpc.ReportEventServiceStub(channel)
            except Exception:
                logger.exception("Connect to dashboard failed.")
            await asyncio.sleep(
                event_consts.RETRY_CONNECT_TO_DASHBOARD_INTERVAL_SECONDS
            )

    @async_loop_forever(event_consts.EVENT_AGENT_REPORT_INTERVAL_SECONDS)
    async def report_events(self):
        """Report events from cached events queue. Reconnect to dashboard if
        report failed. Log error after retry EVENT_AGENT_RETRY_TIMES.

        This method will never returns.
        """
        data = await self._cached_events.get()
        self.total_event_reported += len(data)
        for _ in range(event_consts.EVENT_AGENT_RETRY_TIMES):
            try:
                logger.debug("Report %s events.", len(data))
                request = event_pb2.ReportEventsRequest(event_strings=data)
                await self._stub.ReportEvents(request)
                self.total_request_sent += 1
                break
            except Exception:
                logger.exception("Report event failed, reconnect to the " "dashboard.")
                self._stub = await self._connect_to_dashboard()
        else:
            data_str = str(data)
            limit = event_consts.LOG_ERROR_EVENT_STRING_LENGTH_LIMIT
            logger.error(
                "Report event failed: %s",
                data_str[:limit] + (data_str[limit:] and "..."),
            )

    @dashboard_utils.async_loop_forever(10)
    async def _periodic_state_print(self):
        if self.total_event_reported <= 0 or self.total_request_sent <= 0:
            return

        elapsed = time.monotonic() - self.module_started
        logger.info(
            f"Event module report:\n"
            f"Total events reported: {self.total_event_reported}\n"
            f"Total report request: {self.total_request_sent}\n"
            f"Average report size: {self.total_event_reported / self.total_request_sent}\n"  # noqa
            f"Average num request per second: {self.total_event_reported / elapsed}\n"
            f"Average num events per second: {self.total_request_sent / elapsed}\n"
            f"Queue size: {self._cached_events.qsize()}"
        )

    async def run(self, server):
        # Connect to dashboard.
        self._stub = await self._connect_to_dashboard()
        # Start monitor task.
        self._monitor = monitor_events(
            self._event_dir,
            lambda data: create_task(self._cached_events.put(data)),
            self.monitor_thread_pool_executor,
        )

        await asyncio.gather(
            self.report_events(),
            self._periodic_state_print(),
        )

    @staticmethod
    def is_minimal_module():
        return False
