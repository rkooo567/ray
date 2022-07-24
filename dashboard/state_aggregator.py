import asyncio
import logging

from dataclasses import asdict, fields
from itertools import islice
from typing import List, Tuple

import ray.dashboard.memory_utils as memory_utils
import ray.dashboard.utils as dashboard_utils
from ray._private.utils import binary_to_hex
from ray.core.generated.common_pb2 import TaskStatus
from ray.experimental.state.custom_types import is_literal
from ray.experimental.state.common import (
    ActorState,
    ListApiOptions,
    ListApiResponse,
    NodeState,
    ObjectState,
    PlacementGroupState,
    RuntimeEnvState,
    SummaryApiResponse,
    MAX_LIMIT,
    SummaryApiOptions,
    TaskSummaries,
    StateSchema,
    SupportedFilterType,
    TaskState,
    WorkerState,
    StateSummary,
    ActorSummaries,
    ObjectSummaries,
    filter_fields,
    PredicateType,
)
from ray.experimental.state.state_manager import (
    DataSourceUnavailable,
    StateDataSourceClient,
)
from ray.runtime_env import RuntimeEnv
from ray.experimental.state.util import convert_string_to_type

logger = logging.getLogger(__name__)

GCS_QUERY_FAILURE_WARNING = (
    "Failed to query data from GCS. It is due to "
    "(1) GCS is unexpectedly failed. "
    "(2) GCS is overloaded. "
    "(3) There's an unexpected network issue. "
    "Please check the gcs_server.out log to find the root cause."
)
NODE_QUERY_FAILURE_WARNING = (
    "Failed to query data from {type}. "
    "Queryed {total} {type} "
    "and {network_failures} {type} failed to reply. It is due to "
    "(1) {type} is unexpectedly failed. "
    "(2) {type} is overloaded. "
    "(3) There's an unexpected network issue. Please check the "
    "{log_command} to find the root cause."
)


def _convert_filters_type(
    filter: List[Tuple[str, PredicateType, SupportedFilterType]],
    schema: StateSchema,
) -> List[Tuple[str, SupportedFilterType]]:
    """Convert the given filter's type to SupportedFilterType.

    This method is necessary because click can only accept a single type
    for its tuple (which is string in this case).

    Args:
        filter: A list of filter which is a tuple of (key, val).
        schema: The state schema. It is used to infer the type of the column for filter.

    Returns:
        A new list of filters with correctly types that match the schema.
    """
    new_filter = []
    schema = {field.name: field.type for field in fields(schema)}

    for col, predicate, val in filter:
        if col in schema:
            column_type = schema[col]
            if is_literal(column_type):
                pass
            elif isinstance(val, column_type):
                # Do nothing.
                pass
            elif column_type is int:
                try:
                    val = convert_string_to_type(val, int)
                except ValueError:
                    raise ValueError(
                        f"Invalid filter `--filter {col} {val}` for a int type "
                        "column. Please provide an integer filter "
                        f"`--filter {col} [int]`"
                    )
            elif column_type is float:
                try:
                    val = convert_string_to_type(val, float)
                except ValueError:
                    raise ValueError(
                        f"Invalid filter `--filter {col} {val}` for a float "
                        "type column. Please provide an integer filter "
                        f"`--filter {col} [float]`"
                    )
            elif column_type is bool:
                try:
                    val = convert_string_to_type(val, bool)
                except ValueError:
                    raise ValueError(
                        f"Invalid filter `--filter {col} {val}` for a boolean "
                        "type column. Please provide "
                        f"`--filter {col} [True|true|1]` for True or "
                        f"`--filter {col} [False|false|0]` for False."
                    )
        new_filter.append((col, predicate, val))
    return new_filter


# TODO(sang): Move the class to state/state_manager.py.
# TODO(sang): Remove *State and replaces with Pydantic or protobuf.
# (depending on API interface standardization).
class StateAPIManager:
    """A class to query states from data source, caches, and post-processes
    the entries.
    """

    def __init__(self, state_data_source_client: StateDataSourceClient):
        self._client = state_data_source_client

    @property
    def data_source_client(self):
        return self._client

    def _filter(
        self,
        data: List[dict],
        filters: List[Tuple[str, SupportedFilterType]],
        state_dataclass: StateSchema,
        detail: bool,
    ) -> List[dict]:
        """Return the filtered data given filters.

        Args:
            data: A list of state data.
            filters: A list of KV tuple to filter data (key, val). The data is filtered
                if data[key] != val.
            state_dataclass: The state schema.

        Returns:
            A list of filtered state data in dictionary. Each state data's
            unnecessary columns are filtered by the given state_dataclass schema.
        """
        filters = _convert_filters_type(filters, state_dataclass)
        result = []
        for datum in data:
            match = True
            for filter_column, filter_predicate, filter_value in filters:
                filterable_columns = state_dataclass.filterable_columns()
                filter_column = filter_column.lower()
                if filter_column not in filterable_columns:
                    raise ValueError(
                        f"The given filter column {filter_column} is not supported. "
                        f"Supported filter columns: {filterable_columns}"
                    )

                if filter_predicate == "=":
                    match = datum[filter_column] == filter_value
                elif filter_predicate == "!=":
                    match = datum[filter_column] != filter_value
                else:
                    raise ValueError(
                        f"Unsupported filter predicate {filter_predicate} is given. "
                        "Available predicates: =, !=."
                    )

                if not match:
                    break

            if match:
                result.append(filter_fields(datum, state_dataclass, detail))
        return result

    async def list_actors(self, *, option: ListApiOptions) -> ListApiResponse:
        """List all actor information from the cluster.

        Returns:
            {actor_id -> actor_data_in_dict}
            actor_data_in_dict's schema is in ActorState

        """
        try:
            reply = await self._client.get_all_actor_info(timeout=option.timeout)
        except DataSourceUnavailable:
            raise DataSourceUnavailable(GCS_QUERY_FAILURE_WARNING)

        result = []
        for message in reply.actor_table_data:
            data = self._message_to_dict(message=message, fields_to_decode=["actor_id"])
            result.append(data)

        result = self._filter(result, option.filters, ActorState, option.detail)
        # Sort to make the output deterministic.
        result.sort(key=lambda entry: entry["actor_id"])
        result = list(islice(result, option.limit))
        return ListApiResponse(
            result=result,
            total=reply.total,
        )

    async def list_placement_groups(self, *, option: ListApiOptions) -> ListApiResponse:
        """List all placement group information from the cluster.

        Returns:
            {pg_id -> pg_data_in_dict}
            pg_data_in_dict's schema is in PlacementGroupState
        """
        try:
            reply = await self._client.get_all_placement_group_info(
                timeout=option.timeout
            )
        except DataSourceUnavailable:
            raise DataSourceUnavailable(GCS_QUERY_FAILURE_WARNING)

        result = []
        for message in reply.placement_group_table_data:

            data = self._message_to_dict(
                message=message,
                fields_to_decode=["placement_group_id", "node_id"],
            )
            result.append(data)

        result = self._filter(
            result, option.filters, PlacementGroupState, option.detail
        )
        # Sort to make the output deterministic.
        result.sort(key=lambda entry: entry["placement_group_id"])
        return ListApiResponse(
            result=list(islice(result, option.limit)),
            total=reply.total,
        )

    async def list_nodes(self, *, option: ListApiOptions) -> ListApiResponse:
        """List all node information from the cluster.

        Returns:
            {node_id -> node_data_in_dict}
            node_data_in_dict's schema is in NodeState
        """
        try:
            reply = await self._client.get_all_node_info(timeout=option.timeout)
        except DataSourceUnavailable:
            raise DataSourceUnavailable(GCS_QUERY_FAILURE_WARNING)

        result = []
        for message in reply.node_info_list:
            data = self._message_to_dict(message=message, fields_to_decode=["node_id"])
            data["node_ip"] = data["node_manager_address"]
            result.append(data)

        result = self._filter(result, option.filters, NodeState, option.detail)
        # Sort to make the output deterministic.
        result.sort(key=lambda entry: entry["node_id"])
        total_nodes = len(result)
        result = list(islice(result, option.limit))
        return ListApiResponse(
            result=result,
            # No reason to truncate node because they are usually small.
            total=total_nodes,
        )

    async def list_workers(self, *, option: ListApiOptions) -> ListApiResponse:
        """List all worker information from the cluster.

        Returns:
            {worker_id -> worker_data_in_dict}
            worker_data_in_dict's schema is in WorkerState
        """
        try:
            reply = await self._client.get_all_worker_info(timeout=option.timeout)
        except DataSourceUnavailable:
            raise DataSourceUnavailable(GCS_QUERY_FAILURE_WARNING)

        result = []
        for message in reply.worker_table_data:
            data = self._message_to_dict(
                message=message, fields_to_decode=["worker_id", "raylet_id"]
            )
            data["worker_id"] = data["worker_address"]["worker_id"]
            data["node_id"] = data["worker_address"]["raylet_id"]
            data["ip"] = data["worker_address"]["ip_address"]
            result.append(data)

        result = self._filter(result, option.filters, WorkerState, option.detail)
        # Sort to make the output deterministic.
        result.sort(key=lambda entry: entry["worker_id"])
        result = list(islice(result, option.limit))
        return ListApiResponse(
            result=result,
            total=reply.total,
        )

    def list_jobs(self, *, option: ListApiOptions) -> ListApiResponse:
        # TODO(sang): Support limit & timeout & async calls.
        try:
            result = []
            job_info = self._client.get_job_info()
            for job_id, data in job_info.items():
                data = asdict(data)
                data["job_id"] = job_id
                result.append(data)
        except DataSourceUnavailable:
            raise DataSourceUnavailable(GCS_QUERY_FAILURE_WARNING)
        return ListApiResponse(
            result=result,
            # TODO(sang): Support this.
            total=len(result),
        )

    async def list_tasks(self, *, option: ListApiOptions) -> ListApiResponse:
        """List all task information from the cluster.

        Returns:
            {task_id -> task_data_in_dict}
            task_data_in_dict's schema is in TaskState
        """
        raylet_ids = self._client.get_all_registered_raylet_ids()
        replies = await asyncio.gather(
            *[
                self._client.get_task_info(node_id, timeout=option.timeout)
                for node_id in raylet_ids
            ],
            return_exceptions=True,
        )

        unresponsive_nodes = 0
        running_task_id = set()
        successful_replies = []
        total_tasks = 0
        for reply in replies:
            if isinstance(reply, DataSourceUnavailable):
                unresponsive_nodes += 1
                continue
            elif isinstance(reply, Exception):
                raise reply

            successful_replies.append(reply)
            total_tasks += reply.total
            for task_id in reply.running_task_ids:
                running_task_id.add(binary_to_hex(task_id))

        partial_failure_warning = None
        if len(raylet_ids) > 0 and unresponsive_nodes > 0:
            warning_msg = NODE_QUERY_FAILURE_WARNING.format(
                type="raylet",
                total=len(raylet_ids),
                network_failures=unresponsive_nodes,
                log_command="raylet.out",
            )
            if unresponsive_nodes == len(raylet_ids):
                raise DataSourceUnavailable(warning_msg)
            partial_failure_warning = (
                f"The returned data may contain incomplete result. {warning_msg}"
            )

        result = []
        for reply in successful_replies:
            assert not isinstance(reply, Exception)
            tasks = reply.owned_task_info_entries
            for task in tasks:
                data = self._message_to_dict(
                    message=task,
                    fields_to_decode=["task_id"],
                )
                if data["task_id"] in running_task_id:
                    data["scheduling_state"] = TaskStatus.DESCRIPTOR.values_by_number[
                        TaskStatus.RUNNING
                    ].name
                result.append(data)

        result = self._filter(result, option.filters, TaskState, option.detail)
        # Sort to make the output deterministic.
        result.sort(key=lambda entry: entry["task_id"])
        result = list(islice(result, option.limit))
        return ListApiResponse(
            result=result,
            partial_failure_warning=partial_failure_warning,
            total=total_tasks,
        )

    async def list_objects(self, *, option: ListApiOptions) -> ListApiResponse:
        """List all object information from the cluster.

        Returns:
            {object_id -> object_data_in_dict}
            object_data_in_dict's schema is in ObjectState
        """
        raylet_ids = self._client.get_all_registered_raylet_ids()
        replies = await asyncio.gather(
            *[
                self._client.get_object_info(node_id, timeout=option.timeout)
                for node_id in raylet_ids
            ],
            return_exceptions=True,
        )

        unresponsive_nodes = 0
        worker_stats = []
        total_objects = 0
        for reply, _ in zip(replies, raylet_ids):
            if isinstance(reply, DataSourceUnavailable):
                unresponsive_nodes += 1
                continue
            elif isinstance(reply, Exception):
                raise reply

            total_objects += reply.total
            for core_worker_stat in reply.core_workers_stats:
                # NOTE: Set preserving_proto_field_name=False here because
                # `construct_memory_table` requires a dictionary that has
                # modified protobuf name
                # (e.g., workerId instead of worker_id) as a key.
                worker_stats.append(
                    self._message_to_dict(
                        message=core_worker_stat,
                        fields_to_decode=["object_id"],
                        preserving_proto_field_name=False,
                    )
                )

        partial_failure_warning = None
        if len(raylet_ids) > 0 and unresponsive_nodes > 0:
            warning_msg = NODE_QUERY_FAILURE_WARNING.format(
                type="raylet",
                total=len(raylet_ids),
                network_failures=unresponsive_nodes,
                log_command="raylet.out",
            )
            if unresponsive_nodes == len(raylet_ids):
                raise DataSourceUnavailable(warning_msg)
            partial_failure_warning = (
                f"The returned data may contain incomplete result. {warning_msg}"
            )

        result = []
        memory_table = memory_utils.construct_memory_table(worker_stats)
        for entry in memory_table.table:
            data = entry.as_dict()
            # `construct_memory_table` returns object_ref field which is indeed
            # object_id. We do transformation here.
            # TODO(sang): Refactor `construct_memory_table`.
            data["object_id"] = data["object_ref"]
            del data["object_ref"]
            data["ip"] = data["node_ip_address"]
            del data["node_ip_address"]
            result.append(data)

        result = self._filter(result, option.filters, ObjectState, option.detail)
        # Sort to make the output deterministic.
        result.sort(key=lambda entry: entry["object_id"])
        result = list(islice(result, option.limit))
        return ListApiResponse(
            result=result,
            partial_failure_warning=partial_failure_warning,
            total=total_objects,
        )

    async def list_runtime_envs(self, *, option: ListApiOptions) -> ListApiResponse:
        """List all runtime env information from the cluster.

        Returns:
            A list of runtime env information in the cluster.
            The schema of returned "dict" is equivalent to the
            `RuntimeEnvState` protobuf message.
            We don't have id -> data mapping like other API because runtime env
            doesn't have unique ids.
        """
        agent_ids = self._client.get_all_registered_agent_ids()
        replies = await asyncio.gather(
            *[
                self._client.get_runtime_envs_info(node_id, timeout=option.timeout)
                for node_id in agent_ids
            ],
            return_exceptions=True,
        )

        result = []
        unresponsive_nodes = 0
        total_runtime_envs = 0
        for node_id, reply in zip(self._client.get_all_registered_agent_ids(), replies):
            if isinstance(reply, DataSourceUnavailable):
                unresponsive_nodes += 1
                continue
            elif isinstance(reply, Exception):
                raise reply

            total_runtime_envs += reply.total
            states = reply.runtime_env_states
            for state in states:
                data = self._message_to_dict(message=state, fields_to_decode=[])
                # Need to deseiralize this field.
                data["runtime_env"] = RuntimeEnv.deserialize(
                    data["runtime_env"]
                ).to_dict()
                data["node_id"] = node_id
                result.append(data)

        partial_failure_warning = None
        if len(agent_ids) > 0 and unresponsive_nodes > 0:
            warning_msg = NODE_QUERY_FAILURE_WARNING.format(
                type="agent",
                total=len(agent_ids),
                network_failures=unresponsive_nodes,
                log_command="dashboard_agent.log",
            )
            if unresponsive_nodes == len(agent_ids):
                raise DataSourceUnavailable(warning_msg)
            partial_failure_warning = (
                f"The returned data may contain incomplete result. {warning_msg}"
            )

        result = self._filter(result, option.filters, RuntimeEnvState, option.detail)

        # Sort to make the output deterministic.
        def sort_func(entry):
            # If creation time is not there yet (runtime env is failed
            # to be created or not created yet, they are the highest priority.
            # Otherwise, "bigger" creation time is coming first.
            if "creation_time_ms" not in entry:
                return float("inf")
            elif entry["creation_time_ms"] is None:
                return float("inf")
            else:
                return float(entry["creation_time_ms"])

        result.sort(key=sort_func, reverse=True)
        result = list(islice(result, option.limit))
        return ListApiResponse(
            result=result,
            partial_failure_warning=partial_failure_warning,
            total=total_runtime_envs,
        )

    async def summarize_tasks(self, option: SummaryApiOptions) -> SummaryApiResponse:
        # For summary, try getting as many entries as possible to minimze data loss.
        result = await self.list_tasks(
            option=ListApiOptions(timeout=option.timeout, limit=MAX_LIMIT, filters=[])
        )
        summary = StateSummary(
            node_id_to_summary={
                "cluster": TaskSummaries.to_summary(tasks=result.result)
            }
        )
        return SummaryApiResponse(
            result=summary, partial_failure_warning=result.partial_failure_warning
        )

    async def summarize_actors(self, option: SummaryApiOptions) -> SummaryApiResponse:
        # For summary, try getting as many entries as possible to minimze data loss.
        result = await self.list_actors(
            option=ListApiOptions(timeout=option.timeout, limit=MAX_LIMIT, filters=[])
        )
        summary = StateSummary(
            node_id_to_summary={
                "cluster": ActorSummaries.to_summary(actors=result.result)
            }
        )
        return SummaryApiResponse(
            result=summary, partial_failure_warning=result.partial_failure_warning
        )

    async def summarize_objects(self, option: SummaryApiOptions) -> SummaryApiResponse:
        # For summary, try getting as many entries as possible to minimze data loss.
        result = await self.list_objects(
            option=ListApiOptions(timeout=option.timeout, limit=MAX_LIMIT, filters=[])
        )
        summary = StateSummary(
            node_id_to_summary={
                "cluster": ObjectSummaries.to_summary(objects=result.result)
            }
        )
        return SummaryApiResponse(
            result=summary, partial_failure_warning=result.partial_failure_warning
        )

    def _message_to_dict(
        self,
        *,
        message,
        fields_to_decode: List[str],
        preserving_proto_field_name: bool = True,
    ) -> dict:
        return dashboard_utils.message_to_dict(
            message,
            fields_to_decode,
            including_default_value_fields=True,
            preserving_proto_field_name=preserving_proto_field_name,
        )
