import json
import sys
from dataclasses import fields
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest
import yaml
from click.testing import CliRunner

import ray
import ray.dashboard.consts as dashboard_consts
import ray._private.state as global_state
import ray._private.ray_constants as ray_constants
from ray._private.test_utils import wait_for_condition
from ray.cluster_utils import cluster_not_supported
from ray.core.generated.common_pb2 import (
    Address,
    CoreWorkerStats,
    ObjectRefInfo,
    TaskInfoEntry,
    TaskStatus,
    WorkerType,
    TaskType,
)
from ray.core.generated.gcs_pb2 import (
    ActorTableData,
    GcsNodeInfo,
    PlacementGroupTableData,
    WorkerTableData,
)
from ray.core.generated.gcs_service_pb2 import (
    GetAllActorInfoReply,
    GetAllNodeInfoReply,
    GetAllPlacementGroupReply,
    GetAllWorkerInfoReply,
)
from ray.core.generated.node_manager_pb2 import GetNodeStatsReply, GetTasksInfoReply
from ray.core.generated.reporter_pb2 import ListLogsReply, StreamLogReply
from ray.core.generated.runtime_env_agent_pb2 import GetRuntimeEnvsInfoReply
from ray.core.generated.runtime_env_common_pb2 import (
    RuntimeEnvState as RuntimeEnvStateProto,
)
from ray.dashboard.state_aggregator import (
    GCS_QUERY_FAILURE_WARNING,
    NODE_QUERY_FAILURE_WARNING,
    StateAPIManager,
    _convert_filters_type,
)
from ray.experimental.state.api import (
    get_actor,
    get_node,
    get_objects,
    get_placement_group,
    get_task,
    get_worker,
    list_actors,
    list_jobs,
    list_nodes,
    list_objects,
    list_placement_groups,
    list_runtime_envs,
    list_tasks,
    list_workers,
)
from ray.experimental.state.common import (
    DEFAULT_LIMIT,
    DEFAULT_RPC_TIMEOUT,
    ActorState,
    ListApiOptions,
    NodeState,
    ObjectState,
    PlacementGroupState,
    RuntimeEnvState,
    SupportedFilterType,
    TaskState,
    WorkerState,
)
from ray.experimental.state.exception import DataSourceUnavailable, RayStateApiException
from ray.experimental.state.state_cli import AvailableFormat, format_list_api_output, _parse_filter
from ray.experimental.state.state_cli import get as cli_get
from ray.experimental.state.state_cli import list as cli_list
from ray.experimental.state.state_manager import IdToIpMap, StateDataSourceClient
from ray.job_submission import JobSubmissionClient
from ray.runtime_env import RuntimeEnv

if sys.version_info > (3, 7, 0):
    from unittest.mock import AsyncMock
else:
    from asyncmock import AsyncMock


"""
Unit tests
"""


@pytest.fixture
def state_api_manager():
    data_source_client = AsyncMock(StateDataSourceClient)
    manager = StateAPIManager(data_source_client)
    yield manager


def verify_schema(state, result_dict: dict):
    state_fields_columns = set()
    for field in fields(state):
        state_fields_columns.add(field.name)

    for k in result_dict.keys():
        assert k in state_fields_columns


def generate_actor_data(id, state=ActorTableData.ActorState.ALIVE, class_name="class"):
    return ActorTableData(
        actor_id=id,
        state=state,
        name="abc",
        pid=1234,
        class_name=class_name,
    )


def generate_pg_data(id):
    return PlacementGroupTableData(
        placement_group_id=id,
        state=PlacementGroupTableData.PlacementGroupState.CREATED,
        name="abc",
        creator_job_dead=True,
        creator_actor_dead=False,
    )


def generate_node_data(id):
    return GcsNodeInfo(
        node_id=id,
        state=GcsNodeInfo.GcsNodeState.ALIVE,
        node_manager_address="127.0.0.1",
        raylet_socket_name="abcd",
        object_store_socket_name="False",
    )


def generate_worker_data(id, pid=1234):
    return WorkerTableData(
        worker_address=Address(
            raylet_id=id, ip_address="127.0.0.1", port=124, worker_id=id
        ),
        is_alive=True,
        timestamp=1234,
        worker_type=WorkerType.WORKER,
        pid=pid,
    )


def generate_task_entry(
    id,
    name="class",
    func_or_class="class",
    state=TaskStatus.SCHEDULED,
    type=TaskType.NORMAL_TASK,
):
    return TaskInfoEntry(
        task_id=id,
        name=name,
        func_or_class_name=func_or_class,
        scheduling_state=state,
        type=type,
    )


def generate_task_data(
    id, name="class", func_or_class="class", state=TaskStatus.SCHEDULED
):
    return GetTasksInfoReply(
        owned_task_info_entries=[
            generate_task_entry(
                id=id, name=name, func_or_class=func_or_class, state=state
            )
        ]
    )


def generate_object_info(
    obj_id,
    size_bytes=1,
    callsite="main.py",
    task_state=TaskStatus.SCHEDULED,
    local_ref_count=1,
    attempt_number=1,
    pid=1234,
    ip="1234",
    worker_type=WorkerType.DRIVER,
    pinned_in_memory=True,
):
    return CoreWorkerStats(
        pid=pid,
        worker_type=worker_type,
        ip_address=ip,
        object_refs=[
            ObjectRefInfo(
                object_id=obj_id,
                call_site=callsite,
                object_size=size_bytes,
                local_ref_count=local_ref_count,
                submitted_task_ref_count=1,
                contained_in_owned=[],
                pinned_in_memory=pinned_in_memory,
                task_status=task_state,
                attempt_number=attempt_number,
            )
        ],
    )


def generate_runtime_env_info(runtime_env, creation_time=None, success=True):
    return GetRuntimeEnvsInfoReply(
        runtime_env_states=[
            RuntimeEnvStateProto(
                runtime_env=runtime_env.serialize(),
                ref_cnt=1,
                success=success,
                error=None,
                creation_time_ms=creation_time,
            )
        ]
    )


def create_api_options(
    timeout: int = DEFAULT_RPC_TIMEOUT,
    limit: int = DEFAULT_LIMIT,
    filters: List[Tuple[str, SupportedFilterType]] = None,
):
    if not filters:
        filters = []
    return ListApiOptions(
        limit=limit, timeout=timeout, filters=filters, _server_timeout_multiplier=1.0
    )


def test_parse_filter():
    # Basic
    assert _parse_filter("key=value") == ("key", "=", "value")
    assert _parse_filter("key!=value") == ("key", "!=", "value")

    # Predicate =
    assert _parse_filter("key=value=123=1") == ("key", "=", "value=123=1")
    assert _parse_filter("key=value!=123!=1") == ("key", "=", "value!=123!=1")
    assert _parse_filter("key=value!=123=1") == ("key", "=", "value!=123=1")
    assert _parse_filter("key=value!=123=1!") == ("key", "=", "value!=123=1!")
    assert _parse_filter("key=value!=123=1=") == ("key", "=", "value!=123=1=")
    assert _parse_filter("key=value!=123=1!=") == ("key", "=", "value!=123=1!=")

    # Predicate !=
    assert _parse_filter("key!=value=123=1") == ("key", "!=", "value=123=1")
    assert _parse_filter("key!=value!=123!=1") == ("key", "!=", "value!=123!=1")
    assert _parse_filter("key!=value!=123=1") == ("key", "!=", "value!=123=1")
    assert _parse_filter("key!=value!=123=1!") == ("key", "!=", "value!=123=1!")
    assert _parse_filter("key!=value!=123=1=") == ("key", "!=", "value!=123=1=")
    assert _parse_filter("key!=value!=123=1!=") == ("key", "!=", "value!=123=1!=")

    # Incorrect cases
    with pytest.raises(ValueError):
        _parse_filter("keyvalue")

    with pytest.raises(ValueError):
        _parse_filter("keyvalue!")
    with pytest.raises(ValueError):
        _parse_filter("keyvalue!=")
    with pytest.raises(ValueError):
        _parse_filter("keyvalue=")

    with pytest.raises(ValueError):
        _parse_filter("!keyvalue")
    with pytest.raises(ValueError):
        _parse_filter("!=keyvalue")
    with pytest.raises(ValueError):
        _parse_filter("=keyvalue")

    with pytest.raises(ValueError):
        _parse_filter("=keyvalue=")
    with pytest.raises(ValueError):
        _parse_filter("!=keyvalue=")
    with pytest.raises(ValueError):
        _parse_filter("=keyvalue!=")
    with pytest.raises(ValueError):
        _parse_filter("!=keyvalue!=")

    with pytest.raises(ValueError):
        _parse_filter("key>value")
    with pytest.raises(ValueError):
        _parse_filter("key>value!=")


def test_id_to_ip_map():
    node_id_1 = "1"
    node_ip_1 = "ip_1"
    node_id_2 = "2"
    node_ip_2 = "ip_2"
    m = IdToIpMap()
    m.put(node_id_1, node_ip_1)
    assert m.get_ip(node_ip_2) is None
    assert m.get_node_id(node_id_2) is None
    assert m.get_ip(node_id_1) == node_ip_1
    assert m.get_node_id(node_ip_1) == node_id_1
    m.pop(node_id_1)
    assert m.get_ip(node_id_1) is None
    assert m.get_node_id(node_id_1) is None


@pytest.mark.asyncio
async def test_api_manager_list_actors(state_api_manager):
    data_source_client = state_api_manager.data_source_client
    actor_id = b"1234"
    data_source_client.get_all_actor_info.return_value = GetAllActorInfoReply(
        actor_table_data=[
            generate_actor_data(actor_id),
            generate_actor_data(b"12345", state=ActorTableData.ActorState.DEAD),
        ]
    )
    result = await state_api_manager.list_actors(option=create_api_options())
    data = result.result
    actor_data = data[0]
    verify_schema(ActorState, actor_data)

    """
    Test limit
    """
    assert len(data) == 2
    result = await state_api_manager.list_actors(option=create_api_options(limit=1))
    data = result.result
    assert len(data) == 1

    """
    Test filters
    """
    # If the column is not supported for filtering, it should raise an exception.
    with pytest.raises(ValueError):
        result = await state_api_manager.list_actors(
            option=create_api_options(filters=[("stat", "=", "DEAD")])
        )
    result = await state_api_manager.list_actors(
        option=create_api_options(filters=[("state", "=", "DEAD")])
    )
    assert len(result.result) == 1

    """
    Test error handling
    """
    data_source_client.get_all_actor_info.side_effect = DataSourceUnavailable()
    with pytest.raises(DataSourceUnavailable) as exc_info:
        result = await state_api_manager.list_actors(option=create_api_options(limit=1))
    assert exc_info.value.args[0] == GCS_QUERY_FAILURE_WARNING


@pytest.mark.asyncio
async def test_api_manager_list_pgs(state_api_manager):
    data_source_client = state_api_manager.data_source_client
    id = b"1234"
    data_source_client.get_all_placement_group_info.return_value = (
        GetAllPlacementGroupReply(
            placement_group_table_data=[
                generate_pg_data(id),
                generate_pg_data(b"12345"),
            ]
        )
    )
    result = await state_api_manager.list_placement_groups(option=create_api_options())
    data = result.result
    data = data[0]
    verify_schema(PlacementGroupState, data)

    """
    Test limit
    """
    assert len(result.result) == 2
    result = await state_api_manager.list_placement_groups(
        option=create_api_options(limit=1)
    )
    data = result.result
    assert len(data) == 1

    """
    Test filters
    """
    # If the column is not supported for filtering, it should raise an exception.
    with pytest.raises(ValueError):
        result = await state_api_manager.list_placement_groups(
            option=create_api_options(filters=[("stat", "=", "DEAD")])
        )
    result = await state_api_manager.list_placement_groups(
        option=create_api_options(
            filters=[("placement_group_id", "=", bytearray(id).hex())]
        )
    )
    assert len(result.result) == 1

    """
    Test error handling
    """
    data_source_client.get_all_placement_group_info.side_effect = (
        DataSourceUnavailable()
    )
    with pytest.raises(DataSourceUnavailable) as exc_info:
        result = await state_api_manager.list_placement_groups(
            option=create_api_options(limit=1)
        )
    assert exc_info.value.args[0] == GCS_QUERY_FAILURE_WARNING


@pytest.mark.asyncio
async def test_api_manager_list_nodes(state_api_manager):
    data_source_client = state_api_manager.data_source_client
    id = b"1234"
    data_source_client.get_all_node_info.return_value = GetAllNodeInfoReply(
        node_info_list=[generate_node_data(id), generate_node_data(b"12345")]
    )
    result = await state_api_manager.list_nodes(option=create_api_options())
    data = result.result
    data = data[0]
    verify_schema(NodeState, data)

    """
    Test limit
    """
    assert len(result.result) == 2
    result = await state_api_manager.list_nodes(option=create_api_options(limit=1))
    data = result.result
    assert len(data) == 1

    """
    Test filters
    """
    # If the column is not supported for filtering, it should raise an exception.
    with pytest.raises(ValueError):
        result = await state_api_manager.list_nodes(
            option=create_api_options(filters=[("stat", "=", "DEAD")])
        )
    result = await state_api_manager.list_nodes(
        option=create_api_options(filters=[("node_id", "=", bytearray(id).hex())])
    )
    assert len(result.result) == 1

    """
    Test error handling
    """
    data_source_client.get_all_node_info.side_effect = DataSourceUnavailable()
    with pytest.raises(DataSourceUnavailable) as exc_info:
        result = await state_api_manager.list_nodes(option=create_api_options(limit=1))
    assert exc_info.value.args[0] == GCS_QUERY_FAILURE_WARNING


@pytest.mark.asyncio
async def test_api_manager_list_workers(state_api_manager):
    data_source_client = state_api_manager.data_source_client
    id = b"1234"
    data_source_client.get_all_worker_info.return_value = GetAllWorkerInfoReply(
        worker_table_data=[
            generate_worker_data(id, pid=1),
            generate_worker_data(b"12345", pid=2),
        ]
    )
    result = await state_api_manager.list_workers(option=create_api_options())
    data = result.result
    data = data[0]
    verify_schema(WorkerState, data)

    """
    Test limit
    """
    assert len(result.result) == 2
    result = await state_api_manager.list_workers(option=create_api_options(limit=1))
    data = result.result
    assert len(data) == 1

    """
    Test filters
    """
    # If the column is not supported for filtering, it should raise an exception.
    with pytest.raises(ValueError):
        result = await state_api_manager.list_workers(
            option=create_api_options(filters=[("stat", "=", "DEAD")])
        )
    result = await state_api_manager.list_workers(
        option=create_api_options(filters=[("worker_id", "=", bytearray(id).hex())])
    )
    assert len(result.result) == 1
    # Make sure it works with int type.
    result = await state_api_manager.list_workers(
        option=create_api_options(filters=[("pid", "=", 2)])
    )
    assert len(result.result) == 1

    """
    Test error handling
    """
    data_source_client.get_all_worker_info.side_effect = DataSourceUnavailable()
    with pytest.raises(DataSourceUnavailable) as exc_info:
        result = await state_api_manager.list_workers(
            option=create_api_options(limit=1)
        )
    assert exc_info.value.args[0] == GCS_QUERY_FAILURE_WARNING


@pytest.mark.skipif(
    sys.version_info <= (3, 7, 0),
    reason=("Not passing in CI although it works locally. Will handle it later."),
)
@pytest.mark.asyncio
async def test_api_manager_list_tasks(state_api_manager):
    data_source_client = state_api_manager.data_source_client
    data_source_client.get_all_registered_raylet_ids = MagicMock()
    data_source_client.get_all_registered_raylet_ids.return_value = ["1", "2"]

    first_task_name = "1"
    second_task_name = "2"
    data_source_client.get_task_info = AsyncMock()
    id = b"1234"
    data_source_client.get_task_info.side_effect = [
        generate_task_data(id, first_task_name),
        generate_task_data(b"2345", second_task_name),
    ]
    result = await state_api_manager.list_tasks(option=create_api_options())
    data_source_client.get_task_info.assert_any_await("1", timeout=DEFAULT_RPC_TIMEOUT)
    data_source_client.get_task_info.assert_any_await("2", timeout=DEFAULT_RPC_TIMEOUT)
    data = result.result
    data = data
    assert len(data) == 2
    verify_schema(TaskState, data[0])
    verify_schema(TaskState, data[1])

    """
    Test limit
    """
    data_source_client.get_task_info.side_effect = [
        generate_task_data(id, first_task_name),
        generate_task_data(b"2345", second_task_name),
    ]
    result = await state_api_manager.list_tasks(option=create_api_options(limit=1))
    data = result.result
    assert len(data) == 1

    """
    Test filters
    """
    data_source_client.get_task_info.side_effect = [
        generate_task_data(id, first_task_name),
        generate_task_data(b"2345", second_task_name),
    ]
    result = await state_api_manager.list_tasks(
        option=create_api_options(filters=[("task_id", "=", bytearray(id).hex())])
    )
    assert len(result.result) == 1

    """
    Test error handling
    """
    data_source_client.get_task_info.side_effect = [
        DataSourceUnavailable(),
        generate_task_data(b"2345", second_task_name),
    ]
    result = await state_api_manager.list_tasks(option=create_api_options(limit=1))
    # Make sure warnings are returned.
    warning = result.partial_failure_warning
    assert (
        NODE_QUERY_FAILURE_WARNING.format(
            type="raylet", total=2, network_failures=1, log_command="raylet.out"
        )
        in warning
    )

    # Test if all RPCs fail, it will raise an exception.
    data_source_client.get_task_info.side_effect = [
        DataSourceUnavailable(),
        DataSourceUnavailable(),
    ]
    with pytest.raises(DataSourceUnavailable):
        result = await state_api_manager.list_tasks(option=create_api_options(limit=1))


@pytest.mark.skipif(
    sys.version_info <= (3, 7, 0),
    reason=("Not passing in CI although it works locally. Will handle it later."),
)
@pytest.mark.asyncio
async def test_api_manager_list_objects(state_api_manager):
    data_source_client = state_api_manager.data_source_client
    obj_1_id = b"1" * 28
    obj_2_id = b"2" * 28
    data_source_client.get_all_registered_raylet_ids = MagicMock()
    data_source_client.get_all_registered_raylet_ids.return_value = ["1", "2"]

    data_source_client.get_object_info = AsyncMock()
    data_source_client.get_object_info.side_effect = [
        GetNodeStatsReply(core_workers_stats=[generate_object_info(obj_1_id)]),
        GetNodeStatsReply(core_workers_stats=[generate_object_info(obj_2_id)]),
    ]
    result = await state_api_manager.list_objects(option=create_api_options())
    data = result.result
    data_source_client.get_object_info.assert_any_await(
        "1", timeout=DEFAULT_RPC_TIMEOUT
    )
    data_source_client.get_object_info.assert_any_await(
        "2", timeout=DEFAULT_RPC_TIMEOUT
    )
    data = data
    assert len(data) == 2
    verify_schema(ObjectState, data[0])
    verify_schema(ObjectState, data[1])

    """
    Test limit
    """
    data_source_client.get_object_info.side_effect = [
        GetNodeStatsReply(core_workers_stats=[generate_object_info(obj_1_id)]),
        GetNodeStatsReply(core_workers_stats=[generate_object_info(obj_2_id)]),
    ]
    result = await state_api_manager.list_objects(option=create_api_options(limit=1))
    data = result.result
    assert len(data) == 1

    """
    Test filters
    """
    data_source_client.get_object_info.side_effect = [
        GetNodeStatsReply(core_workers_stats=[generate_object_info(obj_1_id)]),
        GetNodeStatsReply(core_workers_stats=[generate_object_info(obj_2_id)]),
    ]
    result = await state_api_manager.list_objects(
        option=create_api_options(
            filters=[("object_id", "=", bytearray(obj_1_id).hex())]
        )
    )
    assert len(result.result) == 1

    """
    Test error handling
    """
    data_source_client.get_object_info.side_effect = [
        DataSourceUnavailable(),
        GetNodeStatsReply(core_workers_stats=[generate_object_info(obj_2_id)]),
    ]
    result = await state_api_manager.list_objects(option=create_api_options(limit=1))
    # Make sure warnings are returned.
    warning = result.partial_failure_warning
    assert (
        NODE_QUERY_FAILURE_WARNING.format(
            type="raylet", total=2, network_failures=1, log_command="raylet.out"
        )
        in warning
    )

    # Test if all RPCs fail, it will raise an exception.
    data_source_client.get_object_info.side_effect = [
        DataSourceUnavailable(),
        DataSourceUnavailable(),
    ]
    with pytest.raises(DataSourceUnavailable):
        result = await state_api_manager.list_objects(
            option=create_api_options(limit=1)
        )


@pytest.mark.skipif(
    sys.version_info <= (3, 7, 0),
    reason=("Not passing in CI although it works locally. Will handle it later."),
)
@pytest.mark.asyncio
async def test_api_manager_list_runtime_envs(state_api_manager):
    data_source_client = state_api_manager.data_source_client
    data_source_client.get_all_registered_agent_ids = MagicMock()
    data_source_client.get_all_registered_agent_ids.return_value = ["1", "2", "3"]

    data_source_client.get_runtime_envs_info = AsyncMock()
    data_source_client.get_runtime_envs_info.side_effect = [
        generate_runtime_env_info(RuntimeEnv(**{"pip": ["requests"]})),
        generate_runtime_env_info(
            RuntimeEnv(**{"pip": ["tensorflow"]}), creation_time=15
        ),
        generate_runtime_env_info(RuntimeEnv(**{"pip": ["ray"]}), creation_time=10),
    ]
    result = await state_api_manager.list_runtime_envs(option=create_api_options())
    print(result)
    data = result.result
    data_source_client.get_runtime_envs_info.assert_any_await(
        "1", timeout=DEFAULT_RPC_TIMEOUT
    )
    data_source_client.get_runtime_envs_info.assert_any_await(
        "2", timeout=DEFAULT_RPC_TIMEOUT
    )

    data_source_client.get_runtime_envs_info.assert_any_await(
        "3", timeout=DEFAULT_RPC_TIMEOUT
    )
    assert len(data) == 3
    verify_schema(RuntimeEnvState, data[0])
    verify_schema(RuntimeEnvState, data[1])
    verify_schema(RuntimeEnvState, data[2])

    # Make sure the higher creation time is sorted first.
    assert "creation_time_ms" not in data[0]
    data[1]["creation_time_ms"] > data[2]["creation_time_ms"]

    """
    Test limit
    """
    data_source_client.get_runtime_envs_info.side_effect = [
        generate_runtime_env_info(RuntimeEnv(**{"pip": ["requests"]})),
        generate_runtime_env_info(
            RuntimeEnv(**{"pip": ["tensorflow"]}), creation_time=15
        ),
        generate_runtime_env_info(RuntimeEnv(**{"pip": ["ray"]})),
    ]
    result = await state_api_manager.list_runtime_envs(
        option=create_api_options(limit=1)
    )
    data = result.result
    assert len(data) == 1

    """
    Test filters
    """
    data_source_client.get_runtime_envs_info.side_effect = [
        generate_runtime_env_info(RuntimeEnv(**{"pip": ["requests"]}), success=True),
        generate_runtime_env_info(
            RuntimeEnv(**{"pip": ["tensorflow"]}), creation_time=15, success=True
        ),
        generate_runtime_env_info(RuntimeEnv(**{"pip": ["ray"]}), success=False),
    ]
    result = await state_api_manager.list_runtime_envs(
        option=create_api_options(filters=[("success", "=", False)])
    )
    assert len(result.result) == 1

    """
    Test error handling
    """
    data_source_client.get_runtime_envs_info.side_effect = [
        DataSourceUnavailable(),
        generate_runtime_env_info(RuntimeEnv(**{"pip": ["ray"]})),
        generate_runtime_env_info(RuntimeEnv(**{"pip": ["ray"]})),
    ]
    result = await state_api_manager.list_runtime_envs(
        option=create_api_options(limit=1)
    )
    # Make sure warnings are returned.
    warning = result.partial_failure_warning
    assert (
        NODE_QUERY_FAILURE_WARNING.format(
            type="agent", total=3, network_failures=1, log_command="dashboard_agent.log"
        )
        in warning
    )

    # Test if all RPCs fail, it will raise an exception.
    data_source_client.get_runtime_envs_info.side_effect = [
        DataSourceUnavailable(),
        DataSourceUnavailable(),
        DataSourceUnavailable(),
    ]
    with pytest.raises(DataSourceUnavailable):
        result = await state_api_manager.list_runtime_envs(
            option=create_api_options(limit=1)
        )


def test_type_conversion():
    # Test string
    r = _convert_filters_type([("actor_id", "=", "123")], ActorState)
    assert r[0][2] == "123"
    r = _convert_filters_type([("actor_id", "=", "abcd")], ActorState)
    assert r[0][2] == "abcd"
    r = _convert_filters_type([("actor_id", "=", "True")], ActorState)
    assert r[0][2] == "True"

    # Test boolean
    r = _convert_filters_type([("success", "=", "1")], RuntimeEnvState)
    assert r[0][2]
    r = _convert_filters_type([("success", "=", "True")], RuntimeEnvState)
    assert r[0][2]
    r = _convert_filters_type([("success", "=", "true")], RuntimeEnvState)
    assert r[0][2]
    with pytest.raises(ValueError):
        r = _convert_filters_type([("success", "=", "random_string")], RuntimeEnvState)
    r = _convert_filters_type([("success", "=", "false")], RuntimeEnvState)
    assert r[0][2] is False
    r = _convert_filters_type([("success", "=", "False")], RuntimeEnvState)
    assert r[0][2] is False
    r = _convert_filters_type([("success", "=", "0")], RuntimeEnvState)
    assert r[0][2] is False

    # Test int
    r = _convert_filters_type([("pid", "=", "0")], ObjectState)
    assert r[0][2] == 0
    r = _convert_filters_type([("pid", "=", "123")], ObjectState)
    assert r[0][2] == 123
    # Only integer can be provided.
    with pytest.raises(ValueError):
        r = _convert_filters_type([("pid", "=", "123.3")], ObjectState)
    with pytest.raises(ValueError):
        r = _convert_filters_type([("pid", "=", "abc")], ObjectState)

    # currently, there's no schema that has float column.


"""
Integration tests
"""


@pytest.mark.asyncio
async def test_state_data_source_client(ray_start_cluster):
    cluster = ray_start_cluster
    # head
    cluster.add_node(num_cpus=2)
    ray.init(address=cluster.address)
    # worker
    worker = cluster.add_node(num_cpus=2)

    GRPC_CHANNEL_OPTIONS = (
        *ray_constants.GLOBAL_GRPC_OPTIONS,
        ("grpc.max_send_message_length", ray_constants.GRPC_CPP_MAX_MESSAGE_SIZE),
        ("grpc.max_receive_message_length", ray_constants.GRPC_CPP_MAX_MESSAGE_SIZE),
    )
    gcs_channel = ray._private.utils.init_grpc_channel(
        cluster.address, GRPC_CHANNEL_OPTIONS, asynchronous=True
    )
    client = StateDataSourceClient(gcs_channel)

    """
    Test actor
    """
    result = await client.get_all_actor_info()
    assert isinstance(result, GetAllActorInfoReply)

    """
    Test placement group
    """
    result = await client.get_all_placement_group_info()
    assert isinstance(result, GetAllPlacementGroupReply)

    """
    Test node
    """
    result = await client.get_all_node_info()
    assert isinstance(result, GetAllNodeInfoReply)

    """
    Test worker info
    """
    result = await client.get_all_worker_info()
    assert isinstance(result, GetAllWorkerInfoReply)

    """
    Test job
    """
    job_client = JobSubmissionClient(
        f"http://{ray._private.worker.global_worker.node.address_info['webui_url']}"
    )
    job_id = job_client.submit_job(  # noqa
        # Entrypoint shell command to execute
        entrypoint="ls",
    )
    result = client.get_job_info()
    assert list(result.keys())[0] == job_id
    assert isinstance(result, dict)

    """
    Test tasks
    """
    with pytest.raises(ValueError):
        # Since we didn't register this node id, it should raise an exception.
        result = await client.get_task_info("1234")

    wait_for_condition(lambda: len(ray.nodes()) == 2)
    for node in ray.nodes():
        node_id = node["NodeID"]
        ip = node["NodeManagerAddress"]
        port = int(node["NodeManagerPort"])
        client.register_raylet_client(node_id, ip, port)
        result = await client.get_task_info(node_id)
        assert isinstance(result, GetTasksInfoReply)

    assert len(client.get_all_registered_raylet_ids()) == 2

    """
    Test objects
    """
    with pytest.raises(ValueError):
        # Since we didn't register this node id, it should raise an exception.
        result = await client.get_object_info("1234")

    wait_for_condition(lambda: len(ray.nodes()) == 2)
    for node in ray.nodes():
        node_id = node["NodeID"]
        ip = node["NodeManagerAddress"]
        port = int(node["NodeManagerPort"])
        client.register_raylet_client(node_id, ip, port)
        result = await client.get_object_info(node_id)
        assert isinstance(result, GetNodeStatsReply)

    """
    Test runtime env
    """
    with pytest.raises(ValueError):
        # Since we didn't register this node id, it should raise an exception.
        result = await client.get_runtime_envs_info("1234")
    wait_for_condition(lambda: len(ray.nodes()) == 2)
    for node in ray.nodes():
        node_id = node["NodeID"]
        key = f"{dashboard_consts.DASHBOARD_AGENT_PORT_PREFIX}{node_id}"

        def get_port():
            return ray.experimental.internal_kv._internal_kv_get(
                key, namespace=ray_constants.KV_NAMESPACE_DASHBOARD
            )

        wait_for_condition(lambda: get_port() is not None)
        # The second index is the gRPC port
        port = json.loads(get_port())[1]
        ip = node["NodeManagerAddress"]
        client.register_agent_client(node_id, ip, port)
        result = await client.get_runtime_envs_info(node_id)
        assert isinstance(result, GetRuntimeEnvsInfoReply)

    """
    Test logs
    """
    with pytest.raises(ValueError):
        result = await client.list_logs("1234", "*")
    with pytest.raises(ValueError):
        result = await client.stream_log("1234", "raylet.out", True, 100, 1, 5)

    wait_for_condition(lambda: len(ray.nodes()) == 2)
    # The node information should've been registered in the previous section.
    for node in ray.nodes():
        node_id = node["NodeID"]
        result = await client.list_logs(node_id, timeout=30, glob_filter="*")
        assert isinstance(result, ListLogsReply)

        stream = await client.stream_log(node_id, "raylet.out", False, 10, 1, 5)
        async for logs in stream:
            log_lines = len(logs.data.decode().split("\n"))
            assert isinstance(logs, StreamLogReply)
            assert log_lines >= 10
            assert log_lines <= 11

    """
    Test the exception is raised when the RPC error occurs.
    """
    cluster.remove_node(worker)
    # Wait until the dead node information is propagated.
    wait_for_condition(
        lambda: len(list(filter(lambda node: node["Alive"], ray.nodes()))) == 1
    )
    for node in ray.nodes():
        node_id = node["NodeID"]
        if node["Alive"]:
            continue

        # Querying to the dead node raises gRPC error, which should raise an exception.
        with pytest.raises(DataSourceUnavailable):
            await client.get_object_info(node_id)

        # Make sure unregister API works as expected.
        client.unregister_raylet_client(node_id)
        assert len(client.get_all_registered_raylet_ids()) == 1
        # Since the node_id is unregistered, the API should raise ValueError.
        with pytest.raises(ValueError):
            result = await client.get_object_info(node_id)


def is_hex(val):
    try:
        int_val = int(val, 16)
    except ValueError:
        return False
    # Should remove leading 0 because when the value is converted back
    # to hex, it is removed.
    val = val.lstrip("0")
    return f"0x{val}" == hex(int_val)


@pytest.mark.xfail(cluster_not_supported, reason="cluster not supported on Windows")
def test_cli_apis_sanity_check(ray_start_cluster):
    """Test all of CLI APIs work as expected."""
    NUM_NODES = 4
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=2)
    ray.init(address=cluster.address)
    for _ in range(NUM_NODES - 1):
        cluster.add_node(num_cpus=2)
    runner = CliRunner()

    client = JobSubmissionClient(
        f"http://{ray._private.worker.global_worker.node.address_info['webui_url']}"
    )

    @ray.remote
    def f():
        import time

        time.sleep(30)

    @ray.remote
    class Actor:
        pass

    obj = ray.put(3)  # noqa
    task = f.remote()  # noqa
    actor = Actor.remote()  # noqa
    actor_runtime_env = Actor.options(  # noqa
        runtime_env={"pip": ["requests"]}
    ).remote()
    job_id = client.submit_job(  # noqa
        # Entrypoint shell command to execute
        entrypoint="ls",
    )
    pg = ray.util.placement_group(bundles=[{"CPU": 1}])  # noqa

    def verify_output(cmd, args: List[str], necessary_substrings: List[str]):
        result = runner.invoke(cmd, args)
        exit_code_correct = result.exit_code == 0
        substring_matched = all(
            substr in result.output for substr in necessary_substrings
        )
        print(result.output)
        return exit_code_correct and substring_matched

    wait_for_condition(lambda: verify_output(cli_list, ["actors"], ["actor_id"]))
    wait_for_condition(lambda: verify_output(cli_list, ["workers"], ["worker_id"]))
    wait_for_condition(lambda: verify_output(cli_list, ["nodes"], ["node_id"]))
    wait_for_condition(
        lambda: verify_output(cli_list, ["placement-groups"], ["placement_group_id"])
    )
    wait_for_condition(lambda: verify_output(cli_list, ["jobs"], ["raysubmit"]))
    wait_for_condition(lambda: verify_output(cli_list, ["tasks"], ["task_id"]))
    wait_for_condition(lambda: verify_output(cli_list, ["objects"], ["object_id"]))
    wait_for_condition(
        lambda: verify_output(cli_list, ["runtime-envs"], ["runtime_env"])
    )

    # Test get node by id
    nodes = ray.nodes()
    wait_for_condition(
        lambda: verify_output(
            cli_get, ["nodes", nodes[0]["NodeID"]], ["node_id", nodes[0]["NodeID"]]
        )
    )
    # Test get workers by id
    workers = global_state.workers()
    assert len(workers) > 0
    worker_id = list(workers.keys())[0]
    wait_for_condition(
        lambda: verify_output(cli_get, ["workers", worker_id], ["worker_id", worker_id])
    )

    # Test get actors by id
    wait_for_condition(
        lambda: verify_output(
            cli_get,
            ["actors", actor._actor_id.hex()],
            ["actor_id", actor._actor_id.hex()],
        )
    )

    # Test get placement groups by id
    wait_for_condition(
        lambda: verify_output(
            cli_get,
            ["placement-groups", pg.id.hex()],
            ["placement_group_id", pg.id.hex()],
        )
    )

    # Test get objects by id
    wait_for_condition(
        lambda: verify_output(cli_get, ["objects", obj.hex()], ["object_id", obj.hex()])
    )

    # TODO(rickyyx:alpha-obs):
    # - get job by id: jobs is not currently filterable by id
    # - get task by id: no easy access to tasks yet


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Failed on Windows",
)
def test_list_get_actors(shutdown_only):
    ray.init()

    @ray.remote
    class A:
        pass

    a = A.remote()  # noqa

    def verify():
        # Test list
        actors = list_actors()
        assert len(actors) == 1
        assert actors[0]["state"] == "ALIVE"
        assert is_hex(actors[0]["actor_id"])
        assert a._actor_id.hex() == actors[0]["actor_id"]

        # Test get
        for actor in actors:
            get_actor_data = get_actor(actor["actor_id"])
            assert get_actor_data is not None
            assert get_actor_data == actor

        return True

    wait_for_condition(verify)
    print(list_actors())


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Failed on Windows",
)
def test_list_get_pgs(shutdown_only):
    ray.init()
    pg = ray.util.placement_group(bundles=[{"CPU": 1}])  # noqa

    def verify():
        # Test list
        pgs = list_placement_groups()
        assert len(pgs) == 1
        assert pgs[0]["state"] == "CREATED"
        assert is_hex(pgs[0]["placement_group_id"])
        assert pg.id.hex() == pgs[0]["placement_group_id"]

        # Test get
        for pg_data in pgs:
            get_pg_data = get_placement_group(pg_data["placement_group_id"])
            assert get_pg_data is not None
            assert pg_data == get_pg_data

        return True

    wait_for_condition(verify)
    print(list_placement_groups())


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Failed on Windows",
)
def test_list_get_nodes(shutdown_only):
    ray.init()

    def verify():
        nodes = list_nodes()
        assert nodes[0]["state"] == "ALIVE"
        assert is_hex(nodes[0]["node_id"])

        # Check with legacy API
        check_nodes = ray.nodes()
        assert len(check_nodes) == len(nodes)

        sorted(check_nodes, key=lambda n: n["NodeID"])
        sorted(nodes, key=lambda n: n["node_id"])

        for check_node, node in zip(check_nodes, nodes):
            assert check_node["NodeID"] == node["node_id"]
            assert check_node["NodeName"] == node["node_name"]

        # Check the Get api
        for node in nodes:
            get_node_data = get_node(node["node_id"])
            assert get_node_data == node

        return True

    wait_for_condition(verify)
    print(list_nodes())


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Failed on Windows",
)
def test_list_jobs(shutdown_only):
    ray.init()
    client = JobSubmissionClient(
        f"http://{ray._private.worker.global_worker.node.address_info['webui_url']}"
    )
    job_id = client.submit_job(  # noqa
        # Entrypoint shell command to execute
        entrypoint="ls",
    )

    def verify():
        job_data = list_jobs()[0]
        print(job_data)
        job_id_from_api = job_data["job_id"]
        correct_state = job_data["status"] == "SUCCEEDED"
        correct_id = job_id == job_id_from_api
        return correct_state and correct_id

    wait_for_condition(verify)
    print(list_jobs())


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Failed on Windows",
)
def test_list_get_workers(shutdown_only):
    ray.init()

    def verify():
        workers = list_workers()
        assert is_hex(workers[0]["worker_id"])
        # +1 to take into account of drivers.
        assert len(workers) == ray.cluster_resources()["CPU"] + 1

        # Test get worker returns the same result
        for worker in workers:
            got_worker = get_worker(worker["worker_id"])
            assert got_worker == worker

        return True

    wait_for_condition(verify)
    print(list_workers())


def test_list_get_tasks(shutdown_only):
    ray.init(num_cpus=2)

    @ray.remote
    def f():
        import time

        time.sleep(30)

    @ray.remote
    def g(dep):
        import time

        time.sleep(30)

    @ray.remote(num_gpus=1)
    def impossible():
        pass

    out = [f.remote() for _ in range(2)]  # noqa
    g_out = g.remote(f.remote())  # noqa
    im = impossible.remote()  # noqa

    def verify():
        tasks = list_tasks()
        assert len(tasks) == 5
        waiting_for_execution = len(
            list(
                filter(
                    lambda task: task["scheduling_state"] == "WAITING_FOR_EXECUTION",
                    tasks,
                )
            )
        )
        assert waiting_for_execution == 0
        scheduled = len(
            list(filter(lambda task: task["scheduling_state"] == "SCHEDULED", tasks))
        )
        assert scheduled == 2
        waiting_for_dep = len(
            list(
                filter(
                    lambda task: task["scheduling_state"] == "WAITING_FOR_DEPENDENCIES",
                    tasks,
                )
            )
        )
        assert waiting_for_dep == 1
        running = len(
            list(
                filter(
                    lambda task: task["scheduling_state"] == "RUNNING",
                    tasks,
                )
            )
        )
        assert running == 2

        # Test get tasks
        for task in tasks:
            get_task_data = get_task(task["task_id"])
            assert get_task_data == task

        return True

    wait_for_condition(verify)
    print(list_tasks())


def test_list_actor_tasks(shutdown_only):
    ray.init(num_cpus=2)

    @ray.remote
    class Actor:
        def call(self):
            import time

            time.sleep(30)

    a = Actor.remote()
    calls = [a.call.remote() for _ in range(10)]  # noqa

    def verify():
        tasks = list_tasks()
        # Actor.__init__: 1 finished
        # Actor.call: 1 running, 9 waiting for execution (queued).
        correct_num_tasks = len(tasks) == 11
        waiting_for_execution = len(
            list(
                filter(
                    lambda task: task["scheduling_state"] == "WAITING_FOR_EXECUTION",
                    tasks,
                )
            )
        )
        scheduled = len(
            list(filter(lambda task: task["scheduling_state"] == "SCHEDULED", tasks))
        )
        waiting_for_dep = len(
            list(
                filter(
                    lambda task: task["scheduling_state"] == "WAITING_FOR_DEPENDENCIES",
                    tasks,
                )
            )
        )
        running = len(
            list(
                filter(
                    lambda task: task["scheduling_state"] == "RUNNING",
                    tasks,
                )
            )
        )

        return (
            correct_num_tasks
            and running == 1
            and waiting_for_dep == 0
            and waiting_for_execution == 9
            and scheduled == 0
        )

    wait_for_condition(verify)
    print(list_tasks())


def test_list_get_objects(shutdown_only):
    ray.init()
    import numpy as np

    data = np.ones(50 * 1024 * 1024, dtype=np.uint8)
    plasma_obj = ray.put(data)

    @ray.remote
    def f(obj):
        print(obj)

    ray.get(f.remote(plasma_obj))

    def verify():
        obj = list_objects()[0]
        # For detailed output, the test is covered from `test_memstat.py`
        assert obj["object_id"] == plasma_obj.hex()

        got_objs = get_objects(plasma_obj.hex())
        assert len(got_objs) == 1
        assert obj == got_objs[0]

        return True

    wait_for_condition(verify)
    print(list_objects())


@pytest.mark.skipif(
    sys.platform == "win32", reason="Runtime env not working in Windows."
)
def test_list_runtime_envs(shutdown_only):
    ray.init(runtime_env={"pip": ["requests"]})

    @ray.remote
    class Actor:
        def ready(self):
            pass

    a = Actor.remote()  # noqa
    b = Actor.options(runtime_env={"pip": ["nonexistent_dep"]}).remote()  # noqa
    ray.get(a.ready.remote())
    with pytest.raises(ray.exceptions.RuntimeEnvSetupError):
        ray.get(b.ready.remote())

    def verify():
        result = list_runtime_envs()
        correct_num = len(result) == 2

        failed_runtime_env = result[0]
        correct_failed_state = (
            not failed_runtime_env["success"]
            and failed_runtime_env.get("error")
            and failed_runtime_env["ref_cnt"] == "0"
        )

        successful_runtime_env = result[1]
        correct_successful_state = (
            successful_runtime_env["success"]
            and successful_runtime_env["ref_cnt"] == "2"
        )
        return correct_num and correct_failed_state and correct_successful_state

    wait_for_condition(verify)


def test_limit(shutdown_only):
    ray.init()

    @ray.remote
    class A:
        def ready(self):
            pass

    actors = [A.remote() for _ in range(4)]
    ray.get([actor.ready.remote() for actor in actors])

    output = list_actors(limit=2)
    assert len(output) == 2

    # Make sure the output is deterministic.
    assert output == list_actors(limit=2)


def test_network_failure(shutdown_only):
    """When the request fails due to network failure,
    verifies it raises an exception."""
    ray.init()

    @ray.remote
    def f():
        import time

        time.sleep(30)

    a = [f.remote() for _ in range(4)]  # noqa
    wait_for_condition(lambda: len(list_tasks()) == 4)

    # Kill raylet so that list_tasks will have network error on querying raylets.
    ray._private.worker._global_node.kill_raylet()

    with pytest.raises(RayStateApiException):
        list_tasks(_explain=True)


def test_network_partial_failures(ray_start_cluster):
    """When the request fails due to network failure,
    verifies it prints proper warning."""
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=2)
    ray.init(address=cluster.address)
    n = cluster.add_node(num_cpus=2)

    @ray.remote
    def f():
        import time

        time.sleep(30)

    a = [f.remote() for _ in range(4)]  # noqa
    wait_for_condition(lambda: len(list_tasks()) == 4)

    # Make sure when there's 0 node failure, it doesn't print the error.
    with pytest.warns(None) as record:
        list_tasks(_explain=True)
    assert len(record) == 0

    # Kill raylet so that list_tasks will have network error on querying raylets.
    cluster.remove_node(n, allow_graceful=False)

    with pytest.warns(RuntimeWarning):
        list_tasks(_explain=True)

    # Make sure when _explain == False, warning is not printed.
    with pytest.warns(None) as record:
        list_tasks(_explain=False)
    assert len(record) == 0


def test_network_partial_failures_timeout(monkeypatch, ray_start_cluster):
    """When the request fails due to network timeout,
    verifies it prints proper warning."""
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=2)
    ray.init(address=cluster.address)
    with monkeypatch.context() as m:
        # defer for 10s for the second node.
        m.setenv(
            "RAY_testing_asio_delay_us",
            "NodeManagerService.grpc_server.GetTasksInfo=10000000:10000000",
        )
        cluster.add_node(num_cpus=2)

    @ray.remote
    def f():
        import time

        time.sleep(30)

    a = [f.remote() for _ in range(4)]  # noqa

    def verify():
        with pytest.warns(None) as record:
            list_tasks(_explain=True, timeout=5)
        return len(record) == 1

    wait_for_condition(verify)


@pytest.mark.asyncio
async def test_cli_format_print(state_api_manager):
    data_source_client = state_api_manager.data_source_client
    actor_id = b"1234"
    data_source_client.get_all_actor_info.return_value = GetAllActorInfoReply(
        actor_table_data=[generate_actor_data(actor_id), generate_actor_data(b"12345")]
    )
    result = await state_api_manager.list_actors(option=create_api_options())
    result = result.result
    # If the format is not yaml, it will raise an exception.
    yaml.load(
        format_list_api_output(result, format=AvailableFormat.YAML),
        Loader=yaml.FullLoader,
    )
    # If the format is not json, it will raise an exception.
    json.loads(format_list_api_output(result, format=AvailableFormat.JSON))
    # Verify the default format is yaml
    yaml.load(format_list_api_output(result), Loader=yaml.FullLoader)
    with pytest.raises(ValueError):
        format_list_api_output(result, format="random_format")
    with pytest.raises(NotImplementedError):
        format_list_api_output(result, format=AvailableFormat.TABLE)


def test_filter(shutdown_only):
    ray.init()

    # Test unsupported predicates.
    with pytest.raises(ValueError):
        list_actors(filters=[("state", ">", "DEAD")])

    @ray.remote
    class Actor:
        def __init__(self):
            self.obj = None

        def ready(self):
            pass

        def put(self):
            self.obj = ray.put(123)

        def getpid(self):
            import os

            return os.getpid()

    """
    Test basic case.
    """
    a = Actor.remote()
    b = Actor.remote()

    a_pid = ray.get(a.getpid.remote())
    b_pid = ray.get(b.getpid.remote())

    ray.get([a.ready.remote(), b.ready.remote()])
    ray.kill(b)

    def verify():
        result = list_actors(filters=[("state", "=", "DEAD")])
        assert len(result) == 1
        actor = result[0]
        assert actor["pid"] == b_pid

        result = list_actors(filters=[("state", "!=", "DEAD")])
        assert len(result) == 1
        actor = result[0]
        assert actor["pid"] == a_pid
        return True

    wait_for_condition(verify)

    """
    Test filter with different types (integer).
    """
    obj_1 = ray.put(123)  # noqa
    ray.get(a.put.remote())
    pid = ray.get(a.getpid.remote())

    def verify():
        # There's only 1 object.
        result = list_objects(
            filters=[("pid", "=", pid), ("reference_type", "=", "LOCAL_REFERENCE")]
        )
        return len(result) == 1

    wait_for_condition(verify)

    """
    Test CLI
    """
    dead_actor_id = list_actors(filters=[("state", "=", "DEAD")])[0]["actor_id"]
    alive_actor_id = list_actors(filters=[("state", "=", "ALIVE")])[0]["actor_id"]
    runner = CliRunner()
    result = runner.invoke(cli_list, ["actors", "--filter", "state=DEAD"])
    assert result.exit_code == 0
    assert dead_actor_id in result.output
    assert alive_actor_id not in result.output

    result = runner.invoke(cli_list, ["actors", "--filter", "state!=DEAD"])
    assert result.exit_code == 0
    assert dead_actor_id not in result.output
    assert alive_actor_id in result.output


if __name__ == "__main__":
    import os
    import sys

    if os.environ.get("PARALLEL_CI"):
        sys.exit(pytest.main(["-n", "auto", "--boxed", "-vs", __file__]))
    else:
        sys.exit(pytest.main(["-sv", __file__]))
