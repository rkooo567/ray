import logging

from typing import List, Dict
from dataclasses import dataclass, fields

from ray.dashboard.modules.job.common import JobInfo

logger = logging.getLogger(__name__)


def filter_fields(data: dict, state_dataclass) -> dict:
    """Filter the given data using keys from a given state dataclass."""
    filtered_data = {}
    for field in fields(state_dataclass):
        filtered_data[field.name] = data[field.name]
    return filtered_data


@dataclass(init=True)
class ListApiOptions:
    limit: int
    timeout: int


# TODO(sang): Replace it with Pydantic or gRPC schema (once interface is finalized).
@dataclass(init=True)
class ActorState:
    actor_id: str
    state: str
    class_name: str


@dataclass(init=True)
class PlacementGroupState:
    placement_group_id: str
    state: str


@dataclass(init=True)
class NodeState:
    node_id: str
    state: str


JobState = JobInfo


@dataclass(init=True)
class WorkerState:
    worker_id: str
    is_alive: str
    worker_type: str


@dataclass(init=True)
class TaskState:
    task_id: str
    name: str
    scheduling_state: str


@dataclass(init=True)
class ObjectState:
    object_id: str
    pid: int
    node_ip_address: str
    object_size: int
    reference_type: str
    call_site: str
    task_status: str
    local_ref_count: int
    pinned_in_memory: int
    submitted_task_ref_count: int
    contained_in_owned: int
    type: str


@dataclass(init=True)
class DetailedResourceSummary:
    summary: "ResourceSummary"
    usage: "TaskResourceUsage"


@dataclass(init=True)
class ResourceSummary:
    available: Dict[str, float]
    total: Dict[str, float]


@dataclass(init=True)
class TaskResourceUsage:
    task_name: str
    # List of resource sets used by this task
    # and their respective counts
    resource_set_counts: List["ResourceSetCount"]


@dataclass(init=True)
class ResourceSetCount:
    resource_set: Dict[str, float]
    count: int
