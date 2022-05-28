import click
import json
import yaml

from dataclasses import fields
from enum import Enum, unique
from typing import Union, List, Tuple

import ray

import ray._private.services as services
import ray.ray_constants as ray_constants
from ray._private.gcs_utils import use_gcs_for_bootstrap
from ray._private.gcs_utils import GcsClient

from ray.experimental.state.api import (
    list_actors,
    list_nodes,
    list_jobs,
    list_placement_groups,
    list_workers,
    list_tasks,
    list_objects,
    list_runtime_envs,
)
from ray.experimental.state.common import (
    SupportedFilterType,
    StateSchema,
    JobState,
    ActorState,
    PlacementGroupState,
    WorkerState,
    TaskState,
    ObjectState,
    RuntimeEnvState,
    NodeState,
)


@unique
class AvailableFormat(Enum):
    DEFAULT = "default"
    JSON = "json"
    YAML = "yaml"
    TABLE = "table"


def _get_available_formats() -> List[str]:
    """Return the available formats in a list of string"""
    return [format_enum.value for format_enum in AvailableFormat]


def get_state_api_output_to_print(
    state_data: Union[dict, list], *, format: AvailableFormat = AvailableFormat.DEFAULT
):
    if len(state_data) == 0:
        return "No resource in the cluster"

    # Default is yaml.
    if format == AvailableFormat.DEFAULT:
        return yaml.dump(state_data, indent=4, explicit_start=True)
    if format == AvailableFormat.YAML:
        return yaml.dump(state_data, indent=4, explicit_start=True)
    elif format == AvailableFormat.JSON:
        return json.dumps(state_data)
    elif format == AvailableFormat.TABLE:
        raise NotImplementedError("Table formatter is not implemented yet.")
    else:
        raise ValueError(
            f"Unexpected format: {format}. "
            f"Supported formatting: {_get_available_formats()}"
        )


"""
List API
"""


def _should_explain(format: AvailableFormat):
    # If the format is json or yaml, it should not print stats because
    # users don't want additional strings.
    return format == AvailableFormat.DEFAULT or format == AvailableFormat.TABLE


def _convert_filters_type(
    filter: List[Tuple[str, str]], schema: StateSchema
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

    for col, val in filter:
        if col in schema:
            column_type = schema[col]
            if column_type is int:
                try:
                    val = int(val)
                except ValueError:
                    raise ValueError(
                        f"Invalid filter `--filter {col} {val}` for a int type "
                        "column. Please provide an integer filter "
                        f"`--filter {col} [int]`"
                    )
            elif column_type is float:
                try:
                    val = float(val)
                except ValueError:
                    raise ValueError(
                        f"Invalid filter `--filter {col} {val}` for a float "
                        "type column. Please provide an integer filter "
                        f"`--filter {col} [float]`"
                    )
            elif column_type is bool:
                # Without this, "False" will become True.
                if val == "False" or val == "false" or val == "0":
                    val = False
                elif val == "True" or val == "true" or val == "1":
                    val = True
                else:
                    raise ValueError(
                        f"Invalid filter `--filter {col} {val}` for a boolean "
                        "type column. Please provide "
                        f"`--filter {col} [True|true|1]` for True or "
                        f"`--filter {col} [False|false|0]` for False."
                    )
        new_filter.append((col, val))
    return new_filter


@click.group("list")
@click.pass_context
def list_state_cli_group(ctx):
    address = services.canonicalize_bootstrap_address(None)
    gcs_client = GcsClient(address=address, nums_reconnect_retry=0)
    ray.experimental.internal_kv._initialize_internal_kv(gcs_client)
    api_server_url = ray._private.utils.internal_kv_get_with_retry(
        gcs_client,
        ray_constants.DASHBOARD_ADDRESS,
        namespace=ray_constants.KV_NAMESPACE_DASHBOARD,
        num_retries=20,
    )

    if api_server_url is None:
        raise ValueError(
            (
                "Couldn't obtain the API server address from GCS. It is likely that "
                "the GCS server is down. Check gcs_server.[out | err] to see if it is "
                "still alive."
            )
        )

    assert use_gcs_for_bootstrap()
    ctx.ensure_object(dict)
    ctx.obj["api_server_url"] = f"http://{api_server_url.decode()}"


list_format_option = click.option(
    "--format", default="default", type=click.Choice(_get_available_formats())
)
list_filter_option = click.option(
    "-f",
    "--filter",
    help=(
        "A key value pair to filter the result. "
        "For example, specify --filter [column] [value] "
        "to filter out data that satsifies column==value."
    ),
    nargs=2,
    type=click.Tuple([str, str]),
    multiple=True,
)


@list_state_cli_group.command()
@list_format_option
@list_filter_option
@click.pass_context
def actors(ctx, format: str, filter: List[Tuple[str, str]]):
    url = ctx.obj["api_server_url"]
    format = AvailableFormat(format)
    print(
        get_state_api_output_to_print(
            list_actors(
                api_server_url=url,
                filters=_convert_filters_type(filter, ActorState),
                _explain=_should_explain(format),
            ),
            format=format,
        )
    )


@list_state_cli_group.command()
@list_format_option
@list_filter_option
@click.pass_context
def placement_groups(ctx, format: str, filter: List[Tuple[str, str]]):
    url = ctx.obj["api_server_url"]
    format = AvailableFormat(format)
    print(
        get_state_api_output_to_print(
            list_placement_groups(
                api_server_url=url,
                filters=_convert_filters_type(filter, PlacementGroupState),
                _explain=_should_explain(format),
            ),
            format=format,
        )
    )


@list_state_cli_group.command()
@list_format_option
@list_filter_option
@click.pass_context
def nodes(ctx, format: str, filter: List[Tuple[str, str]]):
    url = ctx.obj["api_server_url"]
    format = AvailableFormat(format)
    print(
        get_state_api_output_to_print(
            list_nodes(
                api_server_url=url,
                filters=_convert_filters_type(filter, NodeState),
                _explain=_should_explain(format),
            ),
            format=format,
        )
    )


@list_state_cli_group.command()
@list_format_option
@list_filter_option
@click.pass_context
def jobs(ctx, format: str, filter: List[Tuple[str, str]]):
    url = ctx.obj["api_server_url"]
    format = AvailableFormat(format)
    print(
        get_state_api_output_to_print(
            list_jobs(
                api_server_url=url,
                filters=_convert_filters_type(filter, JobState),
                _explain=_should_explain(format),
            ),
            format=format,
        )
    )


@list_state_cli_group.command()
@list_format_option
@list_filter_option
@click.pass_context
def workers(ctx, format: str, filter: List[Tuple[str, str]]):
    url = ctx.obj["api_server_url"]
    format = AvailableFormat(format)
    print(
        get_state_api_output_to_print(
            list_workers(
                api_server_url=url,
                filters=_convert_filters_type(filter, WorkerState),
                _explain=_should_explain(format),
            ),
            format=format,
        )
    )


@list_state_cli_group.command()
@list_format_option
@list_filter_option
@click.pass_context
def tasks(ctx, format: str, filter: List[Tuple[str, str]]):
    url = ctx.obj["api_server_url"]
    format = AvailableFormat(format)
    print(
        get_state_api_output_to_print(
            list_tasks(
                api_server_url=url,
                filters=_convert_filters_type(filter, TaskState),
                _explain=_should_explain(format),
            ),
            format=format,
        )
    )


@list_state_cli_group.command()
@list_format_option
@list_filter_option
@click.pass_context
def objects(ctx, format: str, filter: List[Tuple[str, str]]):
    url = ctx.obj["api_server_url"]
    format = AvailableFormat(format)
    print(
        get_state_api_output_to_print(
            list_objects(
                api_server_url=url,
                filters=_convert_filters_type(filter, ObjectState),
                _explain=_should_explain(format),
            ),
            format=format,
        )
    )


@list_state_cli_group.command()
@list_format_option
@list_filter_option
@click.pass_context
def runtime_envs(ctx, format: str, filter: List[Tuple[str, str]]):
    url = ctx.obj["api_server_url"]
    format = AvailableFormat(format)
    print(
        get_state_api_output_to_print(
            list_runtime_envs(
                api_server_url=url,
                filters=_convert_filters_type(filter, RuntimeEnvState),
                _explain=_should_explain(format),
            ),
            format=format,
        )
    )
