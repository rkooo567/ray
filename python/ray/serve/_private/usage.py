from enum import Enum
from typing import Dict, Optional

from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag


class ServeUsageTag(Enum):
    API_VERSION = TagKey.SERVE_API_VERSION
    NUM_DEPLOYMENTS = TagKey.SERVE_NUM_DEPLOYMENTS
    GCS_STORAGE = TagKey.GCS_STORAGE
    NUM_GPU_DEPLOYMENTS = TagKey.SERVE_NUM_GPU_DEPLOYMENTS
    FASTAPI_USED = TagKey.SERVE_FASTAPI_USED
    DAG_DRIVER_USED = TagKey.SERVE_DAG_DRIVER_USED
    HTTP_ADAPTER_USED = TagKey.SERVE_HTTP_ADAPTER_USED
    GRPC_INGRESS_USED = TagKey.SERVE_GRPC_INGRESS_USED
    REST_API_VERSION = TagKey.SERVE_REST_API_VERSION
    NUM_APPS = TagKey.SERVE_NUM_APPS
    NUM_REPLICAS_LIGHTWEIGHT_UPDATED = TagKey.SERVE_NUM_REPLICAS_LIGHTWEIGHT_UPDATED
    USER_CONFIG_LIGHTWEIGHT_UPDATED = TagKey.SERVE_USER_CONFIG_LIGHTWEIGHT_UPDATED
    AUTOSCALING_CONFIG_LIGHTWEIGHT_UPDATED = (
        TagKey.SERVE_AUTOSCALING_CONFIG_LIGHTWEIGHT_UPDATED
    )
    RAY_SERVE_HANDLE_API_USED = TagKey.SERVE_RAY_SERVE_HANDLE_API_USED
    RAY_SERVE_SYNC_HANDLE_API_USED = TagKey.SERVE_RAY_SERVE_SYNC_HANDLE_API_USED
    DEPLOYMENT_HANDLE_API_USED = TagKey.SERVE_DEPLOYMENT_HANDLE_API_USED
    DEPLOYMENT_HANDLE_TO_OBJECT_REF_API_USED = (
        TagKey.SERVE_DEPLOYMENT_HANDLE_TO_OBJECT_REF_API_USED
    )
    MULTIPLEXED_API_USED = TagKey.SERVE_MULTIPLEXED_API_USED

    def record(self, value: str):
        """Record telemetry value."""
        record_extra_usage_tag(self.value, value)

    def get_value_from_report(self, report: Dict) -> Optional[str]:
        """Returns `None` if the tag isn't in the report."""
        if "extra_usage_tags" not in report:
            return None

        return report["extra_usage_tags"].get(TagKey.Name(self.value).lower(), None)
