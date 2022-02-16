import pytest
import sys
import ray
import pathlib
import json
import time

from dataclasses import asdict
from pathlib import Path
from jsonschema import validate

import ray._private.usage.usage_lib as ray_usage_lib
import ray._private.usage.usage_constants as usage_constants

from ray._private.test_utils import wait_for_condition, run_string_as_driver
from ray import serve

schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "schema_version": {"type": "string"},
        "source": {"type": "string"},
        "session_id": {"type": "string"},
        "collect_timestamp_ms": {"type": "integer"},
        "ray_version": {"type": "string"},
        "git_commit": {"type": "string"},
        "os": {"type": "string"},
        "python_version": {"type": "string"},
    },
}


@pytest.fixture
def shutdown_serve():
    yield
    serve.shutdown()


def test_usage_lib_cluster_metadata_generation(monkeypatch, shutdown_only):
    with monkeypatch.context() as m:
        m.setenv("RAY_USAGE_STATS_ENABLED", "1")
        ray.init(num_cpus=0)
        """
        Test metadata stored is equivalent to `_generate_cluster_metadata`.
        """
        meta = ray_usage_lib._generate_cluster_metadata()
        cluster_metadata = ray_usage_lib.get_cluster_metadata(
            ray.experimental.internal_kv.internal_kv_get_gcs_client(), num_retries=20
        )
        # Session id is random.
        meta.pop("session_id")
        cluster_metadata.pop("session_id")
        assert meta == cluster_metadata

        """
        Make sure put & get works properly.
        """
        cluster_metadata = ray_usage_lib.put_cluster_metadata(
            ray.experimental.internal_kv.internal_kv_get_gcs_client(), num_retries=20
        )
        assert cluster_metadata == ray_usage_lib.get_cluster_metadata(
            ray.experimental.internal_kv.internal_kv_get_gcs_client(), num_retries=20
        )


def test_usage_lib_cluster_metadata_generation_usage_disabled(shutdown_only):
    """
    Make sure only version information is generated when usage stats are not enabled.
    """
    meta = ray_usage_lib._generate_cluster_metadata()
    assert "ray_version" in meta
    assert "python_version" in meta
    assert len(meta) == 2


def test_usage_lib_report_data(monkeypatch, shutdown_only, shutdown_serve):
    with monkeypatch.context() as m:
        m.setenv("RAY_USAGE_STATS_ENABLED", "1")
        ray.init(num_cpus=0)
        """
        Make sure the generated data is following the schema.
        """
        cluster_metadata = ray_usage_lib.get_cluster_metadata(
            ray.experimental.internal_kv.internal_kv_get_gcs_client(), num_retries=20
        )
        d = ray_usage_lib.generate_report_data(cluster_metadata)
        validate(instance=asdict(d), schema=schema)

        """
        Make sure writing to a file works as expected
        """
        global_node = ray.worker._global_node
        temp_dir = global_node.get_temp_dir_path()
        client = ray_usage_lib.UsageReportClient()
        client._write_usage_data(d, temp_dir)

        def file_exists():
            for path in pathlib.Path(temp_dir).iterdir():
                if usage_constants.USAGE_STATS_FILE in str(path):
                    return True
            return False

        wait_for_condition(file_exists)

        """
        Make sure report usage data works as expected
        """
        # Start the ray serve server to verify requests are sent
        # to the right place.
        serve.start()

        @serve.deployment(ray_actor_options={"num_cpus": 0})
        async def usage(request):
            body = await request.json()
            if body == asdict(d):
                return True
            else:
                return False

        usage.deploy()

        # Query our endpoint over HTTP.
        r = client._report_usage_data("http://127.0.0.1:8000/usage", d)
        r.raise_for_status()
        assert json.loads(r.text) is True


def test_usage_report_e2e(monkeypatch, shutdown_only, shutdown_serve):
    """
    Test usage report works e2e with env vars.
    """
    with monkeypatch.context() as m:
        m.setenv("RAY_USAGE_STATS_ENABLED", "1")
        m.setenv("RAY_USAGE_STATS_REPORT_URL", "http://127.0.0.1:8000/usage")
        m.setenv("RAY_USAGE_STATS_REPORT_INTERVAL_S", "1")
        ray.init(num_cpus=0)

        @ray.remote(num_cpus=0)
        class StatusReporter:
            def __init__(self):
                self.reported = 0
                self.payload = None

            def report_payload(self, payload):
                self.payload = payload

            def reported(self):
                self.reported += 1

            def get(self):
                return self.reported

            def get_payload(self):
                return self.payload

        reporter = StatusReporter.remote()

        serve.start()

        # Usage report should be sent to the URL every 1 second.
        @serve.deployment(ray_actor_options={"num_cpus": 0})
        async def usage(request):
            body = await request.json()
            reporter.reported.remote()
            reporter.report_payload.remote(body)
            return True

        usage.deploy()
        # Since the interval is 1 second, there must have been
        # more than 5 requests sent within 30 seconds.
        wait_for_condition(lambda: ray.get(reporter.get.remote()) > 5, timeout=30)
        validate(instance=ray.get(reporter.get_payload.remote()), schema=schema)


def test_usage_report_error_not_displayed_to_users(monkeypatch):
    """
    Make sure when the incorrect URL is set, the error message is not printed to users.
    """
    with monkeypatch.context() as m:
        m.setenv("RAY_USAGE_STATS_ENABLED", "1")
        m.setenv("RAY_USAGE_STATS_REPORT_URL", "http://127.0.0.1:8000")
        m.setenv("RAY_USAGE_STATS_REPORT_INTERVAL_S", "1")
        script = """
import ray
import time

ray.init(num_cpus=0)
# Wait long enough
time.sleep(2)
        """
        out = run_string_as_driver(script)
        # Only the basic message;
        # View the Ray dashboard at http://127.0.0.1:8265
        # should be displayed. No more output should be displayed although
        # the usage stats report fail.
        assert len(out.strip().split("\n")) <= 1


def test_usage_report_disabled(monkeypatch):
    """
    Make sure usage report module is disabled when the env var is not set.
    It also verifies that the failure message is not printed (note that
    the invalid report url is given as an env var).
    """
    with monkeypatch.context() as m:
        m.setenv("RAY_USAGE_STATS_ENABLED", "0")
        m.setenv("RAY_USAGE_STATS_REPORT_URL", "http://127.0.0.1:8000")
        m.setenv("RAY_USAGE_STATS_REPORT_INTERVAL_S", "1")
        ray.init(num_cpus=0)
        # Wait enough so that usage report should happen.
        time.sleep(5)

        session_dir = ray.worker.global_worker.node.address_info["session_dir"]
        session_path = Path(session_dir)
        log_dir_path = session_path / "logs"

        paths = list(log_dir_path.iterdir())

        contents = None
        for path in paths:
            if "dashboard.log" in str(path):
                with open(str(path), "r") as f:
                    contents = f.readlines()
        assert contents is not None

        keyword_found = False
        for c in contents:
            if "Usage reporting is disabled" in c:
                keyword_found = True

        # Make sure the module was disabled.
        assert keyword_found

        for c in contents:
            assert "Failed to report usage stats" not in c


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
