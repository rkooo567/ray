import uuid

import pytest
import requests
import starlette
from fastapi import FastAPI

import ray
from ray import serve
from ray.serve._private.constants import RAY_SERVE_REQUEST_ID_HEADER


def test_request_id_header_by_default(serve_instance):
    """Test that a request_id is generated by default and returned as a header."""

    @serve.deployment
    class Model:
        def __call__(self):
            request_id = ray.serve.context._serve_request_context.get().request_id
            return request_id

    serve.run(Model.bind())
    resp = requests.get("http://localhost:8000")
    assert resp.status_code == 200
    assert RAY_SERVE_REQUEST_ID_HEADER in resp.headers
    assert resp.text == resp.headers[RAY_SERVE_REQUEST_ID_HEADER]
    assert resp.text == resp.headers["x-request-id"]

    def is_valid_uuid(num: str):
        try:
            uuid.UUID(num, version=4)
            return True
        except ValueError:
            return False

    assert is_valid_uuid(resp.text)


class TestUserProvidedRequestIDHeader:
    def verify_result(self):
        for header_attr in [RAY_SERVE_REQUEST_ID_HEADER, "X-Request-ID"]:
            resp = requests.get(
                "http://localhost:8000", headers={header_attr: "123-234"}
            )
            assert resp.status_code == 200
            assert resp.json() == 1
            assert resp.headers[header_attr] == "123-234"

    def test_basic(self, serve_instance):
        @serve.deployment
        class Model:
            def __call__(self) -> int:
                request_id = ray.serve.context._serve_request_context.get().request_id
                assert request_id == "123-234"
                return 1

        serve.run(Model.bind())
        self.verify_result()

    def test_fastapi(self, serve_instance):
        app = FastAPI()

        @serve.deployment
        @serve.ingress(app)
        class Model:
            @app.get("/")
            def say_hi(self) -> int:
                request_id = ray.serve.context._serve_request_context.get().request_id
                assert request_id == "123-234"
                return 1

        serve.run(Model.bind())
        self.verify_result()

    def test_starlette_resp(self, serve_instance):
        @serve.deployment
        class Model:
            def __call__(self) -> int:
                request_id = ray.serve.context._serve_request_context.get().request_id
                assert request_id == "123-234"
                return starlette.responses.Response("1", media_type="application/json")

        serve.run(Model.bind())
        self.verify_result()


def test_set_request_id_headers_with_two_attributes(serve_instance):
    """Test that request id is set with X-Request-ID and RAY_SERVE_REQUEST_ID.
    x-request-id has higher priority.
    """

    @serve.deployment
    class Model:
        def __call__(self):
            request_id = ray.serve.context._serve_request_context.get().request_id
            return request_id

    serve.run(Model.bind())
    resp = requests.get(
        "http://localhost:8000",
        headers={
            RAY_SERVE_REQUEST_ID_HEADER: "123",
            "X-Request-ID": "234",
        },
    )

    assert resp.status_code == 200
    assert "x-request-id" in resp.headers
    assert resp.text == resp.headers["x-request-id"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", "-s", __file__]))
