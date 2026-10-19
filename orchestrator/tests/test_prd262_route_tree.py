"""PRD-262 US-008: the readers of FastAPI's route list see included routers.

Since FastAPI 0.137 an included router is one node of ``app.routes`` (no path, no
endpoint), and a routed request's ``scope["route"]`` is the route as its router
declared it, without the prefixes it was included under.

- ``core.route_tree.route_template`` gives the API stats (main.py) the full template,
  include prefixes and all, as ``scope["route"].path`` did before;
- ``core.route_tree.matching_endpoint`` finds the endpoint of a route of an included
  router, which slowapi's own lookup no longer does;
- so ``RouteTreeSlowAPIMiddleware`` applies the default limits to those routes, where
  slowapi's stock middleware exempts them. That last check pins slowapi's current
  behaviour: when it fails, slowapi sees the tree itself and the subclass can go.
"""
from __future__ import annotations

from fastapi import APIRouter, FastAPI, Request
from fastapi.testclient import TestClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from core.route_tree import matching_endpoint, route_template
from core.security.rate_limit_middleware import RouteTreeSlowAPIMiddleware


def _items_router() -> APIRouter:
    parent = APIRouter(prefix="/api/things")
    child = APIRouter()

    @child.get("/items/{item_id}")
    def read_item(item_id: int):
        return {"item": item_id}

    parent.include_router(child)
    return parent


def _app_recording_templates(seen: list) -> FastAPI:
    app = FastAPI()
    app.include_router(_items_router())
    twice = APIRouter()

    @twice.get("/ping")
    def ping():
        return {"pong": True}

    app.include_router(twice, prefix="/v1")
    app.include_router(twice, prefix="/v2")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.middleware("http")
    async def record(request: Request, call_next):
        response = await call_next(request)
        seen.append(route_template(request.scope))
        return response

    return app


def test_the_template_keeps_every_include_prefix():
    seen: list = []
    client = TestClient(_app_recording_templates(seen))
    for path in ("/api/things/items/7", "/v1/ping", "/v2/ping", "/health", "/nowhere"):
        client.get(path)
    assert seen == ["/api/things/items/{item_id}", "/v1/ping", "/v2/ping", "/health", None]


def test_the_endpoint_of_an_included_route_is_found():
    app = FastAPI()
    app.include_router(_items_router())
    scope = {"type": "http", "method": "GET", "path": "/api/things/items/7", "root_path": ""}
    endpoint = matching_endpoint(app.routes, scope)
    assert endpoint is not None and endpoint.__name__ == "read_item"
    assert matching_endpoint(app.routes, {**scope, "path": "/api/things/elsewhere"}) is None


def _limited_app(middleware) -> TestClient:
    app = FastAPI()
    app.include_router(_items_router())
    app.state.limiter = Limiter(key_func=lambda request: "one-caller", default_limits=["1/minute"])
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(middleware)
    return TestClient(app)


def test_the_default_limit_applies_to_a_route_of_an_included_router():
    client = _limited_app(RouteTreeSlowAPIMiddleware)
    assert client.get("/api/things/items/7").status_code == 200
    assert client.get("/api/things/items/7").status_code == 429


def test_slowapis_own_middleware_still_exempts_it():
    client = _limited_app(SlowAPIMiddleware)
    assert [client.get("/api/things/items/7").status_code for _ in range(3)] == [200, 200, 200]
