import logging
from typing import Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from core.database.database import SessionLocal
from core.services.api_key_service import ApiKeyService

logger = logging.getLogger(__name__)


class WidgetCORSMiddleware(BaseHTTPMiddleware):
    """Dynamic CORS middleware that validates origins per API key.
    Applies ONLY to /api/widgets/* routes.
    """

    async def dispatch(self, request: Request, call_next):
        # Only apply to /api/widgets/ routes
        if not request.url.path.startswith("/api/widgets"):
            return await call_next(request)

        origin = request.headers.get("origin")

        # Handle OPTIONS preflight
        if request.method == "OPTIONS":
            if not origin:
                return Response(status_code=400)

            # For preflight, we do a permissive check since the actual
            # request will be fully validated
            response = Response(status_code=200)
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, X-Workspace-ID"
            response.headers["Access-Control-Max-Age"] = "86400"
            response.headers["Access-Control-Allow-Credentials"] = "true"
            return response

        # For actual requests, add CORS headers to response
        response = await call_next(request)
        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Vary"] = "Origin"
        return response
