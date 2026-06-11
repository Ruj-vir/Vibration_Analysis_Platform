"""
auth_interceptor.py — Middleware แนบ token ทุก API request อัตโนมัติ

หน้าที่:
  - ดึง token จาก session
  - แนบ token ไว้ใน request.state.token ให้ทุก route ใช้ได้
  - ไม่บล็อก request ถ้าไม่มี token (auth_guard จัดการแทน)

วิธีใช้ใน main.py:
  from auth_interceptor import AuthInterceptor
  app.add_middleware(AuthInterceptor)
"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class AuthInterceptor(BaseHTTPMiddleware):
    """Middleware — แนบ token จาก session เข้า request.state.token"""

    async def dispatch(self, request: Request, call_next) -> Response:
        token: str | None = None
        if "session" in request.scope:
            token = request.session.get("token")

        request.state.token = token

        response = await call_next(request)
        return response
