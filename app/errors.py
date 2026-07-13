"""Consistent API error responses."""

from typing import Any, Optional

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class APIErrorBody(BaseModel):
    error: str
    message: str
    detail: Optional[str] = None
    hint: Optional[str] = None
    fields: Optional[list[dict[str, Any]]] = None


def api_error(
    status_code: int,
    error: str,
    message: str,
    *,
    detail: Optional[str] = None,
    hint: Optional[str] = None,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "error": error,
            "message": message,
            "detail": detail or message,
            "hint": hint,
        },
    )


async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        body = exc.detail
    else:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        body = {
            "error": "request_error",
            "message": detail,
            "detail": detail,
            "hint": None,
        }
    return JSONResponse(status_code=exc.status_code, content=body)


async def validation_exception_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    fields = []
    for err in exc.errors():
        loc = " -> ".join(str(part) for part in err.get("loc", []))
        fields.append({"field": loc, "message": err.get("msg", "Invalid value")})

    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Request validation failed. Check the highlighted fields.",
            "detail": fields[0]["message"] if fields else "Invalid request body",
            "hint": "Review your form inputs and try again.",
            "fields": fields,
        },
    )
