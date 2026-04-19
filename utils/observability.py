import contextvars
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional


_run_id_ctx = contextvars.ContextVar("run_id", default=None)
_phase_ctx = contextvars.ContextVar("phase", default=None)
_node_ctx = contextvars.ContextVar("node", default=None)
_original_query_ctx = contextvars.ContextVar("original_query", default="")
_write_lock = threading.Lock()


def _ensure_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _ensure_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_ensure_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _ensure_jsonable(value.model_dump())
        except Exception:
            return str(value)
    if hasattr(value, "dict"):
        try:
            return _ensure_jsonable(value.dict())
        except Exception:
            return str(value)
    return str(value)


def _log_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    day = datetime.utcnow().strftime("%Y%m%d")
    return os.path.join(logs_dir, f"pipeline_{day}.jsonl")


def new_run_id() -> str:
    return str(uuid.uuid4())


def set_run_context(run_id: Optional[str], phase: Optional[str] = None, node: Optional[str] = None) -> None:
    _run_id_ctx.set(run_id)
    if phase is not None:
        _phase_ctx.set(phase)
    if node is not None:
        _node_ctx.set(node)


def get_run_id() -> Optional[str]:
    return _run_id_ctx.get()


def set_phase(phase: str) -> None:
    _phase_ctx.set(phase)


def set_node(node: str) -> None:
    _node_ctx.set(node)


def set_original_query(original_query: str) -> None:
    _original_query_ctx.set(original_query or "")


def get_original_query() -> str:
    return _original_query_ctx.get() or ""


def log_event(event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
    record = {
        "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
        "event_type": event_type,
        "run_id": get_run_id(),
        "phase": _phase_ctx.get(),
        "node": _node_ctx.get(),
    }
    if payload:
        record.update(_ensure_jsonable(payload))

    try:
        line = json.dumps(record, ensure_ascii=True)
        with _write_lock:
            with open(_log_path(), "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        # Logging must never break runtime behavior.
        return


@contextmanager
def node_span(node_name: str, input_payload: Optional[Dict[str, Any]] = None):
    prev_node = _node_ctx.get()
    set_node(node_name)
    started = time.perf_counter()
    log_event("node_started", {"node_name": node_name, "input": input_payload or {}})
    try:
        yield
    except Exception as exc:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        log_event(
            "node_error",
            {
                "node_name": node_name,
                "duration_ms": duration_ms,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    else:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        log_event("node_finished", {"node_name": node_name, "duration_ms": duration_ms})
    finally:
        _node_ctx.set(prev_node)


def log_node_output(node_name: str, output_payload: Dict[str, Any]) -> None:
    log_event("node_output", {"node_name": node_name, "output": output_payload})


def logged_chat_completion(
    client,
    *,
    node_name: str,
    model: str,
    messages: List[Dict[str, Any]],
    **kwargs,
):
    started = time.perf_counter()
    log_event(
        "llm_request",
        {
            "node_name": node_name,
            "model": model,
            "messages": messages,
            "params": kwargs,
        },
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )
    except Exception as exc:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        log_event(
            "llm_error",
            {
                "node_name": node_name,
                "model": model,
                "duration_ms": duration_ms,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    content = ""
    usage = {}
    try:
        content = response.choices[0].message.content
    except Exception:
        content = ""

    try:
        usage_obj = getattr(response, "usage", None)
        if usage_obj is not None:
            if hasattr(usage_obj, "model_dump"):
                usage = usage_obj.model_dump()
            elif hasattr(usage_obj, "dict"):
                usage = usage_obj.dict()
            else:
                usage = {"raw": str(usage_obj)}
    except Exception:
        usage = {}

    log_event(
        "llm_response",
        {
            "node_name": node_name,
            "model": model,
            "duration_ms": duration_ms,
            "output_text": content,
            "usage": usage,
        },
    )
    return response
