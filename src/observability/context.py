from contextvars import ContextVar

current_run_id: ContextVar[str | None] = ContextVar("current_run_id", default=None)
current_thread_id: ContextVar[str | None] = ContextVar(
    "current_thread_id", default=None
)
