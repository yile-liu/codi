# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Ground-truth execution traces in CWM's frame format.

Runs ``f(input)`` under ``sys.settrace`` and records CALL/LINE/RETURN/EXCEPTION
frames with diff-based locals (unchanged vars render as ``".."``), values via
``repr``. A synthetic ``def main(): return f(<input>)`` wraps the function; the
seeded ``call main()`` frame is dropped by default to align with the trace
prompt. Not a bit-exact replica of Meta's internal tracer (see README.md).
"""

from __future__ import annotations

import linecache
import sys
from typing import Any

from .trace_format import DIFF_PLACEHOLDER, TraceEvent, TraceFrame, normalize_source

_FILENAME = "<cwm_trace>"
_ENTRY = "main"


class _FramesExceeded(Exception):
    pass


def make_trace_context(code: str, input_str: str) -> str:
    return f"\n{code}\ndef main():  # << START_OF_TRACE\n    return f({input_str})\n"


def render_value(value: Any) -> str:
    try:
        return repr(value)
    except Exception:
        return "<unrepr>"


def ground_truth_trace(
    code: str, input_str: str, align_to_prompt: bool = True, max_frames: int = -1
) -> tuple[list[TraceFrame], str | None]:
    """Return (frames, error) for executing ``f(input_str)``. ``error`` is
    non-None if the program raised; frames up to that point are still returned.
    """
    context = make_trace_context(code, input_str)
    linecache.cache[_FILENAME] = (len(context), None, context.splitlines(keepends=True), _FILENAME)

    frames: list[TraceFrame] = []
    scope_prev: dict[int, dict[str, str]] = {}  # id(frame) -> last rendered locals
    entry = None

    def source(frame):
        return normalize_source(linecache.getline(_FILENAME, frame.f_lineno))

    def diff_locals(frame):
        prev = scope_prev.get(id(frame), {})
        out, rendered = {}, {}
        for name, val in frame.f_locals.items():
            r = render_value(val)
            rendered[name] = r
            out[name] = DIFF_PLACEHOLDER if prev.get(name) == r else r
        scope_prev[id(frame)] = rendered
        return out

    def trace(frame, event, arg):
        nonlocal entry
        # Abort loop-heavy programs, but only from our file (not GC/__del__ frames).
        if max_frames > 0 and len(frames) >= max_frames and frame.f_code.co_filename == _FILENAME:
            raise _FramesExceeded
        if entry is None:
            if event == "call" and frame.f_code.co_name == _ENTRY:
                entry = id(frame)
            else:
                return None
        # Only trace user code from our context, not library frames.
        if frame.f_code.co_filename != _FILENAME:
            return None
        if event == "call":
            frames.append(TraceFrame(event=TraceEvent.CALL, source=source(frame), locals=diff_locals(frame)))
        elif event == "line":
            frames.append(TraceFrame(event=TraceEvent.LINE, source=source(frame), locals=diff_locals(frame)))
        elif event == "return":
            frames.append(TraceFrame(event=TraceEvent.RETURN, source=source(frame), arg=render_value(arg)))
        elif event == "exception":
            name = getattr(arg[0], "__name__", str(arg[0]))
            frames.append(TraceFrame(event=TraceEvent.EXCEPTION, source=source(frame), arg=render_value(name)))
        return trace

    ns: dict[str, Any] = {}
    exec(compile(context, _FILENAME, "exec"), ns)  # define f, main untraced
    error = None
    old = sys.gettrace()
    sys.settrace(trace)
    try:
        ns[_ENTRY]()
    except _FramesExceeded:
        error = "frames_exceeded"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    finally:
        sys.settrace(old)

    # Drop the seeded ``call main()`` frame so frames align with the prompt.
    if align_to_prompt and frames and frames[0].event == TraceEvent.CALL and frames[0].source.startswith("def main()"):
        frames = frames[1:]
    return frames, error
