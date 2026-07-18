"""Compact, consistent terminal output for project scripts."""

from __future__ import annotations

import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO


_EPOCH_PATTERN = re.compile(r"(?:^|\s)epoch=(\d+)(?:\s|$)")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return "nan" if value != value else f"{value:.6g}"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        preview = ", ".join(str(item) for item in items[:5])
        suffix = f", +{len(items) - 5}" if len(items) > 5 else ""
        return f"[{preview}{suffix}]"
    return str(value)


class TerminalConsole:
    """Small output facade with safe defaults for long experiment runs."""

    def __init__(
        self,
        *,
        verbose: bool | None = None,
        max_lines: int | None = None,
        max_width: int | None = None,
        epoch_every: int | None = None,
    ) -> None:
        self.verbose = _env_flag("PAPERV4_VERBOSE") if verbose is None else verbose
        self.max_lines = max_lines or int(os.getenv("PAPERV4_TERMINAL_MAX_LINES", "10"))
        self.max_width = max_width or int(os.getenv("PAPERV4_TERMINAL_WIDTH", "160"))
        self.epoch_every = epoch_every or int(os.getenv("PAPERV4_EPOCH_EVERY", "10"))
        self._mute_depth = 0

    @contextmanager
    def muted(self) -> Iterator[None]:
        """Temporarily silence nested runs while their caller prints a summary."""
        self._mute_depth += 1
        try:
            yield
        finally:
            self._mute_depth -= 1

    def _should_skip_epoch(self, text: str) -> bool:
        if self.verbose or text.startswith("  ") or "early_stop" in text or "best_epoch" in text:
            return False
        match = _EPOCH_PATTERN.search(text)
        if match is None:
            return False
        epoch = int(match.group(1))
        return epoch != 1 and epoch % self.epoch_every != 0

    def _compact_json(self, text: str) -> str | None:
        stripped = text.strip()
        if not stripped.startswith(("{", "[")):
            return None
        try:
            value = json.loads(stripped)
        except (TypeError, ValueError):
            return None
        if isinstance(value, dict):
            items = list(value.items())
            preview = " | ".join(f"{key}={_format_value(item)}" for key, item in items[:6])
            suffix = f" | +{len(items) - 6} fields" if len(items) > 6 else ""
            return f"config | {preview}{suffix}"
        if isinstance(value, list):
            return f"list | items={len(value)}"
        return None

    def _compact_text(self, text: str) -> str:
        if self.verbose:
            return text
        json_preview = self._compact_json(text)
        if json_preview is not None:
            text = json_preview
        lines = text.splitlines() or [""]
        clipped = [line if len(line) <= self.max_width else f"{line[: self.max_width - 3]}..." for line in lines]
        if len(clipped) > self.max_lines:
            hidden = len(clipped) - self.max_lines
            clipped = [*clipped[: self.max_lines], f"... omitted {hidden} lines; see saved result files"]
        return "\n".join(clipped)

    def print(
        self,
        *values: Any,
        sep: str = " ",
        end: str = "\n",
        file: TextIO | None = None,
        flush: bool = False,
    ) -> None:
        """Drop-in replacement for print with bounded multiline output."""
        if self._mute_depth:
            return
        text = sep.join(str(value) for value in values)
        if self._should_skip_epoch(text):
            return
        output = self._compact_text(text)
        stream = file or sys.stdout
        stream.write(output)
        stream.write(end)
        if flush:
            stream.flush()

    def phase(self, title: str, *, current: int | None = None, total: int | None = None) -> None:
        prefix = f"[{current}/{total}]" if current is not None and total is not None else "[stage]"
        self.print(f"\n{prefix} {title}", flush=True)

    def info(self, label: str, **fields: Any) -> None:
        details = " | ".join(f"{key}={_format_value(value)}" for key, value in fields.items())
        self.print(f"  {label}" + (f" | {details}" if details else ""), flush=True)

    def model_result(
        self,
        target: str,
        *,
        best_epoch: int,
        val_rmse: float,
        test_rmse: float,
    ) -> None:
        self.info(target, epoch=best_epoch, val_rmse=val_rmse, test_rmse=test_rmse)

    def table(
        self,
        title: str,
        frame: Any,
        *,
        columns: tuple[str, ...] | list[str] | None = None,
        max_rows: int = 8,
        index: bool = False,
    ) -> None:
        """Print a bounded DataFrame-like preview without affecting saved tables."""
        self.phase(title)
        display = frame
        if columns is not None:
            available = [column for column in columns if column in display.columns]
            display = display.loc[:, available]
        total_rows = len(display)
        preview = display if self.verbose else display.head(max_rows)
        self.print(preview.to_string(index=index), flush=True)
        if not self.verbose and total_rows > len(preview):
            self.info("terminal preview", shown=len(preview), total=total_rows)

    def done(self, output: str | Path, **fields: Any) -> None:
        self.phase("completed")
        self.info("result", output=Path(output), **fields)


console = TerminalConsole()
