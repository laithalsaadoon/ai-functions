"""ai_functions CLI — the ``ai-functions`` command installed by the package.

This subpackage holds the Typer app that powers the ``ai_functions`` console
script; :func:`main` is the entry point referenced from ``pyproject.toml``
under ``[project.scripts]``. :func:`print_event` prints the same event feed
the commands do, as a :meth:`~ai_functions.protocols.Coordinator.on`
subscriber for scripts.

Command surface (verbs modelled on ``docker`` / ``kubectl``):

- ``ai-functions server``         — start the coordinator endpoint in the
                             foreground; publish a runtime file; clean
                             up on exit.
- ``ai-functions server stop``    — read the runtime file, ask the server to
                             shut down.
- ``ai-functions server status``  — show the advertised URL, PID, uptime.
- ``ai-functions ps``             — list registered threads (one row per
                             :class:`ThreadInfo`).
- ``ai-functions attach <tid>``   — open the Textual TUI for a thread: live
                             event feed, input box for chat-shaped
                             threads (see :mod:`ai_functions.cli.attach`).
- ``ai-functions logs <tid>``     — non-TUI event dump; honours ``--follow``
                             and ``--since``.
- ``ai-functions notify <tid> <text>`` — one-shot ``notify`` (side channel,
                             no cycle).
- ``ai-functions submit <tid> <text>`` — ``submit`` one string prompt, block
                             until the cycle resolves, print the result
                             (``--json`` adds token usage and timing).
- ``ai-functions kill <tid>``     — ``terminate`` by default, ``--now`` for
                             ``terminate_now``.
- ``ai-functions run <script>``   — execute a user script whose module exposes
                             a main :class:`Spawnable` attribute (see
                             :func:`ai_functions.cli.commands.run_cmd`).

Every command reads the target coordinator URL from, in order:
``--url`` flag, ``AI_FUNCTIONS_COORDINATOR_URL`` env var, the runtime file.
"""

from __future__ import annotations

import sys
from typing import cast

import click

from .events import print_event

__all__ = ["main", "print_event"]


def _usage_error_types() -> tuple[type[click.ClickException], ...]:
    """Click ``ClickException`` classes to translate in :func:`main`.

    Typer may run on upstream ``click`` or on its own vendored copy
    (``typer._click``); the two define distinct exception classes, so a
    usage error raised by the app is an instance of one or the other.
    Collect both so the handler matches whichever Typer is installed. The
    vendored class is structurally a ``ClickException`` (it carries
    ``show`` and ``exit_code``), so the upstream type is accurate enough
    for the handler.
    """
    types: list[type[click.ClickException]] = [click.ClickException]
    try:
        from typer._click import exceptions as _vendored

        types.append(cast("type[click.ClickException]", _vendored.ClickException))
    except ImportError:
        pass
    return tuple(types)


def _abort_types() -> tuple[type[BaseException], ...]:
    """``Abort`` classes to translate in :func:`main`.

    ``typer.Abort`` is a ``RuntimeError``, not a ``click.Abort`` subclass, so
    both types are needed.
    """
    import typer

    return (click.Abort, typer.Abort)


def main() -> int:
    """CLI entry point installed as the ``ai_functions`` console script.

    Returns:
        Process exit code. ``0`` on success; non-zero for user errors,
        discovery failures, or command-specific failures. The concrete
        mapping is:

        - ``0`` — command succeeded.
        - ``1`` — generic user error (bad argument, unknown thread id).
        - ``2`` — no coordinator could be discovered.
        - ``130`` — interrupted by the user (Ctrl-C).

    Ensures:
        Never raises; exceptions are caught and translated into exit
        codes + stderr messages formatted for human readers.
    """
    from .app import app

    try:
        # With ``standalone_mode=False`` Click does not ``sys.exit`` itself;
        # a command that raises ``typer.Exit(code)`` is caught inside Click,
        # which then RETURNS the code rather than raising. (``typer.Exit`` is
        # a ``RuntimeError``, not ``SystemExit``, so it would not be caught
        # below.) Honour that return value as the process exit code; ``None``
        # means the command returned normally.
        rv = app(standalone_mode=False)
    except KeyboardInterrupt:
        return 130
    except _abort_types():
        # Raised by Click on Ctrl-C / EOF at a prompt; ``standalone_mode``
        # re-raises it instead of printing, so translate it here.
        click.echo("aborted", err=True)
        return 130
    except _usage_error_types() as exc:
        # Usage errors — missing / bad arguments, unknown options,
        # unknown commands. ``standalone_mode=False`` re-raises these
        # rather than showing them, which would otherwise surface as a
        # traceback. Show the friendly message and map to the contract's
        # generic user-error code.
        exc.show()
        return 1
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            return code
        return 0 if code is None else 1
    return rv if isinstance(rv, int) else 0


if __name__ == "__main__":
    sys.exit(main())
