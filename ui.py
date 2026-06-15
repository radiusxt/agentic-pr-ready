#!/usr/bin/env python3
"""Terminal UI for the PR merge monitor agent."""

from __future__ import annotations

import argparse
import json
import sys
from agent import Agent
from clients import MonitorConfig, ROOT_DIR, ensure_logs_dir
from pathlib import Path


def _load_config(args: argparse.Namespace) -> MonitorConfig:
    """
    Resolve and load the MonitorConfig from the path stored in parsed args.
    A thin convenience wrapper so every command function can load config in 1 line without repeating args.config attribute access.
    The actual validation and parsing is handled by MonitorConfig.load().
 
    Args:
        args: Parsed argument namespace. Must contain a `config` attribute holding a Path to the YAML config file, as added by build_parser().
 
    Returns:
        A validated MonitorConfig instance.
 
    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file contains a placeholder or invalid repo/PR value.
    """
    return MonitorConfig.load(args.config)


def cmd_status(args: argparse.Namespace) -> int:
    """
    Observation-only command: fetch and print the current PR state.
    Instantiates the agent with use_llm=False so no Anthropic API call is made but rather a pure read against the GitHub CLI.
    Useful for checking the PR state without triggering any agent actions or incurring API costs.
    Output is a single JSON object containing both the status and comments payloads in stdout.
 
    Args:
        args: Parsed argument namespace. Uses `args.config` via _load_config. Doesn't use cwd, docker, or no_llm (always no-LLM).
 
    Returns:
        0: An observation cannot meaningfully "fail" from the CLI's perspective. Any underlying errors will raise before this returns.
    """
    config = _load_config(args)
    agent = Agent(config, use_llm=False)
    status, comments = agent.observe()
    print(json.dumps({"status": status, "comments": comments}, indent=2))
    return 0


def cmd_once(args: argparse.Namespace) -> int:
    """
    Run a single agent iteration and report the outcome.
    Constructs a fully configured Agent (with optional LLM and Docker support),
    runs one observe → act → evaluate cycle via run_once() and prints result.
    Intended for one-shot use in scripts or as a manual trigger in CI pipelines.

    Args:
        args: Parsed argument namespace with attributes (config, cwd, no_llm, docker).
 
    Returns:
        0: PR is merge-ready or iteration made progress (status != "blocked").
        2: Iteration ended in a blocked state requiring human intervention before continuing further.
    """
    config = _load_config(args)
    agent = Agent(
        config,
        cwd=Path(args.cwd).resolve() if args.cwd else Path.cwd(),
        use_llm=not args.no_llm,
        use_docker=args.docker,
    )
    result = agent.run_once()
    agent._print_result(result)
    return 0 if result.status != "blocked" else 2


def cmd_loop(args: argparse.Namespace) -> int:
    """
    Run the agent continuously until a terminal condition is reached.
    Delegates entirely to Agent.run_loop() which handles the polling interval, dynamic vs fixed mode and
    all three terminal conditions (merge-ready, max iterations reached, human intervention required).
    This command blocks until one of those conditions fires.
 
    Args:
        args: Parsed argument namespace with attributes (config, cwd, no_llm, docker, mode, max_iterations).
 
    Returns:
        0: Terminal conditions are reported via _print_result output rather than exit codes at the loop level.
                Use cmd_once if you need shell-script-friendly exit codes.
    """
    config = _load_config(args)
    agent = Agent(
        config,
        cwd=Path(args.cwd).resolve() if args.cwd else Path.cwd(),
        use_llm=not args.no_llm,
        use_docker=args.docker,
    )
    agent.run_loop(mode=args.mode, max_iterations=args.max_iterations)
    return 0


def cmd_repl(args: argparse.Namespace) -> int:
    """
    Start an interactive terminal session with the agent.
    Presents a prompt loop accepting a small set of commands, allowing the operator to
    repeatedly query and drive the agent without restarting the process between actions.
    This is useful for hands-on debugging or supervised monitoring where the operator checks state, triggers single iterations and
    optionally starts a full loop all within one session sharing a single Agent instance (and its accumulated state and history).
    The REPL exits cleanly on EOF (Ctrl-D) and KeyboardInterrupt (Ctrl-C) in addition to explicit quit commands
    so it behaves naturally in both interactive terminals and piped input scenarios.
 
    Supported REPL commands:
        status: Observe the PR and print status + comments JSON.
        once: Run one agent iteration and print the result.
        loop [fixed | dynamic]: Start the continuous loop. Blocks until a terminal condition fires.
        quit / exit / q: Exit the REPL cleanly.
 
    Args:
        args: Parsed argument namespace with attributes (config, cwd, no_llm, docker, max_iterations).
                Note: mode is not a global flag — it can be passed inline as `loop fixed` or `loop dynamic`.
 
    Returns: 0
    """
    config = _load_config(args)
    agent = Agent(
        config,
        cwd=Path(args.cwd).resolve() if args.cwd else Path.cwd(),
        use_llm=not args.no_llm,
        use_docker=args.docker,
    )

    print("PR monitor REPL. Commands: status | once | loop | quit")

    while True:
        try:
            line = input("monitor> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line in {"quit", "exit", "q"}:
            break

        if line == "status":
            status, comments = agent.observe()
            print(json.dumps({"status": status, "comments": comments}, indent=2))
            continue

        if line == "once":
            agent._print_result(agent.run_once())
            continue

        # Optional mode after "loop", e.g. "loop dynamic". Falls back to None letting Agent.run_loop() read mode from config.
        if line.startswith("loop"):
            parts = line.split()
            mode = parts[1] if len(parts) > 1 else None
            agent.run_loop(mode=mode, max_iterations=args.max_iterations)
            continue

        print("Unknown command. Try: status | once | loop [fixed|dynamic] | quit")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """
    Construct and return the CLI argument parser for the monitor agent.
    Defines a top-level parser with shared flags common to all subcommands and
    4 subcommands each mapping to a cmd_* function via set_defaults(func=).
    The shared flags are attached to the top-level parser so they're available to every subcommand without repetition.
 
    Shared flags (available on all subcommands):
        --config PATH: Path to the YAML config file. Defaults to config/active.yaml relative to the project root.
        --cwd DIR: Working directory for tool execution (target repo). Defaults to the current process directory.
        --no-llm: Disable LLM calls; observation-only mode for monitoring.
        --docker: Enable the docker_exec tool (requires Docker).
        --max-iterations N: Cap the number of loop iterations.
 
    Subcommands:
        status: cmd_status (no additional arguments)
        once: cmd_once (no additional arguments)
        loop: cmd_loop (--mode fixed | dynamic)
        repl: cmd_repl (no additional arguments)
 
    Returns:
        A configured ArgumentParser ready to call parse_args() on.
    """
    parser = argparse.ArgumentParser(description="PR merge monitor agent")

    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "config" / "active.yaml",
        help="Path to active config",
    )
    parser.add_argument("--cwd", help="Target application repo working directory")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Observation-only mode (no Anthropic API calls)",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Register docker_exec tool (requires Docker)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Stop loop after N iterations",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="Fetch PR status and comments once")

    once = sub.add_parser("once", help="Run one monitor iteration")
    once.set_defaults(func=cmd_once)

    loop = sub.add_parser("loop", help="Run until merge-ready or blocked")
    loop.add_argument("--mode", choices=["fixed", "dynamic"], default=None)
    loop.set_defaults(func=cmd_loop)

    repl = sub.add_parser("repl", help="Interactive terminal session")
    repl.set_defaults(func=cmd_repl)

    return parser


def main() -> int:
    """
    Entry point for the CLI: parse arguments and dispatch to the appropriate command.
    Ensures the logs directory exists before any command runs so the tools which write log files never need to handle a missing directory mid-run.
    Dispatches to the correct cmd_* function via the `func` attribute attached to each subparser by set_defaults(),
    a standard argparse pattern that avoids a manual if/elif chain over subcommand names.
 
    Returns:
        The integer exit code returned by the dispatched command function passed directly to sys.exit() by the __main__ block.
    """
    ensure_logs_dir()
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    """
    Run main() and forward its return value as the process exit code so
    shell scripts and CI systems can detect blocked/failed runs via $?.
    """
    sys.exit(main())
