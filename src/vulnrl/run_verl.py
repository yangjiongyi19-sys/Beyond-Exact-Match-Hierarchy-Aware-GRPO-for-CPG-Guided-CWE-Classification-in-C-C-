"""Register the CPG agent before starting VERL's standard GRPO entry point."""

from __future__ import annotations


def main() -> None:
    try:
        # Importing the loop registers it with VERL's experimental agent registry.
        from . import verl_agent_loop  # noqa: F401
        from verl.trainer.main_ppo import main as verl_main
    except ImportError as exc:
        raise RuntimeError("Install the project's train extra on the Linux/H100 cluster") from exc
    verl_main()


if __name__ == "__main__":
    main()

