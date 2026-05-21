"""A-RAG package initializer.

Expose core helpers at package level for convenience.
"""

from .agent_loop import run_agent  # re-export for convenience

__all__ = ["run_agent"]
