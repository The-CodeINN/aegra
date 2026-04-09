"""React Agent.

This module defines a custom reasoning and action agent graph.
It invokes tools in a simple loop.
"""

__all__ = ["graph"]


def __getattr__(name: str):
    if name == "graph":
        from react_agent.graph import graph

        return graph
    raise AttributeError(name)
