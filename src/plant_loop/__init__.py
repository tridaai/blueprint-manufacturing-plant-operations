"""Trida AI Blueprint — Plant Operations Loop Agent.

Two nested agent loops for a manufacturing shift's exception board:
an outer board loop that works cases in code-computed priority
order, and a per-case diagnose → plan → verify loop that only stops
at a human supervisor's gate. Powered by Muse Spark (Meta Model API)
behind a replaceable reasoning port.
"""

__version__ = "0.1.0"
