"""
Shared thread-safe state for cross-thread communication.

This module exists separately so that Python's import cache preserves
the Event objects across Streamlit script re-runs. Streamlit re-executes
the main script top-to-bottom on every interaction, but imported modules
are cached by Python and NOT re-executed.
"""

from threading import Event

# WebRTC callback thread sets these; main Streamlit thread reads them.
drowsy_event = Event()
yawn_event = Event()
