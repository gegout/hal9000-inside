"""Deterministic fake clock for cognitive scheduling and testing."""

import time


class FakeClock:
    """Controllable monotonic and real-time clock."""

    def __init__(self, initial_time: float = 1700000000.0):
        self._current_time = initial_time

    def time(self) -> float:
        """Returns current simulated UNIX timestamp in seconds."""
        return self._current_time

    def monotonic(self) -> float:
        """Returns current simulated monotonic time in seconds."""
        return self._current_time

    def advance(self, seconds: float):
        """Advance simulated clock by given seconds."""
        if seconds < 0:
            raise ValueError("Clock cannot travel backward in time")
        self._current_time += seconds

    def sleep(self, seconds: float):
        """Advance simulated clock without actual wall-clock delay."""
        self.advance(seconds)
