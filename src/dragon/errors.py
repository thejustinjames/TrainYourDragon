"""One exception type, so the CLI can fail with a message rather than a traceback."""


class DragonError(RuntimeError):
    """Something the user can fix: a missing path, a bad config, a tool not installed."""
