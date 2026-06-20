
class CourtVisionError(Exception):
    """Base exception for all CourtVision-specific errors."""
    pass


class PlayerNotFoundError(CourtVisionError):
    """Raised when a player search returns no results from the API."""
    pass


class InvalidStatError(CourtVisionError):
    """Raised when a player has no entry in known_stats."""
    pass
