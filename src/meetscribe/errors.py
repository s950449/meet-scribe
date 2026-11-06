"""Expected failures. The CLI prints these without a traceback."""


class MeetscribeError(Exception):
    """Base class for errors the user can act on."""


class ConfigError(MeetscribeError):
    """The config file or a CLI override is invalid."""


class AudioError(MeetscribeError):
    """Audio could not be read, probed, or resampled."""


class UnknownBackend(MeetscribeError):
    """The requested backend name is not registered."""


class BackendNotInstalled(MeetscribeError):
    """A backend's optional dependency is missing."""


class BackendNotImplemented(MeetscribeError):
    """The backend is a documented stub. The dependency may already be installed."""


class LlmError(MeetscribeError):
    """The OpenAI-compatible endpoint failed or returned unusable output."""
