class StoneColorAppException(Exception):
    """Base exception for the Stone Preprocessing application."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class DownloadError(StoneColorAppException):
    """Raised when the image downloader fails."""
    pass


class InvalidImageError(StoneColorAppException):
    """Raised when an image is corrupted or invalid."""
    pass


class SlabDetectionError(StoneColorAppException):
    """Raised when stone slab detection or cropping fails."""
    pass


class MessagingError(StoneColorAppException):
    """Raised when internal messaging via ZeroMQ fails."""
    pass


class ConfigurationError(StoneColorAppException):
    """Raised when application configuration is invalid."""
    pass
