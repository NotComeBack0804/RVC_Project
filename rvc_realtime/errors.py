"""用户可处理的错误，带稳定退出码。"""

class AppError(Exception):
    exit_code = 1

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ModelNotFoundError(AppError):
    exit_code = 2


class BaseModelNotFoundError(AppError):
    exit_code = 2


class MissingDependencyError(AppError):
    exit_code = 3


class AudioIOError(AppError):
    exit_code = 4


class UsageError(AppError):
    exit_code = 2
