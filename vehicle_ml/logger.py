import logging
import logging.config


class ImportantFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "[IMPORTANT]" in record.getMessage()


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "default": {"format": "%(asctime)s %(levelname)8s | %(message)s"},
        "app": {
            "format": "%(asctime)s %(levelname)8s > %(message)s  (%(filename)s:%(lineno)s)",
        },
        "console": {"format": "[%(asctime)s] [%(levelname)s] %(message)s"},
        "important": {"format": "[%(asctime)s] %(message)s"},
    },
    "filters": {
        "important_filter": {
            "()": ImportantFilter,
        }
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "console",
            "stream": "ext://sys.stdout",
        },
        "important_handler": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "formatter": "important",
            "filters": ["important_filter"],
            "filename": "important.log",
        },
        "file_handler": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": "app.log",
            "mode": "w",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file_handler", "important_handler"]},
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("app")
