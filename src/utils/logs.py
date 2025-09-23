import logging.config
import sys


def setup_logging():
    """
    Configures logging for the entire application.
    """
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '[%(asctime)s] - [%(name)s] - %(levelname)s - %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'level': 'DEBUG',
                'stream': sys.stdout,
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'standard',
                'filename': 'app.log',
                'maxBytes': 10485760,  
                'backupCount': 5,
                'level': 'INFO',
            },
        },
        'loggers': {
            '': { 
                'handlers': ['console', 'file'],
                'level': 'INFO',
                'propagate': True
            },
            'scrapy': {
                'handlers': ['console'],
                'level': 'ERROR',
                'propagate': False
            }
        }
    }
    logging.config.dictConfig(LOGGING_CONFIG)


def save_attendance_log():
    """
    Saves Student Attendance Logs
    """
    pass