import logging.config
import sys
from log_formatters import CsvFormatter
from datetime import datetime

def setup_logging():
    """
    Configures logging for the entire application, including a dedicated
    CSV logger for attendance.
    """
    
    CSV_FIELDS = ['date', 'name', 'id', 'hall']

    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '[%(asctime)s] - [%(name)s] - %(levelname)s - %(message)s'
            },
            'csv': {
                '()': CsvFormatter,
                'fields': CSV_FIELDS
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
            'csv_file': {
                'class': 'logging.FileHandler',
                'formatter': 'csv',
                'filename': 'attendance.csv',
                'level': 'INFO',
            },
        },
        'loggers': {
            '': { 
                'handlers': ['console', 'file'],
                'level': 'INFO',
                'propagate': True
            },
            'attendance': {
                'handlers': ['csv_file'],
                'level': 'INFO',
                'propagate': False
            }
        }
    }
    logging.config.dictConfig(LOGGING_CONFIG)

def record_attendance(name, user_id, location):
    """
    Example function to record an attendance event.
    """
    attendance_data = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'name': name,
        'id': user_id,
        'hall': location
    }
    
    attendance_logger = logging.getLogger('attendance')

    attendance_logger.info(
        f"Attendance recorded for {name}", 
        extra={'csv_data': attendance_data}
    )
