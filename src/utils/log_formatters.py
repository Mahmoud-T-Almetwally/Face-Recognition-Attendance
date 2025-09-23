import logging
import csv
from io import StringIO

class CsvFormatter(logging.Formatter):
    """
    A custom logging formatter that outputs records in a CSV format.
    """
    def __init__(self, fields, header=True):
        """
        Initializes the formatter.

        Args:
            fields (list): A list of field names for the CSV columns.
            header (bool): Whether to write the header row.
        """
        super().__init__()
        self.fields = fields
        self.header = header
        self.header_written = not header

    def format(self, record):
        """
        Formats the log record into a CSV string.
        The data for the CSV is expected to be in the `record.csv_data` attribute.
        """
        # Use an in-memory string buffer to handle CSV writing
        string_io = StringIO()
        writer = csv.writer(string_io, quoting=csv.QUOTE_ALL)
        
        # Write header if it hasn't been written yet
        if not self.header_written:
            writer.writerow(self.fields)
            self.header_written = True

        # Ensure record.csv_data exists and is a dictionary
        if hasattr(record, 'csv_data') and isinstance(record.csv_data, dict):
            row = [record.csv_data.get(field, '') for field in self.fields]
            writer.writerow(row)
        
        return string_io.getvalue().strip()