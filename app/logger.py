import logging
import os
import queue
import threading
from datetime import datetime

# Create a 'logs' directory if it doesn't exist
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Generate a log file name with the current date and time
log_filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log'
log_filepath = os.path.join(log_dir, log_filename)

# Create a thread-safe queue for logging
log_queue = queue.Queue()

class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_queue.put(msg + '\n')
        except Exception:
            self.handleError(record)

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create formatter %(asctime)s 😅😅❤️ %(levelname)s ☠️☠️☠️ 
formatter = logging.Formatter('%(message)s')

# File handler
file_handler = logging.FileHandler(log_filepath)
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Queue handler for streaming
queue_handler = QueueHandler(log_queue)
queue_handler.setFormatter(formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.addHandler(queue_handler)

def get_logger():
    return logger
