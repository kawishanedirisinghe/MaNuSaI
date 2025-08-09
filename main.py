# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import mimetypes
import os
import time
from pathlib import Path
import asyncio
import queue
from app.agent.manus import Manus
from app.logger import logger, log_queue
from app.config import config as app_config
import threading
import toml
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
import uuid
from werkzeug.utils import secure_filename
import shutil

app = Flask(__name__)
app.config['WORKSPACE'] = 'workspace'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CHAT_HISTORY_FILE'] = 'chat_history.json'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs(app.config['WORKSPACE'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable to track running tasks
running_tasks = {}

# Load configuration
config = toml.load('config/config.toml')

# Advanced API Key Management System
class AdvancedAPIKeyManager:
    def __init__(self, api_keys_config, usage_file='api_key_usage.json'):
        self.api_keys = []
        self.usage_stats = {}
        self.disabled_keys = {}  # {key: disabled_until_timestamp}
        self.failure_counts = {}  # {key: consecutive_failures}
        self.last_used = {}  # {key: last_used_timestamp}
        self.usage_file = usage_file

        # Load existing usage data
        self._load_usage_data()

        # Initialize API keys from config
        for key_config in api_keys_config:
            self.api_keys.append({
                'api_key': key_config['api_key'],
                'name': key_config.get('name', f"Key_{key_config['api_key'][:8]}"),
                'max_requests_per_minute': key_config.get('max_requests_per_minute', 5),
                'max_requests_per_hour': key_config.get('max_requests_per_hour', 100),
                'max_requests_per_day': key_config.get('max_requests_per_day', 100),
                'priority': key_config.get('priority', 1),
                'enabled': key_config.get('enabled', True)
            })

            # Initialize stats for each key if not already loaded
            key = key_config['api_key']
            if key not in self.usage_stats:
                self.usage_stats[key] = {
                    'requests_this_minute': [],
                    'requests_this_hour': [],
                    'requests_this_day': [],
                    'total_requests': 0
                }
            if key not in self.failure_counts:
                self.failure_counts[key] = 0
            if key not in self.last_used:
                self.last_used[key] = None

        logger.info(f"Initialized advanced API key manager with {len(self.api_keys)} keys")

    def _load_usage_data(self):
        """Load usage data from JSON file"""
        try:
            if os.path.exists(self.usage_file):
                with open(self.usage_file, 'r') as f:
                    data = json.load(f)
                    self.usage_stats = data.get('usage_stats', self.usage_stats)
                    self.disabled_keys = data.get('disabled_keys', self.disabled_keys)
                    self.failure_counts = data.get('failure_counts', self.failure_counts)
                    self.last_used = data.get('last_used', self.last_used)
                logger.info(f"Loaded API key usage data from {self.usage_file}")
        except Exception as e:
            logger.error(f"Error loading API key usage data: {e}")

    def _save_usage_data(self):
        """Save usage data to JSON file"""
        try:
            data = {
                'usage_stats': self.usage_stats,
                'disabled_keys': self.disabled_keys,
                'failure_counts': self.failure_counts,
                'last_used': self.last_used
            }
            with open(self.usage_file, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving API key usage data: {e}")

    def _clean_old_usage_data(self, api_key: str):
        """Clean old usage data for accurate rate limiting"""
        current_time = time.time()
        stats = self.usage_stats[api_key]

        # Clean minute data (older than 60 seconds)
        stats['requests_this_minute'] = [
            t for t in stats['requests_this_minute'] 
            if current_time - t < 60
        ]

        # Clean hour data (older than 3600 seconds)
        stats['requests_this_hour'] = [
            t for t in stats['requests_this_hour'] 
            if current_time - t < 3600
        ]

        # Clean day data (older than 86400 seconds)
        stats['requests_this_day'] = [
            t for t in stats['requests_this_day'] 
            if current_time - t < 86400
        ]

    def _is_key_available(self, key_config: dict) -> bool:
        """Check if an API key is available for use"""
        api_key = key_config['api_key']
        current_time = time.time()

        # Check if key is disabled
        if not key_config['enabled']:
            return False

        # Check if key is in cooldown period (24 hours after rate limit)
        if api_key in self.disabled_keys:
            if current_time < self.disabled_keys[api_key]:
                logger.debug(f"Key {key_config['name']} still in cooldown")
                return False
            else:
                # Cooldown expired, remove from disabled list
                del self.disabled_keys[api_key]
                self.failure_counts[api_key] = 0  # Reset failure count
                logger.info(f"Key {key_config['name']} cooldown expired, re-enabling")

        # Clean old usage data
        self._clean_old_usage_data(api_key)

        # Check rate limits
        stats = self.usage_stats[api_key]

        if len(stats['requests_this_minute']) >= key_config['max_requests_per_minute']:
            logger.debug(f"Key {key_config['name']} hit minute limit")
            return False

        if len(stats['requests_this_hour']) >= key_config['max_requests_per_hour']:
            logger.debug(f"Key {key_config['name']} hit hour limit")
            return False

        if len(stats['requests_this_day']) >= key_config['max_requests_per_day']:
            logger.debug(f"Key {key_config['name']} hit daily limit")
            # Disable key for 24 hours
            self._disable_key_for_rate_limit(api_key, key_config['name'])
            return False

        return True

    def _disable_key_for_rate_limit(self, api_key: str, key_name: str):
        """Disable API key for 24 hours due to rate limit"""
        disable_until = time.time() + 24 * 60 * 60  # 24 hours
        self.disabled_keys[api_key] = disable_until

        logger.warning(f"API key {key_name} disabled for 24 hours due to rate limit")

    def _calculate_key_score(self, key_config: dict) -> float:
        """Calculate a score for key selection (higher is better)"""
        api_key = key_config['api_key']
        current_time = time.time()

        # Base score from priority (lower priority number = higher score)
        priority_score = 10.0 / max(key_config['priority'], 1)

        # Usage-based score (less recent usage = higher score)
        stats = self.usage_stats[api_key]
        minute_usage = len(stats['requests_this_minute'])
        hour_usage = len(stats['requests_this_hour'])
        day_usage = len(stats['requests_this_day'])

        # Calculate remaining capacity
        minute_capacity = 1.0 - (minute_usage / key_config['max_requests_per_minute'])
        hour_capacity = 1.0 - (hour_usage / key_config['max_requests_per_hour'])
        day_capacity = 1.0 - (day_usage / key_config['max_requests_per_day'])

        capacity_score = (minute_capacity + hour_capacity + day_capacity) / 3

        # Failure-based score (fewer failures = higher score)
        failure_score = 1.0 / (self.failure_counts[api_key] + 1)

        # Time since last use (longer = slightly higher score)
        time_score = 1.0
        if self.last_used[api_key]:
            time_since_use = current_time - self.last_used[api_key]
            time_score = min(1.0 + (time_since_use / 3600), 2.0)  # Max 2x after 1 hour

        # Combine all factors
        final_score = priority_score * capacity_score * failure_score * time_score
        return max(final_score, 0.1)  # Minimum score

    def get_available_api_key(self, use_random: bool = True) -> Optional[Tuple[str, dict]]:
        """Get an available API key with advanced selection logic"""
        available_keys = []

        # Find all available keys
        for key_config in self.api_keys:
            if self._is_key_available(key_config):
                available_keys.append(key_config)

        if not available_keys:
            logger.warning("No API keys available")
            return None

        if use_random and len(available_keys) > 1:
            # Advanced weighted random selection
            weights = []
            for key_config in available_keys:
                score = self._calculate_key_score(key_config)
                weights.append(score)

            # Weighted random choice
            selected_key = random.choices(available_keys, weights=weights)[0]
            logger.info(f"Randomly selected API key: {selected_key['name']} (weighted selection)")
        else:
            # Priority-based selection with health metrics
            available_keys.sort(key=lambda k: (
                k['priority'],
                -self._calculate_key_score(k),
                self.failure_counts[k['api_key']],
                k['api_key']  # Deterministic tie-breaker
            ))
            selected_key = available_keys[0]
            logger.info(f"Priority selected API key: {selected_key['name']}")

        return selected_key['api_key'], selected_key

    def record_successful_request(self, api_key: str):
        """Record a successful API request"""
        current_time = time.time()
        stats = self.usage_stats[api_key]

        # Add timestamps
        stats['requests_this_minute'].append(current_time)
        stats['requests_this_hour'].append(current_time)
        stats['requests_this_day'].append(current_time)
        stats['total_requests'] += 1

        # Update last used time
        self.last_used[api_key] = current_time

        # Reset failure count on success
        self.failure_counts[api_key] = 0

        logger.info(f"Recorded successful request for API key")
        self._save_usage_data()

    def record_rate_limit_error(self, api_key: str, key_name: str):
        """Record a rate limit error and disable the key"""
        self._disable_key_for_rate_limit(api_key, key_name)
        self.failure_counts[api_key] += 1
        logger.warning(f"Rate limit error recorded for {key_name}")
        self._save_usage_data()

    def record_failure(self, api_key: str, key_name: str, error_type: str = "unknown"):
        """Record a failure for an API key"""
        self.failure_counts[api_key] += 1
        logger.warning(f"Failure recorded for {key_name}: {error_type} (consecutive: {self.failure_counts[api_key]})")

        # If too many consecutive failures, disable temporarily
        if self.failure_counts[api_key] >= 5:
            # Exponential backoff: 5 minutes * 2^(failures-5)
            backoff_minutes = 5 * (2 ** (self.failure_counts[api_key] - 5))
            backoff_minutes = min(backoff_minutes, 240)  # Cap at 4 hours

            disable_until = time.time() + (backoff_minutes * 60)
            self.disabled_keys[api_key] = disable_until
            logger.warning(f"Temporarily disabled {key_name} for {backoff_minutes} minutes due to failures")

    def get_keys_status(self) -> List[Dict]:
        """Get detailed status of all API keys"""
        status_list = []
        current_time = time.time()

        for key_config in self.api_keys:
            api_key = key_config['api_key']
            self._clean_old_usage_data(api_key)

            stats = self.usage_stats[api_key]
            is_available = self._is_key_available(key_config)

            status = {
                'name': key_config['name'],
                'enabled': key_config['enabled'],
                'available': is_available,
                'usage': {
                    'requests_this_minute': len(stats['requests_this_minute']),
                    'requests_this_hour': len(stats['requests_this_hour']),
                    'requests_this_day': len(stats['requests_this_day']),
                    'total_requests': stats['total_requests']
                },
                'limits': {
                    'max_per_minute': key_config['max_requests_per_minute'],
                    'max_per_hour': key_config['max_requests_per_hour'],
                    'max_per_day': key_config['max_requests_per_day']
                },
                'failures': self.failure_counts[api_key],
                'last_used': datetime.fromtimestamp(self.last_used[api_key]).isoformat() if self.last_used[api_key] else "Never"
            }

            # Add cooldown info if applicable
            if api_key in self.disabled_keys:
                remaining_time = int(self.disabled_keys[api_key] - current_time)
                if remaining_time > 0:
                    status['cooldown_remaining_seconds'] = remaining_time
                    status['cooldown_remaining_readable'] = f"{remaining_time // 3600}h {(remaining_time % 3600) // 60}m"

            status_list.append(status)

        return status_list

# Initialize the advanced API key manager
api_key_manager = AdvancedAPIKeyManager(config['llm']['api_keys'])

# 初始化工作目录
os.makedirs(app.config['WORKSPACE'], exist_ok=True)
LOG_FILE = 'logs/root_stream.log'
FILE_CHECK_INTERVAL = 2  # 文件检查间隔（秒）
PROCESS_TIMEOUT = 6099999990    # 最长处理时间（秒）

def get_files_pathlib(root_dir):
    """使用pathlib递归获取文件路径"""
    root = Path(root_dir)
    return [str(path) for path in root.glob('**/*') if path.is_file()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/file/<filename>')
def file(filename):
    file_path = os.path.join(app.config['WORKSPACE'], filename)
    if os.path.isfile(file_path):
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type and mime_type.startswith('text/'):
            if mime_type == 'text/html':
                return send_from_directory(app.config['WORKSPACE'], filename)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return render_template('code.html', filename=filename, content=content)
        elif mime_type == 'application/pdf':
            return send_from_directory(app.config['WORKSPACE'], filename)
        else:
            return send_from_directory(app.config['WORKSPACE'], filename)
    else:
        return "File not found", 404

@app.route('/api/keys/status')
def api_keys_status():
    """API endpoint to get status of all API keys"""
    return jsonify(api_key_manager.get_keys_status())

@app.route('/api/agents')
def get_agents():
    """API endpoint to get available agents"""
    agents = [
        {
            "id": "manus",
            "name": "Manus",
            "description": "Advanced code analysis and generation agent",
            "status": "ready",
            "capabilities": ["Code Analysis", "Bug Fixing", "Documentation"]
        },
        {
            "id": "flow",
            "name": "Flow Agent", 
            "description": "Workflow automation and task management",
            "status": "ready",
            "capabilities": ["Task Automation", "Process Design", "Integration"]
        },
        {
            "id": "data-analysis",
            "name": "Data Analysis",
            "description": "Data processing and insights generation", 
            "status": "stopped",
            "capabilities": ["Data Processing", "Visualization", "Analytics"]
        }
    ]
    return jsonify(agents)

@app.route('/api/tasks')
def get_tasks():
    """API endpoint to get recent tasks"""
    tasks = [
        {
            "id": 1,
            "title": "Code Review",
            "description": "Review Python application code",
            "status": "completed",
            "agent": "Manus"
        },
        {
            "id": 2, 
            "title": "Data Processing",
            "description": "Process CSV data file",
            "status": "running",
            "agent": "Data Analysis"
        },
        {
            "id": 3,
            "title": "Bug Analysis", 
            "description": "Fix bot stop display issue",
            "status": "pending",
            "agent": "Manus"
        }
    ]
    return jsonify(tasks)

# File upload utilities
def allowed_file(filename):
    """Check if file type is allowed"""
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx', 'csv', 'xlsx', 'py', 'js', 'html', 'css', 'json', 'xml', 'md'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_chat_history(chat_data):
    """Save chat history to file"""
    try:
        if os.path.exists(app.config['CHAT_HISTORY_FILE']):
            with open(app.config['CHAT_HISTORY_FILE'], 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []

        history.append(chat_data)

        # Keep only last 100 conversations
        if len(history) > 100:
            history = history[-100:]

        with open(app.config['CHAT_HISTORY_FILE'], 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")

def load_chat_history():
    """Load chat history from file"""
    try:
        if os.path.exists(app.config['CHAT_HISTORY_FILE']):
            with open(app.config['CHAT_HISTORY_FILE'], 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"Error loading chat history: {e}")
        return []

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add timestamp to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Get file info
            file_size = os.path.getsize(filepath)
            file_info = {
                'filename': filename,
                'original_name': file.filename,
                'size': file_size,
                'upload_time': datetime.now().isoformat(),
                'path': filepath
            }

            return jsonify({
                'success': True,
                'file_info': file_info,
                'message': f'File {file.filename} uploaded successfully'
            })
        else:
            return jsonify({'error': 'File type not allowed'}), 400

    except Exception as e:
        logger.error(f"File upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/files')
def get_uploaded_files():
    """Get list of uploaded files"""
    try:
        files = []
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if os.path.isfile(filepath):
                    file_info = {
                        'filename': filename,
                        'size': os.path.getsize(filepath),
                        'modified_time': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                    }
                    files.append(file_info)

        return jsonify({'files': files})
    except Exception as e:
        logger.error(f"Error getting files: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat-history')
def get_chat_history():
    """Get chat history"""
    try:
        history = load_chat_history()
        return jsonify({'history': history})
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-task', methods=['POST'])
def stop_task():
    """Stop running AI task"""
    try:
        data = request.get_json()
        task_id = data.get('task_id', 'default')

        if task_id in running_tasks:
            # Signal the task to stop
            running_tasks[task_id]['stop_flag'] = True
            logger.info(f"Stop signal sent for task: {task_id}")
            return jsonify({'success': True, 'message': 'Stop signal sent'})
        else:
            return jsonify({'success': False, 'message': 'No running task found'})

    except Exception as e:
        logger.error(f"Error stopping task: {e}")
        return jsonify({'error': str(e)}), 500

async def main(prompt, task_id=None):
    """Enhanced main function with advanced API key rotation and stop functionality"""
    max_retries = len(api_key_manager.api_keys)
    retry_count = 0

    while retry_count < max_retries:
        # Check for stop signal before starting
        if task_id and running_tasks.get(task_id, {}).get('stop_flag', False):
            logger.info(f"Task {task_id} stopped before starting agent")
            return

        # Get available API key
        result = api_key_manager.get_available_api_key(use_random=True)
        if not result:
            logger.error("No API keys available for request")

            # Wait for next available key
            max_wait_time = 10  # 5 minutes
            wait_time = 5
            start_time = time.time()

            while time.time() - start_time < max_wait_time:
                logger.info(f"Waiting {wait_time}s for API key availability...")
                await asyncio.sleep(wait_time)

                # Check for stop signal during wait
                if task_id and running_tasks.get(task_id, {}).get('stop_flag', False):
                    logger.info(f"Task {task_id} stopped during API key wait")
                    return

                result = api_key_manager.get_available_api_key(use_random=True)
                if result:
                    break

                wait_time = min(wait_time * 2, 60)  # Exponential backoff, max 60s

            if not result:
                raise Exception("No API keys became available within timeout period")

        api_key, key_config = result
        key_name = key_config['name']

        try:
            logger.info(f"Using API key: {key_name}")

            # Create Manus agent with advanced API key manager and task_id for stop checking
            agent = await Manus.create(
                api_key_manager=api_key_manager,
                api_key=api_key,
                task_id=task_id,
                running_tasks=running_tasks
            )

            # Execute the task
            await agent.run(prompt)

            # Record successful request
            api_key_manager.record_successful_request(api_key)
            logger.info(f"Task completed successfully with key: {key_name}")
            break

        except Exception as e:
            error_str = str(e).lower()

            # Handle different types of errors
            if any(keyword in error_str for keyword in ["rate limit", "quota", "too many requests"]):
                logger.warning(f"Rate limit error with key {key_name}: {e}")
                api_key_manager.record_rate_limit_error(api_key, key_name)
            elif any(keyword in error_str for keyword in ["authentication", "invalid api key", "unauthorized"]):
                logger.error(f"Authentication error with key {key_name}: {e}")
                api_key_manager.record_failure(api_key, key_name, "auth_error")
            elif any(keyword in error_str for keyword in ["timeout", "connection"]):
                logger.warning(f"Connection error with key {key_name}: {e}")
                api_key_manager.record_failure(api_key, key_name, "connection_error")
            else:
                logger.error(f"Unexpected error with key {key_name}: {e}")
                api_key_manager.record_failure(api_key, key_name, "unknown_error")

            retry_count += 1
            if retry_count >= max_retries:
                logger.error("All API keys exhausted, task failed")
                raise Exception(f"Task failed after trying all available API keys. Last error: {e}")

            logger.info(f"Retrying with different API key (attempt {retry_count + 1}/{max_retries})")

        finally:
            if 'agent' in locals():
                await agent.cleanup()

# Thread wrapper
def run_async_task(message, task_id=None):
    """Run async task in new thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main(message, task_id))
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
    finally:
        loop.close()

@app.route('/api/chat-stream', methods=['POST'])
def chat_stream():
    """Enhanced streaming chat interface with stop functionality"""
    # Clear log file
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    # Get request data
    prompt_data = request.get_json()
    message = prompt_data["message"]
    task_id = prompt_data.get("task_id", str(uuid.uuid4()))
    uploaded_files = prompt_data.get("uploaded_files", [])

    logger.info(f"Received request: {message}")

    # Process uploaded files if any
    file_context = ""
    if uploaded_files:
        file_context = "\n\nUploaded files context:\n"
        for file_info in uploaded_files:
            try:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_info['filename'])
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()[:2000]  # Limit content size
                    file_context += f"\n--- {file_info['original_name']} ---\n{content}\n"
            except Exception as e:
                logger.error(f"Error reading file {file_info['filename']}: {e}")

    full_message = message + file_context

    # Initialize task tracking
    running_tasks[task_id] = {
        'stop_flag': False,
        'start_time': time.time()
    }

    # Start async task thread
    task_thread = threading.Thread(
        target=run_async_task,
        args=(full_message, task_id)
    )
    task_thread.start()

    # Streaming generator
    def generate():
        start_time = time.time()
        full_response = ""

        while task_thread.is_alive() or not log_queue.empty():
            # Check for stop signal
            if running_tasks.get(task_id, {}).get('stop_flag', False):
                yield "Task stopped by user.\n"
                break

            # Timeout check
            if time.time() - start_time > PROCESS_TIMEOUT:
                yield """0303030"""
                break

            new_content = ""
            try:
                new_content = log_queue.get(timeout=0.1)
            except queue.Empty:
                pass

            if new_content:
                full_response += new_content
                yield new_content

            # Pause when no new content
            if not new_content:
                time.sleep(FILE_CHECK_INTERVAL)

        # Save chat history
        chat_data = {
            'id': task_id,
            'timestamp': datetime.now().isoformat(),
            'user_message': message,
            'agent_response': full_response,
            'agent_type': 'manus',
            'uploaded_files': uploaded_files
        }
        save_chat_history(chat_data)

        # Clean up task tracking
        if task_id in running_tasks:
            del running_tasks[task_id]

        # Final confirmation
        yield """0303030"""

    return Response(generate(), mimetype="text/plain")

# Run flow async task
async def run_flow_task(prompt, task_id=None):
    """Enhanced run_flow function with advanced API key rotation and stop functionality"""
    from app.agent.data_analysis import DataAnalysis
    from app.flow.flow_factory import FlowFactory, FlowType

    max_retries = len(api_key_manager.api_keys)
    retry_count = 0

    while retry_count < max_retries:
        # Check for stop signal before starting
        if task_id and running_tasks.get(task_id, {}).get('stop_flag', False):
            logger.info(f"Flow task {task_id} stopped before starting agent")
            return

        # Get available API key
        result = api_key_manager.get_available_api_key(use_random=True)
        if not result:
            logger.error("No API keys available for request")

            # Wait for next available key
            max_wait_time = 300  # 5 minutes
            wait_time = 5
            start_time = time.time()

            while time.time() - start_time < max_wait_time:
                logger.info(f"Waiting {wait_time}s for API key availability...")
                await asyncio.sleep(wait_time)

                # Check for stop signal during wait
                if task_id and running_tasks.get(task_id, {}).get('stop_flag', False):
                    logger.info(f"Flow task {task_id} stopped during API key wait")
                    return

                result = api_key_manager.get_available_api_key(use_random=True)
                if result:
                    break

                wait_time = min(wait_time * 2, 60)  # Exponential backoff, max 60s

            if not result:
                raise Exception("No API keys became available within timeout period")

        api_key, key_config = result
        key_name = key_config['name']

        try:
            logger.info(f"Using API key: {key_name}")

            # Create agents with advanced API key manager and task_id for stop checking
            agents = {
                "manus": await Manus.create(
                    api_key_manager=api_key_manager,
                    api_key=api_key,
                    task_id=task_id,
                    running_tasks=running_tasks
                ),
            }

            if app_config.run_flow_config.use_data_analysis_agent:
                # Create data analysis agent with stop tracking
                data_agent = DataAnalysis()
                if task_id and running_tasks:
                    data_agent.task_id = task_id
                    data_agent.running_tasks = running_tasks
                agents["data_analysis"] = data_agent

            # Create and execute flow
            flow = FlowFactory.create_flow(
                flow_type=FlowType.PLANNING,
                agents=agents,
            )

            logger.warning("Processing your request with flow...")

            try:
                start_time = time.time()
                result = await asyncio.wait_for(
                    flow.execute(prompt),
                    timeout=3600,  # 60 minute timeout for the entire execution
                )
                elapsed_time = time.time() - start_time
                logger.info(f"Request processed in {elapsed_time:.2f} seconds")
                logger.info(result)

                # Record successful request
                api_key_manager.record_successful_request(api_key)
                logger.info(f"Flow task completed successfully with key: {key_name}")
                break

            except asyncio.TimeoutError:
                logger.error("Request processing timed out after 1 hour")
                logger.info("Operation terminated due to timeout. Please try a simpler request.")
                break

        except Exception as e:
            error_str = str(e).lower()

            # Handle different types of errors
            if any(keyword in error_str for keyword in ["rate limit", "quota", "too many requests"]):
                logger.warning(f"Rate limit error with key {key_name}: {e}")
                api_key_manager.record_rate_limit_error(api_key, key_name)
            elif any(keyword in error_str for keyword in ["authentication", "invalid api key", "unauthorized"]):
                logger.error(f"Authentication error with key {key_name}: {e}")
                api_key_manager.record_failure(api_key, key_name, "auth_error")
            elif any(keyword in error_str for keyword in ["timeout", "connection"]):
                logger.warning(f"Connection error with key {key_name}: {e}")
                api_key_manager.record_failure(api_key, key_name, "connection_error")
            else:
                logger.error(f"Unexpected error with key {key_name}: {e}")
                api_key_manager.record_failure(api_key, key_name, "unknown_error")

            retry_count += 1
            if retry_count >= max_retries:
                logger.error("All API keys exhausted, flow task failed")
                raise Exception(f"Flow task failed after trying all available API keys. Last error: {e}")

            logger.info(f"Retrying with different API key (attempt {retry_count + 1}/{max_retries})")

        finally:
            if 'agents' in locals():
                for agent in agents.values():
                    if hasattr(agent, 'cleanup'):
                        await agent.cleanup()

def run_flow_async_task(message, task_id=None):
    """Run flow async task in new thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_flow_task(message, task_id))
    except Exception as e:
        logger.error(f"Flow task execution failed: {e}")
    finally:
        loop.close()

@app.route('/api/flow-stream', methods=['POST'])
def flow_stream():
    """Enhanced Flow streaming interface with stop functionality"""
    # Clear log file
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    # Get request data
    prompt_data = request.get_json()
    message = prompt_data["message"]
    task_id = prompt_data.get("task_id", str(uuid.uuid4()))
    uploaded_files = prompt_data.get("uploaded_files", [])

    logger.info(f"Received Flow request: {message}")

    # Process uploaded files if any
    file_context = ""
    if uploaded_files:
        file_context = "\n\nUploaded files context:\n"
        for file_info in uploaded_files:
            try:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_info['filename'])
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()[:2000]  # Limit content size
                    file_context += f"\n--- {file_info['original_name']} ---\n{content}\n"
            except Exception as e:
                logger.error(f"Error reading file {file_info['filename']}: {e}")

    full_message = message + file_context

    # Initialize task tracking
    running_tasks[task_id] = {
        'stop_flag': False,
        'start_time': time.time()
    }

    # Start async task thread
    task_thread = threading.Thread(
        target=run_flow_async_task,
        args=(full_message, task_id)
    )
    task_thread.start()

    # Streaming generator
    def generate():
        start_time = time.time()
        full_response = ""

        while task_thread.is_alive() or not log_queue.empty():
            # Check for stop signal
            if running_tasks.get(task_id, {}).get('stop_flag', False):
                yield "Task stopped by user.\n"
                break

            # Timeout check
            if time.time() - start_time > PROCESS_TIMEOUT:
                yield """0303030"""
                break

            new_content = ""
            try:
                new_content = log_queue.get(timeout=0.1)
            except queue.Empty:
                pass

            if new_content:
                full_response += new_content
                yield new_content

            # Pause when no new content
            if not new_content:
                time.sleep(FILE_CHECK_INTERVAL)

        # Save chat history
        chat_data = {
            'id': task_id,
            'timestamp': datetime.now().isoformat(),
            'user_message': message,
            'agent_response': full_response,
            'agent_type': 'flow',
            'uploaded_files': uploaded_files
        }
        save_chat_history(chat_data)

        # Clean up task tracking
        if task_id in running_tasks:
            del running_tasks[task_id]

        # Final confirmation
        yield """0303030"""

    return Response(generate(), mimetype="text/plain")

# WSGI entry point for deployment
application = app

if __name__ == '__main__':
    # Log initial API key status
    logger.info("=== Initial API Key Status ===")
    for status in api_key_manager.get_keys_status():
        logger.info(f"Key {status['name']}: Available={status['available']}, "
                    f"Usage={status['usage']['requests_this_day']}/{status['limits']['max_per_day']} today")

    app.run(host='0.0.0.0', port=3000,  debug=False)
