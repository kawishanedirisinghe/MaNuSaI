"""
API Key Manager with Rate Limiting and Automatic Rotation

This module provides functionality to manage multiple API keys with individual rate limits
and automatic rotation when keys hit their limits.
"""

import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

from .config import APIKeySettings

logger = logging.getLogger(__name__)


@dataclass
class KeyUsageTracker:
    """Tracks usage statistics for an API key"""
    requests_this_minute: deque = field(default_factory=deque)
    requests_this_hour: deque = field(default_factory=deque)
    requests_this_day: deque = field(default_factory=deque)
    last_used: Optional[datetime] = None
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    is_rate_limited: bool = False
    rate_limit_reset_time: Optional[datetime] = None


class APIKeyManager:
    """
    Manages multiple API keys with rate limiting and automatic rotation.
    
    Features:
    - Rate limiting per key (minute/hour/day)
    - Automatic key rotation when limits are hit
    - Priority-based key selection
    - Failure tracking and temporary key disabling
    - Thread-safe operations
    """
    
    def __init__(self):
        self._usage_trackers: Dict[str, KeyUsageTracker] = {}
        self._lock = threading.RLock()
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
        
    def register_keys(self, api_keys: List[APIKeySettings]) -> None:
        """Register API keys for management"""
        with self._lock:
            for key_config in api_keys:
                if key_config.enabled and key_config.api_key not in self._usage_trackers:
                    self._usage_trackers[key_config.api_key] = KeyUsageTracker()
                    logger.info(f"Registered API key: {key_config.name or 'Unnamed'}")
    
    def get_available_key(self, api_keys: List[APIKeySettings], use_random_selection: bool = True, exclude_key: str = None) -> Optional[Tuple[str, APIKeySettings]]:
        """
        Get the next available API key based on rate limits with enhanced selection logic.
        
        Args:
            api_keys: List of API key configurations
            use_random_selection: If True, randomly select from available keys instead of priority order
            exclude_key: API key to exclude from selection (useful for forced rotation)
        
        Returns:
            Tuple of (api_key, key_config) or None if no keys available
        """
        with self._lock:
            self._cleanup_old_usage_data()
            
            # Filter enabled keys
            enabled_keys = [k for k in api_keys if k.enabled]
            current_time = datetime.now()
            available_keys = []
            rate_limited_keys = []
            disabled_keys = []
            
            # Categorize all keys
            for key_config in enabled_keys:
                api_key = key_config.api_key
                
                # Skip excluded key
                if exclude_key and api_key == exclude_key:
                    continue
                
                # Initialize tracker if not exists
                if api_key not in self._usage_trackers:
                    self._usage_trackers[api_key] = KeyUsageTracker()
                
                tracker = self._usage_trackers[api_key]
                
                # Check if key is temporarily disabled due to failures
                if self._is_key_temporarily_disabled(tracker, current_time):
                    disabled_keys.append((api_key, key_config, "failure_backoff"))
                    logger.debug(f"API key {key_config.name or 'Unnamed'} is temporarily disabled due to failures")
                    continue
                
                # Check if key is within rate limits
                if self._is_key_within_limits(key_config, tracker, current_time):
                    available_keys.append((api_key, key_config))
                    logger.debug(f"API key {key_config.name or 'Unnamed'} is available")
                else:
                    rate_limited_keys.append((api_key, key_config, tracker.rate_limit_reset_time))
                    logger.debug(f"API key {key_config.name or 'Unnamed'} is rate limited")
            
            # Log key status summary
            logger.info(f"Key status: {len(available_keys)} available, {len(rate_limited_keys)} rate-limited, {len(disabled_keys)} disabled")
            
            if not available_keys:
                # Check if any rate-limited keys will be available soon
                if rate_limited_keys:
                    next_available = min([reset_time for _, _, reset_time in rate_limited_keys if reset_time])
                    if next_available:
                        wait_seconds = int((next_available - current_time).total_seconds())
                        logger.warning(f"No available API keys. Next key available in {wait_seconds} seconds")
                else:
                    logger.warning("No available API keys within rate limits")
                return None
            
            # Enhanced selection logic
            if use_random_selection and len(available_keys) > 1:
                # Advanced weighted random selection
                selected_key = self._weighted_random_selection(available_keys)
                logger.info(f"Randomly selected API key: {selected_key[1].name or 'Unnamed'} (advanced weighted selection)")
                return selected_key
            else:
                # Enhanced priority-based selection
                selected_key = self._priority_based_selection(available_keys)
                logger.info(f"Priority selected API key: {selected_key[1].name or 'Unnamed'}")
                return selected_key
    
    def _weighted_random_selection(self, available_keys: List[Tuple[str, APIKeySettings]]) -> Tuple[str, APIKeySettings]:
        """Advanced weighted random selection with multiple factors"""
        import random
        import math
        
        weights = []
        for api_key, key_config in available_keys:
            tracker = self._usage_trackers.get(api_key, KeyUsageTracker())
            current_time = datetime.now()
            
            # Base weight from priority (lower priority number = higher weight)
            priority_weight = 10.0 / max(key_config.priority, 1)
            
            # Usage-based weight (less recent usage = higher weight)
            minute_usage = len(tracker.requests_this_minute)
            hour_usage = len(tracker.requests_this_hour)
            day_usage = len(tracker.requests_this_day)
            
            minute_factor = 1.0 / (minute_usage + 1)
            hour_factor = 1.0 / (hour_usage + 1)
            day_factor = 1.0 / (day_usage + 1)
            usage_weight = (minute_factor * 3 + hour_factor * 2 + day_factor) / 6
            
            # Failure-based weight (fewer failures = higher weight)
            failure_weight = 1.0 / (tracker.consecutive_failures + 1)
            
            # Time since last use (longer = slightly higher weight)
            time_weight = 1.0
            if tracker.last_used:
                time_since_use = (current_time - tracker.last_used).total_seconds()
                time_weight = min(1.0 + (time_since_use / 3600), 2.0)  # Max 2x weight after 1 hour
            
            # Capacity-based weight (more remaining capacity = higher weight)
            minute_capacity = 1.0 - (minute_usage / key_config.max_requests_per_minute)
            hour_capacity = 1.0 - (hour_usage / key_config.max_requests_per_hour)
            day_capacity = 1.0 - (day_usage / key_config.max_requests_per_day)
            capacity_weight = (minute_capacity + hour_capacity + day_capacity) / 3
            
            # Combine all factors
            final_weight = priority_weight * usage_weight * failure_weight * time_weight * capacity_weight
            weights.append(max(final_weight, 0.1))  # Minimum weight to ensure all keys have some chance
        
        # Weighted random choice
        return random.choices(available_keys, weights=weights)[0]
    
    def _priority_based_selection(self, available_keys: List[Tuple[str, APIKeySettings]]) -> Tuple[str, APIKeySettings]:
        """Enhanced priority-based selection with health metrics"""
        def sort_key(item):
            api_key, key_config = item
            tracker = self._usage_trackers.get(api_key, KeyUsageTracker())
            
            # Calculate health score for tie-breaking
            health_score = 100
            health_score -= tracker.consecutive_failures * 10
            health_score -= len(tracker.requests_this_minute) * 2
            
            return (
                key_config.priority,  # Primary: priority
                -health_score,        # Secondary: health (negative for descending)
                len(tracker.requests_this_minute),  # Tertiary: current usage
                tracker.consecutive_failures,       # Quaternary: failure count
                api_key              # Final: deterministic tie-breaker
            )
        
        available_keys.sort(key=sort_key)
        return available_keys[0]
    
    def record_request(self, api_key: str) -> None:
        """Record a successful API request"""
        with self._lock:
            if api_key not in self._usage_trackers:
                self._usage_trackers[api_key] = KeyUsageTracker()
            
            tracker = self._usage_trackers[api_key]
            current_time = datetime.now()
            
            # Add timestamp to all tracking queues
            tracker.requests_this_minute.append(current_time)
            tracker.requests_this_hour.append(current_time)
            tracker.requests_this_day.append(current_time)
            tracker.last_used = current_time
            
            # Reset failure counter on successful request
            tracker.consecutive_failures = 0
            tracker.is_rate_limited = False
            tracker.rate_limit_reset_time = None
            
            logger.debug(f"Recorded successful request for API key")
    
    def record_rate_limit_error(self, api_key: str, reset_time: Optional[datetime] = None) -> None:
        """Record a rate limit error for an API key"""
        with self._lock:
            if api_key not in self._usage_trackers:
                self._usage_trackers[api_key] = KeyUsageTracker()
            
            tracker = self._usage_trackers[api_key]
            tracker.is_rate_limited = True
            tracker.rate_limit_reset_time = reset_time or datetime.now() + timedelta(minutes=1)
            
            logger.warning(f"API key hit rate limit, disabled until {tracker.rate_limit_reset_time}")
    
    def record_failure(self, api_key: str, error_type: str = "unknown") -> None:
        """Record a failure for an API key"""
        with self._lock:
            if api_key not in self._usage_trackers:
                self._usage_trackers[api_key] = KeyUsageTracker()
            
            tracker = self._usage_trackers[api_key]
            tracker.consecutive_failures += 1
            tracker.last_failure_time = datetime.now()
            
            logger.warning(f"Recorded failure for API key: {error_type} (consecutive: {tracker.consecutive_failures})")
    
    def get_usage_stats(self, api_key: str) -> Dict:
        """Get usage statistics for an API key"""
        with self._lock:
            if api_key not in self._usage_trackers:
                return {}
            
            tracker = self._usage_trackers[api_key]
            current_time = datetime.now()
            
            # Clean old data for accurate counts
            self._clean_usage_queue(tracker.requests_this_minute, current_time, timedelta(minutes=1))
            self._clean_usage_queue(tracker.requests_this_hour, current_time, timedelta(hours=1))
            self._clean_usage_queue(tracker.requests_this_day, current_time, timedelta(days=1))
            
            return {
                "requests_this_minute": len(tracker.requests_this_minute),
                "requests_this_hour": len(tracker.requests_this_hour),
                "requests_this_day": len(tracker.requests_this_day),
                "last_used": tracker.last_used,
                "consecutive_failures": tracker.consecutive_failures,
                "is_rate_limited": tracker.is_rate_limited,
                "rate_limit_reset_time": tracker.rate_limit_reset_time
            }
    
    def _is_key_within_limits(self, key_config: APIKeySettings, tracker: KeyUsageTracker, current_time: datetime) -> bool:
        """Check if a key is within its rate limits"""
        # Check if manually marked as rate limited
        if tracker.is_rate_limited:
            if tracker.rate_limit_reset_time and current_time >= tracker.rate_limit_reset_time:
                tracker.is_rate_limited = False
                tracker.rate_limit_reset_time = None
                logger.info(f"API key {key_config.name or 'Unnamed'} rate limit reset, re-enabling")
            else:
                return False
        
        # Clean old usage data
        self._clean_usage_queue(tracker.requests_this_minute, current_time, timedelta(minutes=1))
        self._clean_usage_queue(tracker.requests_this_hour, current_time, timedelta(hours=1))
        self._clean_usage_queue(tracker.requests_this_day, current_time, timedelta(days=1))
        
        # Check rate limits with detailed logging
        minute_usage = len(tracker.requests_this_minute)
        hour_usage = len(tracker.requests_this_hour)
        day_usage = len(tracker.requests_this_day)
        
        if minute_usage >= key_config.max_requests_per_minute:
            logger.warning(f"API key {key_config.name or 'Unnamed'} hit minute limit: {minute_usage}/{key_config.max_requests_per_minute}")
            return False
        if hour_usage >= key_config.max_requests_per_hour:
            logger.warning(f"API key {key_config.name or 'Unnamed'} hit hour limit: {hour_usage}/{key_config.max_requests_per_hour}")
            return False
        if day_usage >= key_config.max_requests_per_day:
            logger.warning(f"API key {key_config.name or 'Unnamed'} hit daily limit: {day_usage}/{key_config.max_requests_per_day}")
            # Mark as rate limited until next day
            tracker.is_rate_limited = True
            # Reset at midnight of next day
            next_day = (current_time + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            tracker.rate_limit_reset_time = next_day
            logger.info(f"API key {key_config.name or 'Unnamed'} disabled until {next_day}")
            return False
        
        return True
    
    def _is_key_temporarily_disabled(self, tracker: KeyUsageTracker, current_time: datetime) -> bool:
        """Check if a key should be temporarily disabled due to failures with exponential backoff"""
        if tracker.consecutive_failures >= 3:
            if tracker.last_failure_time:
                # Exponential backoff: 5 minutes * 2^(failures-3)
                backoff_minutes = 5 * (2 ** (tracker.consecutive_failures - 3))
                backoff_minutes = min(backoff_minutes, 240)  # Cap at 4 hours
                
                if current_time - tracker.last_failure_time > timedelta(minutes=backoff_minutes):
                    logger.info(f"Re-enabling API key after {backoff_minutes} minute backoff")
                    tracker.consecutive_failures = 0
                    return False
                
                logger.debug(f"API key still in backoff period ({backoff_minutes} minutes)")
                return True
        return False
    
    def _clean_usage_queue(self, usage_queue: deque, current_time: datetime, time_window: timedelta) -> None:
        """Remove old entries from usage queue"""
        cutoff_time = current_time - time_window
        while usage_queue and usage_queue[0] < cutoff_time:
            usage_queue.popleft()
    
    def _cleanup_old_usage_data(self) -> None:
        """Periodic cleanup of old usage data"""
        current_time_seconds = time.time()
        if current_time_seconds - self._last_cleanup < self._cleanup_interval:
            return
        
        current_time = datetime.now()
        for tracker in self._usage_trackers.values():
            self._clean_usage_queue(tracker.requests_this_minute, current_time, timedelta(minutes=1))
            self._clean_usage_queue(tracker.requests_this_hour, current_time, timedelta(hours=1))
            self._clean_usage_queue(tracker.requests_this_day, current_time, timedelta(days=1))
        
        self._last_cleanup = current_time_seconds

    def force_key_rotation(self, current_api_key: str, api_keys: List[APIKeySettings]) -> Optional[Tuple[str, APIKeySettings]]:
        """Force rotation to a different API key, excluding the current one"""
        with self._lock:
            available_keys = []
            current_time = datetime.now()
            
            for key_config in api_keys:
                if not key_config.enabled or key_config.api_key == current_api_key:
                    continue
                
                api_key = key_config.api_key
                
                if api_key not in self._usage_trackers:
                    self._usage_trackers[api_key] = KeyUsageTracker()
                
                tracker = self._usage_trackers[api_key]
                
                # Check if key is available (more lenient for forced rotation)
                if not self._is_key_temporarily_disabled(tracker, current_time):
                    if self._is_key_within_limits(key_config, tracker, current_time):
                        available_keys.append((api_key, key_config))
            
            if available_keys:
                # Random selection from available alternatives
                import random
                selected_key = random.choice(available_keys)
                logger.info(f"Force rotated to API key: {selected_key[1].name or 'Unnamed'}")
                return selected_key
            
            return None

    def get_key_rotation_stats(self) -> Dict:
        """Get statistics about key rotation and usage"""
        with self._lock:
            stats = {
                "total_keys": len(self._usage_trackers),
                "keys_detail": {}
            }
            
            current_time = datetime.now()
            
            for api_key, tracker in self._usage_trackers.items():
                # Clean old data for accurate counts
                self._clean_usage_queue(tracker.requests_this_minute, current_time, timedelta(minutes=1))
                self._clean_usage_queue(tracker.requests_this_hour, current_time, timedelta(hours=1))
                self._clean_usage_queue(tracker.requests_this_day, current_time, timedelta(days=1))
                
                stats["keys_detail"][api_key[:8] + "..."] = {
                    "requests_this_minute": len(tracker.requests_this_minute),
                    "requests_this_hour": len(tracker.requests_this_hour),
                    "requests_this_day": len(tracker.requests_this_day),
                    "consecutive_failures": tracker.consecutive_failures,
                    "is_rate_limited": tracker.is_rate_limited,
                    "last_used": str(tracker.last_used) if tracker.last_used else "Never"
                }
            
            return stats

    def get_all_keys_status(self) -> List[Dict]:
        """Get detailed status of all registered API keys"""
        from app.config import config
        llm_config = config.llm.get("default", config.llm["default"])
        
        if not llm_config.api_keys:
            return []
        
        status_list = []
        current_time = datetime.now()
        
        for key_config in llm_config.api_keys:
            api_key = key_config.api_key
            stats = self.get_usage_stats(api_key)
            
            # Check if key can make request
            tracker = self._usage_trackers.get(api_key)
            can_make_request = False
            next_available_in = None
            
            if tracker:
                can_make_request = self._is_key_within_limits(key_config, tracker, current_time)
                if not can_make_request and tracker.rate_limit_reset_time:
                    next_available_in = int((tracker.rate_limit_reset_time - current_time).total_seconds())
            else:
                can_make_request = True
            
            status = {
                "name": key_config.name or f"Key {api_key[:8]}...",
                "enabled": key_config.enabled,
                "can_make_request": can_make_request,
                "rate_limits": {
                    "requests_this_minute": stats.get("requests_this_minute", 0),
                    "requests_this_hour": stats.get("requests_this_hour", 0),
                    "requests_this_day": stats.get("requests_this_day", 0),
                    "max_per_minute": key_config.max_requests_per_minute,
                    "max_per_hour": key_config.max_requests_per_hour,
                    "max_per_day": key_config.max_requests_per_day,
                },
                "consecutive_failures": stats.get("consecutive_failures", 0),
                "last_used": str(stats.get("last_used", "Never")) if stats.get("last_used") else "Never"
            }
            
            if next_available_in and next_available_in > 0:
                status["next_available_in"] = next_available_in
            
            status_list.append(status)
        
        return status_list


# Global instance
api_key_manager = APIKeyManager()
