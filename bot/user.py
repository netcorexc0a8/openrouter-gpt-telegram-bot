"""
Module for managing user data, conversation history, and usage statistics.

This module provides classes and functions for:
- Storing user information, message history, and usage statistics
- Managing conversation history with thread-safe operations
- Loading and saving user data to/from JSON files
- Implementing buffering and periodic saving of user data
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import json
import os
import threading
from datetime import datetime
import time


@dataclass
class Message:
    """
    Represents a message in the conversation history.
    
    Attributes:
        role: The role of the message sender ('user' or 'assistant')
        content: The content of the message
    """
    role: str  # 'user' or 'assistant'
    content: str


@dataclass
class UsageHist:
    """
    Stores usage history including chat costs by date.
    
    Attributes:
        chat_cost: Dictionary mapping dates to chat costs
    """
    chat_cost: Dict[str, float] = field(default_factory=dict)


@dataclass
class UserUsage:
    """
    Stores user usage statistics including name and usage history.
    
    Attributes:
        user_name: The name of the user
        usage_history: The usage history for the user
    """
    user_name: str = ""
    usage_history: UsageHist = field(default_factory=UsageHist)


@dataclass
class History:
    """
    Manages conversation history with thread-safe operations.
    
    Attributes:
        messages: List of messages in the history
        _lock: Internal lock for thread safety
    """
    messages: List[Message] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the history with thread safety.
        """
        with self._lock:
            self.messages.append(Message(role=role, content=content))

    def get_messages(self) -> List[Message]:
        """
        Get all messages from history with thread safety.
        """
        with self._lock:
            return self.messages.copy()

    def clear_history(self) -> None:
        """
        Clear all messages from history with thread safety.
        """
        with self._lock:
            self.messages = []

    def check_history(self, max_messages: int, max_time: int) -> None:
        """
        Check and clean up history based on max messages and time constraints.
        """
        with self._lock:
            # Check if we need to remove old messages based on time
            if len(self.messages) > 0:
                # For simplicity, we'll just check if the history needs to be cleared based on time
                # In a real implementation, you'd track timestamps per message
                pass
            
            # Remove old messages if exceeding max_messages
            if len(self.messages) > max_messages:
                # Keep only the most recent messages
                self.messages = self.messages[-max_messages:]


@dataclass
class User:
    """
    Represents a user with their history, usage, and settings.
    
    Attributes:
        user_id: The unique identifier for the user
        user_name: The name of the user
        system_prompt: The system prompt for the user
        last_message_time: The timestamp of the last message from the user
        history: The conversation history for the user
        usage: The usage statistics for the user
        _lock: Internal lock for thread safety
    """
    user_id: str
    user_name: str = ""
    system_prompt: str = ""
    last_message_time: Optional[float] = None
    history: History = field(default_factory=History)
    usage: UserUsage = field(default_factory=UserUsage)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def add_message_to_history(self, role: str, content: str) -> None:
        """
        Add a message to the user's history.
        """
        self.history.add_message(role, content)

    def get_history(self) -> List[Message]:
        """
        Get the user's message history.
        """
        return self.history.get_messages()

    def reset_history(self) -> None:
        """
        Reset the user's message history.
        """
        self.history.clear_history()

    def update_usage(self, cost: float) -> None:
        """
        Update the user's usage statistics with a new cost.
        """
        with self._lock:
            today = datetime.now().strftime("%Y-%m-%d")
            if self.usage.usage_history.chat_cost is None:
                self.usage.usage_history.chat_cost = {}
            self.usage.usage_history.chat_cost[today] = self.usage.usage_history.chat_cost.get(today, 0) + cost

    def get_current_cost(self, period: str) -> float:
        """
        Get the current cost based on the specified period ('daily', 'monthly', 'total').
        """
        with self._lock:
            today = datetime.now().strftime("%Y-%m-%d")
            cost = 0.0

            if period == "daily":
                cost = self._calculate_cost_for_day(today)
            elif period == "monthly":
                cost = self._calculate_cost_for_month(today)
            elif period == "total":
                cost = self._calculate_total_cost()
            else:
                import logging
                logging.warning(f"Invalid period: {period}. Valid periods are 'daily', 'monthly', 'total'.")
                return 0.0

            return cost

    def _calculate_cost_for_day(self, day: str) -> float:
        """
        Calculate the cost for a specific day.
        """
        return self.usage.usage_history.chat_cost.get(day, 0.0)

    def _calculate_cost_for_month(self, today: str) -> float:
        """
        Calculate the cost for the current month.
        """
        cost = 0.0
        month = today[:7]  # Get year and month in "YYYY-MM" format

        for date, daily_cost in self.usage.usage_history.chat_cost.items():
            if date.startswith(month):
                cost += daily_cost

        return cost

    def _calculate_total_cost(self) -> float:
        """
        Calculate the total cost from usage history.
        """
        total_cost = 0.0
        for cost in self.usage.usage_history.chat_cost.values():
            total_cost += cost
        return total_cost


class UserManager:
    """
    Manages all users, their data, and histories.
    Handles loading/saving user data to/from JSON files with buffering and periodic saving.
    
    Attributes:
        user_data_dir: Directory to store user data files
        users: Dictionary mapping user IDs to User objects
        _lock: Internal lock for thread safety
        _dirty_users: Set of user IDs that need to be saved
        _save_interval: Interval in seconds to periodically save user data
        _timer: Timer for periodic saving
    """
    def __init__(self, user_data_dir: str = "user_data", save_interval: int = 30):
        """
        Initialize the UserManager.
        
        Args:
            user_data_dir: Directory to store user data files
            save_interval: Interval in seconds to periodically save user data
        """
        self.user_data_dir = user_data_dir
        self.users: Dict[str, User] = {}
        self._lock = threading.Lock()
        self._dirty_users: Set[str] = set()  # Track users that need saving
        self._save_interval = save_interval
        self._timer: Optional[threading.Timer] = None
        
        # Create user data directory if it doesn't exist
        os.makedirs(user_data_dir, exist_ok=True)
        
        # Start the periodic save timer
        self._start_periodic_save()
    
    def _start_periodic_save(self) -> None:
        """Start the periodic save timer."""
        if self._timer:
            self._timer.cancel()
        
        self._timer = threading.Timer(self._save_interval, self._periodic_save)
        self._timer.daemon = True  # Allow the program to exit even if timer is running
        self._timer.start()
    
    def _periodic_save(self) -> None:
        """Save all dirty users and restart the timer."""
        try:
            self.flush_all_users()
        finally:
            # Restart the timer
            self._start_periodic_save()
    
    def mark_user_dirty(self, user_id: str) -> None:
        """
        Mark a user as needing to be saved.
        
        Args:
            user_id: ID of the user to mark as dirty
        """
        with self._lock:
            self._dirty_users.add(user_id)
    
    def flush_user(self, user_id: str) -> None:
        """
        Force save a specific user's data to file.
        
        Args:
            user_id: ID of the user to save
        """
        with self._lock:
            if user_id in self.users:
                user = self.users[user_id]
                self.save_user(user)
                self._dirty_users.discard(user_id)
    
    def flush_all_users(self) -> None:
        """Force save all dirty users' data to files."""
        with self._lock:
            for user_id in list(self._dirty_users):
                if user_id in self.users:
                    user = self.users[user_id]
                    self.save_user(user)
            self._dirty_users.clear()
    
    def close(self) -> None:
        """Close the UserManager and flush all pending changes."""
        if self._timer:
            self._timer.cancel()
        self.flush_all_users()

    def load_user(self, user_id: str) -> Optional[User]:
        """
        Load user data from a JSON file.
        """
        file_path = os.path.join(self.user_data_dir, f"{user_id}.json")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Create User object from loaded data
            user = User(
                user_id=data.get('user_id', user_id),
                user_name=data.get('user_name', ''),
                system_prompt=data.get('system_prompt', ''),
                last_message_time=data.get('last_message_time')
            )
            
            # Reconstruct history
            history_data = data.get('history', {})
            messages_data = history_data.get('messages', [])
            messages = [Message(role=msg['role'], content=msg['content']) for msg in messages_data if 'role' in msg and 'content' in msg]
            user.history = History(messages=messages)
            
            # Reconstruct usage
            usage_data = data.get('usage', {})
            usage_hist_data = usage_data.get('usage_history', {})
            chat_cost = usage_hist_data.get('chat_cost', {})
            user.usage = UserUsage(
                user_name=usage_data.get('user_name', ''),
                usage_history=UsageHist(chat_cost=chat_cost)
            )
            
            # Store in cache
            with self._lock:
                self.users[user_id] = user
                
            return user
        except FileNotFoundError:
            # User file doesn't exist, return None
            return None
        except Exception as e:
            import logging
            logging.error(f"Error loading user {user_id}: {e}")
            return None

    def save_user(self, user: User) -> None:
        """
        Save user data to a JSON file.
        """
        file_path = os.path.join(self.user_data_dir, f"{user.user_id}.json")
        
        try:
            # Prepare data for serialization
            data = {
                'user_id': user.user_id,
                'user_name': user.user_name,
                'system_prompt': user.system_prompt,
                'last_message_time': user.last_message_time,
                'history': {
                    'messages': [{'role': msg.role, 'content': msg.content} for msg in user.history.get_messages()]
                },
                'usage': {
                    'user_name': user.usage.user_name,
                    'usage_history': {
                        'chat_cost': user.usage.usage_history.chat_cost
                    }
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            import logging
            logging.error(f"Error saving user {user.user_id}: {e}")

    def get_or_create_user(self, user_id: str, user_name: str = "") -> User:
        """
        Get an existing user or create a new one if they don't exist.
        """
        with self._lock:
            if user_id in self.users:
                return self.users[user_id]
        
        # Try to load from file
        user = self.load_user(user_id)
        
        if user is None:
            # Create new user
            user = User(user_id=user_id, user_name=user_name)
            # Save the new user to file immediately to ensure it exists
            self.save_user(user)
        
        # Cache the user
        with self._lock:
            self.users[user_id] = user
            
        return user

    def add_message_to_history(self, user_id: str, role: str, content: str) -> None:
        """
        Add a message to a user's history.
        """
        user = self.get_or_create_user(user_id)
        user.add_message_to_history(role, content)
        # Update last message time
        user.last_message_time = time.time()
        # Mark user as dirty for periodic saving
        self.mark_user_dirty(user_id)

    def reset_history(self, user_id: str) -> None:
        """
        Reset a user's history.
        """
        user = self.get_or_create_user(user_id)
        user.reset_history()
        # Mark user as dirty for periodic saving
        self.mark_user_dirty(user_id)

    def update_usage(self, user_id: str, cost: float) -> None:
        """
        Update a user's usage statistics with a new cost.
        """
        user = self.get_or_create_user(user_id)
        user.update_usage(cost)
        # Mark user as dirty for periodic saving
        self.mark_user_dirty(user_id)