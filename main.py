"""
Main module for the Telegram bot.

This module contains the TelegramBot class which handles all bot functionality including:
- Command handlers (/start, /help, /reset, /stats, /stop)
- Message processing (text and images)
- User management and history
- API communication with OpenRouter
- Configuration loading and validation
"""
import asyncio
import logging
from typing import Dict, List, Optional
from telegram import Update, Message
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from bot.config import load_config
from bot.api import OpenRouterAPI
from bot.user import UserManager
from bot.localization import Localization
import json
import os
import httpx


# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class TelegramBot:
    """
    Main Telegram Bot class that handles all bot functionality.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the bot with configuration, API client, and user manager.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize OpenRouter API client
        self.api_client = OpenRouterAPI(
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_base_url,
            requests_per_minute=self.config.api_requests_per_minute,
            tokens_per_minute=self.config.api_tokens_per_minute,
            tokens_per_day=self.config.api_tokens_per_day,
            concurrent_requests=self.config.api_concurrent_requests,
            gcra_requests_limit=self.config.gcra_requests_limit,
            gcra_tokens_limit=self.config.gcra_tokens_limit
        )
        
        # Initialize user manager
        self.user_manager = UserManager("user_data")
        
        # Initialize bot application
        self.application = Application.builder().token(self.config.telegram_bot_token).build()
        
        # Dictionary to keep track of active streams per user
        self.active_streams: Dict[int, bool] = {}
        
        # Initialize localization
        self.localization = Localization()
        self.localization.set_language(self.config.lang.upper())
    
    def translate(self, key: str, lang_code: Optional[str] = None) -> str:
        """
        Translate a key using the localization system.
        
        Args:
            key: The key to translate (e.g., "commands.start")
            lang_code: Optional language code to override the default
            
        Returns:
            Translated string or the key if not found
        """
        return self.localization.get(key, lang_code)
    
    def have_access(self, user_id: int) -> bool:
        """
        Check if a user has access based on budget and allowed users/groups.
        
        Args:
            user_id: The ID of the user to check
            
        Returns:
            True if the user has access, False otherwise
        """
        # Check if user is in allowed users list
        if self.config.allowed_user_chat_ids and user_id not in self.config.allowed_user_chat_ids:
            # If allowed users list is specified, only those users have access
            if user_id not in self.config.admin_chat_ids:
                return False
        
        # Check budget
        user = self.user_manager.get_or_create_user(str(user_id))
        current_cost = user.get_current_cost(self.config.budget_period)
        
        if user_id in self.config.admin_chat_ids:
            # Admins have unlimited access
            return True
        elif user_id in self.config.allowed_user_chat_ids:
            # Check user budget
            return current_cost < self.config.user_budget
        else:
            # Check guest budget
            return current_cost < self.config.guest_budget
    
    def can_view_stats(self, user_id: int) -> bool:
        """
        Check if a user can view detailed statistics based on their role.
        
        Args:
            user_id: The ID of the user to check
            
        Returns:
            True if the user can view detailed stats, False otherwise
        """
        if user_id in self.config.admin_chat_ids:
            return True
        
        if self.config.stats_min_role == "admin":
            return False
        elif self.config.stats_min_role == "user":
            return user_id in self.config.allowed_user_chat_ids or user_id in self.config.admin_chat_ids
        else:
            # Default: everyone can view stats (but maybe limited)
            return True
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handler for the /start command.
        """
        message = self.translate("commands.start") + \
                 self.translate("commands.help") + \
                 self.translate("commands.start_end")
        
        await update.message.reply_text(message, parse_mode="HTML")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handler for the /help command.
        """
        message = self.translate("commands.help")
        await update.message.reply_text(message, parse_mode="HTML")
    
    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handler for the /reset command.
        Can reset history, system prompt to default, or set a new system prompt.
        """
        user_id = update.effective_user.id
        args = context.args
        
        user = self.user_manager.get_or_create_user(str(user_id), update.effective_user.username or "")
        
        if args:
            if args[0] == "system":
                # Reset system prompt to default
                user.system_prompt = self.config.system_prompt
                self.user_manager.save_user(user)
                message = self.translate("commands.reset_system")
            else:
                # Set new system prompt
                new_prompt = " ".join(args)
                user.system_prompt = new_prompt
                self.user_manager.save_user(user)
                message = self.translate("commands.reset_prompt") + new_prompt + "."
        else:
            # Reset history only
            self.user_manager.reset_history(str(user_id))
            user.system_prompt = self.config.system_prompt  # Reset system prompt to default when clearing history
            self.user_manager.save_user(user)
            message = self.translate("commands.reset")
        
        await update.message.reply_text(message)
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handler for the /stats command.
        """
        user_id = update.effective_user.id
        user = self.user_manager.get_or_create_user(str(user_id), update.effective_user.username or "")
        
        # Check history based on max limits
        user.history.check_history(self.config.max_history_size, self.config.max_history_time)
        
        # Get usage statistics
        counted_usage = f"{user.get_current_cost(self.config.budget_period):.6f}"
        today_usage = f"{user.get_current_cost('daily'):.6f}"
        month_usage = f"{user.get_current_cost('monthly'):.6f}"
        total_usage = f"{user.get_current_cost('total'):.6f}"
        messages_count = len(user.get_history())
        
        if self.can_view_stats(user_id):
            # Detailed stats for users with access
            message = self.translate("commands.stats") % (
                counted_usage,
                today_usage,
                month_usage,
                total_usage,
                str(messages_count)
            )
        else:
            # Limited stats for users without detailed access
            message = self.translate("commands.stats_min") % str(messages_count)
        
        await update.message.reply_text(message, parse_mode="HTML")
    
    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handler for the /stop command.
        """
        user_id = update.effective_user.id
        
        if user_id in self.active_streams and self.active_streams[user_id]:
            # In a real implementation, we would stop the active stream
            # For now, we just clear the active stream flag
            self.active_streams[user_id] = False
            message = self.translate("commands.stop")
        else:
            message = self.translate("commands.stop_err")
        
        await update.message.reply_text(message)
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handler for text messages.
        """
        if update.message is None:
            return
        
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        
        # Check if it's a group chat and if the bot is mentioned
        is_group_chat = update.effective_chat.type in ['group', 'supergroup']
        is_mentioned = False
        
        if is_group_chat:
            # Check if the bot is mentioned in group chats
            if update.message.entities:
                for entity in update.message.entities:
                    if entity.type in ["mention", "text_mention"]:
                        if entity.type == "mention":
                            # For username mentions like @botname
                            mention_text = update.message.text[entity.offset:entity.offset + entity.length]
                            if f"@{context.bot.username}" in mention_text:
                                is_mentioned = True
                        elif entity.type == "text_mention":
                            # For text mentions where the entity contains a user object
                            if hasattr(entity, 'user') and entity.user and entity.user.id == context.bot.id:
                                is_mentioned = True
                        
                        if is_mentioned:
                            # Extract the clean message text without the mention
                            clean_message_text = (
                                update.message.text[:entity.offset] +
                                update.message.text[entity.offset + entity.length:]
                            )
                            # Update the message text to remove the bot mention
                            update.message.text = clean_message_text.strip()
                            break
            
            # Check if the list of allowed groups is not empty and the current ChatID is not in the list
            if self.config.allowed_group_ids and chat_id not in self.config.allowed_group_ids:
                # If not mentioned in allowed groups, ignore the message
                if not is_mentioned:
                    return
        
        # For private chats, process normally; for group chats, only if mentioned or in allowed groups
        if not is_group_chat or is_mentioned or chat_id in self.config.allowed_group_ids:
            # Handle the message asynchronously in a separate task
            asyncio.create_task(self.process_message(update.message, user_id))
    
    async def process_message(self, message: Message, user_id: int) -> None:
        """
        Process a message asynchronously.
        
        Args:
            message: The message to process
            user_id: The ID of the user who sent the message
        """
        # Get or create user
        user = self.user_manager.get_or_create_user(str(user_id), message.from_user.username or "")
        
        # Check access
        if not self.have_access(user_id):
            await message.reply_text(self.translate("budget_out"))
            return
        
        # Add user message to history
        self.user_manager.add_message_to_history(str(user_id), "user", message.text)
        
        # Set active stream flag
        self.active_streams[user_id] = True
        
        try:
            # Prepare messages for the API request
            history_messages = user.get_history()
            
            # Prepare the full message list with system prompt if available
            api_messages = []
            
            # Add system prompt if it exists
            if user.system_prompt:
                api_messages.append({"role": "system", "content": user.system_prompt})
            else:
                # Use default system prompt from config
                api_messages.append({"role": "system", "content": self.config.system_prompt})
            
            # Add conversation history
            for msg in history_messages:
                api_messages.append({"role": msg.role, "content": msg.content})
            
            # Apply user's GCRA settings to the API client if they exist
            if user.gcra_settings:
                self.api_client.set_user_gcra_configs(str(user_id), user.gcra_settings)
            
            # Send request to OpenRouter API
            full_response = ""
            response_message = await message.reply_text("...") # Initial response to show bot is working
            
            # Stream the response from the API
            async for chunk in self.api_client.send_chat_completion(
                messages=api_messages,
                model=self.config.model.model_name,
                temperature=self.config.model.temperature,
                top_p=self.config.model.top_p,
                frequency_penalty=self.config.model.frequency_penalty,
                presence_penalty=self.config.model.presence_penalty,
                user_id=str(user_id)  # Pass user_id for rate limiting
            ):
                if not self.active_streams.get(user_id, False):
                    # If stream was stopped, break the loop
                    break
                
                # Process the chunk
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        content = delta["content"]
                        full_response += content
                        
                        # Update the response message with the new content
                        # Limit message length to avoid hitting Telegram's limits
                        # Check if the message content has actually changed to avoid unnecessary updates
                        current_text = response_message.text if hasattr(response_message, 'text') else ""
                        if current_text != full_response:
                            if len(full_response) > 4096:
                                await response_message.edit_text(full_response[:4093] + "...")
                            else:
                                await response_message.edit_text(full_response)
            
            # Add assistant's response to history
            if full_response:
                self.user_manager.add_message_to_history(str(user_id), "assistant", full_response)
                
                # Update usage if needed (in a real implementation, you would get actual usage from API)
                # For now, we'll add a small cost based on response length
                estimated_cost = len(full_response) * 0.000001  # Placeholder cost calculation
                self.user_manager.update_usage(str(user_id), estimated_cost)
        
        except httpx.HTTPStatusError as e:
            # Handle HTTP errors from the API
            status_code = e.response.status_code if e.response else "Unknown"
            error_message = str(e)
            logger.error(f"HTTP error occurred while processing message for user {user_id}: {status_code} - {error_message}")
            
            if status_code == 429:
                await message.reply_text(self.translate("errors.rate_limit"))
            elif status_code == 500:
                await message.reply_text(self.translate("errors.server_error"))
            elif status_code in [502, 503, 504]:
                await message.reply_text(self.translate("errors.gateway_error"))
            else:
                await message.reply_text(self.translate("errors.api_error") + f" (Code: {status_code})")
        except httpx.TimeoutException as e:
            # Handle timeout errors
            logger.error(f"Timeout error occurred while processing message for user {user_id}: {str(e)}")
            await message.reply_text(self.translate("errors.timeout"))
        except httpx.RequestError as e:
            # Handle other request errors (connection issues, etc.)
            logger.error(f"Request error occurred while processing message for user {user_id}: {str(e)}")
            await message.reply_text(self.translate("errors.connection_error"))
        except Exception as e:
            # Handle any other exceptions
            logger.error(f"Unexpected error processing message for user {user_id}: {str(e)}")
            await message.reply_text(self.translate("errors.general_error"))
        finally:
            # Clear active stream flag
            self.active_streams[user_id] = False
    
    async def handle_image_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handler for image messages.
        """
        # Check if image processing is enabled in config
        if not self.config.vision or self.config.vision.lower() == "false":
            await update.message.reply_text("Image processing is not enabled.")
            return
        
        try:
            # Get the highest quality photo from the message
            if update.message.photo:
                # Get the file ID of the largest photo (last in the list)
                file_id = update.message.photo[-1].file_id
                file = await context.bot.get_file(file_id)
                
                # Download the image
                import tempfile
                import os
                from io import BytesIO
                
                # Create a temporary file to store the image
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    await file.download_to_memory(temp_file)
                    temp_file_path = temp_file.name
                
                try:
                    # Prepare messages for the API request
                    user_id = update.effective_user.id
                    user = self.user_manager.get_or_create_user(str(user_id), update.effective_user.username or "")
                    
                    # Check access
                    if not self.have_access(user_id):
                        await update.message.reply_text(self.translate("budget_out"))
                        return
                    
                    # Add user message to history (with image info)
                    image_info = f"[Image received at {update.message.date}]"
                    if update.message.caption:
                        image_info += f" Caption: {update.message.caption}"
                    
                    self.user_manager.add_message_to_history(str(user_id), "user", image_info)
                    
                    # Set active stream flag
                    self.active_streams[user_id] = True
                    
                    # Prepare the full message list with system prompt if available
                    history_messages = user.get_history()
                    api_messages = []
                    
                    # Add system prompt if it exists
                    if user.system_prompt:
                        api_messages.append({"role": "system", "content": user.system_prompt})
                    else:
                        # Use default system prompt from config
                        api_messages.append({"role": "system", "content": self.config.system_prompt})
                    
                    # Add vision prompt if configured
                    if self.config.vision_prompt:
                        api_messages.append({"role": "user", "content": [
                            {"type": "text", "text": self.config.vision_prompt},
                            {"type": "image_url", "image_url": {"url": f"file://{temp_file_path}"}}
                        ]})
                    else:
                        # Add conversation history
                        for msg in history_messages:
                            api_messages.append({"role": msg.role, "content": msg.content})
                    
                    # Apply user's GCRA settings to the API client if they exist
                    if user.gcra_settings:
                        self.api_client.set_user_gcra_configs(str(user_id), user.gcra_settings)
                    
                    # Send request to OpenRouter API
                    full_response = ""
                    response_message = await update.message.reply_text("...") # Initial response to show bot is working
                    
                    # Stream the response from the API
                    async for chunk in self.api_client.send_chat_completion(
                        messages=api_messages,
                        model=self.config.model.model_name,
                        temperature=self.config.model.temperature,
                        top_p=self.config.model.top_p,
                        frequency_penalty=self.config.model.frequency_penalty,
                        presence_penalty=self.config.model.presence_penalty,
                        user_id=str(user_id)  # Pass user_id for rate limiting
                    ):
                        if not self.active_streams.get(user_id, False):
                            # If stream was stopped, break the loop
                            break
                        
                        # Process the chunk
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                content = delta["content"]
                                full_response += content
                                
                                # Update the response message with the new content
                                # Limit message length to avoid hitting Telegram's limits
                                # Check if the message content has actually changed to avoid unnecessary updates
                                current_text = response_message.text if hasattr(response_message, 'text') else ""
                                if current_text != full_response:
                                    if len(full_response) > 4096:
                                        await response_message.edit_text(full_response[:4093] + "...")
                                    else:
                                        await response_message.edit_text(full_response)
                    
                    # Add assistant's response to history
                    if full_response:
                        self.user_manager.add_message_to_history(str(user_id), "assistant", full_response)
                        
                        # Update usage if needed (in a real implementation, you would get actual usage from API)
                        # For now, we'll add a small cost based on response length
                        estimated_cost = len(full_response) * 0.000001  # Placeholder cost calculation
                        self.user_manager.update_usage(str(user_id), estimated_cost)
                
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
            
        except httpx.HTTPStatusError as e:
            # Handle HTTP errors from the API
            status_code = e.response.status_code if e.response else "Unknown"
            error_message = str(e)
            logger.error(f"HTTP error occurred while processing image for user {user_id}: {status_code} - {error_message}")
            
            if status_code == 429:
                await update.message.reply_text(self.translate("errors.rate_limit"))
            elif status_code == 500:
                await update.message.reply_text(self.translate("errors.server_error"))
            elif status_code in [502, 503, 504]:
                await update.message.reply_text(self.translate("errors.gateway_error"))
            else:
                await update.message.reply_text(self.translate("errors.api_error") + f" (Code: {status_code})")
        except httpx.TimeoutException as e:
            # Handle timeout errors
            logger.error(f"Timeout error occurred while processing image for user {user_id}: {str(e)}")
            await update.message.reply_text(self.translate("errors.timeout"))
        except httpx.RequestError as e:
            # Handle other request errors (connection issues, etc.)
            logger.error(f"Request error occurred while processing image for user {user_id}: {str(e)}")
            await update.message.reply_text(self.translate("errors.connection_error"))
        except Exception as e:
            # Handle any other exceptions
            logger.error(f"Unexpected error processing image for user {user_id}: {str(e)}")
            await update.message.reply_text(self.translate("errors.general_error"))
        finally:
            # Clear active stream flag
            user_id = update.effective_user.id
            self.active_streams[user_id] = False
    
    def setup_handlers(self):
        """
        Set up all command and message handlers.
        """
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("reset", self.reset_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("stop", self.stop_command))
        
        # Message handlers
        # Handle text messages
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Handle image messages
        self.application.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, self.handle_image_message))
    
    async def set_bot_commands(self):
        """
        Set the bot's command descriptions in Telegram.
        """
        commands = [
            ("start", self.translate("description.start")),
            ("help", self.translate("description.help")),
            ("reset", self.translate("description.reset")),
            ("stats", self.translate("description.stats")),
            ("stop", self.translate("description.stop"))
        ]
        
        await self.application.bot.set_my_commands(commands)
    
    async def run(self):
        """
        Run the bot.
        """
        # Set up handlers
        self.setup_handlers()
        
        # Set bot commands
        await self.set_bot_commands()
        
        # Start the bot
        logger.info("Starting bot...")
        await self.application.run_polling()


# Main function to run the bot
async def main():
    """
    Main function to initialize and run the bot.
    """
    bot = TelegramBot("config.yaml")
    try:
        await bot.run()
    finally:
        # Close the API client to free up resources
        await bot.api_client.close()
        # Close the user manager to flush any pending changes
        bot.user_manager.close()


if __name__ == "__main__":
    asyncio.run(main())