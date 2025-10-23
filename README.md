# Openrouter-GPT-Telegram-Bot
This repository contains a Telegram bot that integrates with Openrouter.AI API to provide interactive, AI-driven responses.

## Installation

### Prerequisites
Before you begin, ensure you have the following:
- Git installed on your machine.
- Docker installed if you prefer to use Docker for running the bot.
- Go installed if you choose to run the bot directly with Go.

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/defriend/openrouter-gpt-telegram-bot.git
   cd openrouter-gpt-telegram-bot
   ```

2. **Configure environment settings:**
    - Rename `example.env` to `.env`.
    - Populate the `.env` file with your API keys and Telegram bot token:
        - Obtain API keys from [Openrouter API](https://openrouter.ai/keys).
        - Get your Telegram API Token from [@BotFather](https://t.me/BotFather).
    - Set `ADMIN_USER_IDS` to specify Telegram user IDs of admins if you want to enable admin commands.
    - Set `ALLOWED_TELEGRAM_USER_IDS` to specify which Telegram user IDs are allowed to interact with the bot or leave blank to allow all users (Don't forget to set up Guest Budget).
    - Set `ALLOWED_GROUP_IDS` to restrict bot interaction to specific group chats. Provide a comma-separated list of Telegram group IDs. The bot will only respond in these specified groups. Example: `-1001234567890, -1009876543210`.

3. **Set up user permissions and budgets in the `.env` file:**
    - Define budgets for users and guests using `USER_BUDGETS` and `GUEST_BUDGET` both variables can be set to 0.

4. **Choose an AI model:**
    - Set the `MODEL` variable in the `.env` file to select the AI model you wish to use, such as `meta-llama/llama-3-70b-instruct`.

### Running the Bot

#### Using Go
If you have Go installed, you can run the bot directly:
```bash
go run main.go
```

#### Using Docker Compose
To build and run the bot using Docker Compose, execute:
```bash
docker compose up
```

## Features
- **Customizable AI Models:** Choose from a variety of AI models to suit your specific needs.
- **User Management:** Manages user interactions and tracks usage with a detailed usage tracker that supports budget management for different user roles including admins, registered users, and guests.
- **Docker Support:** Offers Docker compatibility for easy deployment and scalability.
-  **Command Support:** Includes several commands for user interaction:
- - `/help`: Displays available commands.
- - `/reset`: Clears the user history and can reset the system prompt to a default or specified state.
- - `/stats`: Provides current usage statistics and message count.
- - `/stop`: Terminates any active AI response streams.
- **Group Chat Mentions:** The bot now responds in group chats only when explicitly mentioned (`@botusername`).
## Acknowledgments
- This project was inspired by and has used resources from:
    - [n3d1117/chatgpt-telegram-bot](https://github.com/n3d1117/chatgpt-telegram-bot)
    - [Openrouter.AI API Documentation](https://openrouter.ai)

Feel free to contribute to the project by submitting issues or pull requests. For more detailed information on how to configure and use the bot, refer to the API documentation provided by Openrouter.AI.
