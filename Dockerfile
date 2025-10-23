# Use the official Golang image as the base
FROM golang:1.23.4

# Set the working directory inside the container
WORKDIR /app

# Copy go.mod and go.sum and install dependencies
COPY go.mod ./
COPY go.sum ./
RUN go mod download

# Copy the rest of the project files
COPY . .

# Build the application
RUN go build -o /openrouter-gpt-telegram-bot

# Specify the command to run the application
CMD ["/openrouter-gpt-telegram-bot"]
