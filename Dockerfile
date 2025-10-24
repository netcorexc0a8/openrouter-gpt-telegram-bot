# Use the official Golang image as the base
FROM golang:1.23.4

# Set the working directory inside the container
WORKDIR /app

# Copy go.mod and install dependencies (go.sum will be updated by tidy)
COPY go.mod ./
COPY . .
RUN git config --global --unset credential.helper
RUN go mod tidy

# Build the application
RUN go build -o /openrouter-gpt-telegram-bot

# Specify the command to run the application
CMD ["/openrouter-gpt-telegram-bot"]
