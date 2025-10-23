package config

import (
	"fmt"
	"github.com/joho/godotenv"
	"github.com/sashabaranov/go-openai"
	"github.com/spf13/viper"
	"log"
	"reflect"

	"openrouter-gpt-telegram-bot/lang"
	"os"
	"strconv"
	"strings"
)

type Config struct {
	TelegramBotToken   string
	OpenAIApiKey       string
	Model              ModelParameters
	MaxTokens          int
	BotLanguage        string
	OpenAIBaseURL      string
	SystemPrompt       string
	BudgetPeriod       string
	GuestBudget        float64
	UserBudget         float64
	AdminChatIDs       []int64
	AllowedUserChatIDs []int64
	AllowedGroupIDs    []int64
	MaxHistorySize     int
	MaxHistoryTime     int
	Vision             string
	VisionPrompt       string
	VisionDetails      string
	StatsMinRole       string
	Lang               string
}

type ModelParameters struct {
	Type              string
	ModelName         string
	ModelReq          openai.ChatCompletionRequest
	FrequencyPenalty  float64
	MinP              float64
	PresencePenalty   float64
	RepetitionPenalty float64
	Temperature       float64
	TopA              float64
	TopK              float64
	TopP              float64
}

func Load() (*Config, error) {
	err := godotenv.Load()
	if err != nil {
		return nil, err
	}

	viper.SetDefault("MAX_TOKENS", 2000)
	viper.SetDefault("TEMPERATURE", 1)
	viper.SetDefault("TOP_P", 0.7)
	viper.SetDefault("BASE_URL", "https://api.openai.com/v1")
	viper.SetDefault("BUDGET_PERIOD", "monthly")
	viper.SetDefault("MAX_HISTORY_SIZE", 10)
	viper.SetDefault("MAX_HISTORY_TIME", 60)
	viper.SetDefault("LANG", "en")

	config := &Config{
		TelegramBotToken: os.Getenv("TELEGRAM_BOT_TOKEN"),
		OpenAIApiKey:     os.Getenv("API_KEY"),
		Model: ModelParameters{
			Type:        viper.GetString("TYPE"),
			ModelName:   viper.GetString("MODEL"),
			Temperature: viper.GetFloat64("TEMPERATURE"),
			TopP:        viper.GetFloat64("TOP_P"),
		},
		MaxTokens:          viper.GetInt("MAX_TOKENS"),
		OpenAIBaseURL:      viper.GetString("BASE_URL"),
		SystemPrompt:       viper.GetString("ASSISTANT_PROMPT"),
		BudgetPeriod:       viper.GetString("BUDGET_PERIOD"),
		GuestBudget:        viper.GetFloat64("GUEST_BUDGET"),
		UserBudget:         viper.GetFloat64("USER_BUDGET"),
		AdminChatIDs:       getStrAsIntList("ADMIN_IDS"),
		AllowedUserChatIDs: getStrAsIntList("ALLOWED_USER_IDS"),
		AllowedGroupIDs:    getEnvAsIntList("ALLOWED_GROUP_IDS"),
		MaxHistorySize:     viper.GetInt("MAX_HISTORY_SIZE"),
		MaxHistoryTime:     viper.GetInt("MAX_HISTORY_TIME"),
		Vision:             viper.GetString("VISION"),
		VisionPrompt:       viper.GetString("VISION_PROMPT"),
		VisionDetails:      viper.GetString("VISION_DETAIL"),
		StatsMinRole:       viper.GetString("STATS_MIN_ROLE"),
		Lang:               viper.GetString("LANG"),
	}
	if config.BudgetPeriod == "" {
		log.Fatalf("Set budget_period in config file")
	}
	language := lang.Translate("language", config.Lang)
	config.SystemPrompt = "Always answer in " + language + " language." + config.SystemPrompt
	//Config model
	//config = setupParameters(config)
	printConfig(config)
	return config, nil
}

func setupParameters(conf *Config) *Config {
	parameters, err := GetParameters(conf)
	if err != nil {
		log.Fatal(err)
	}
	conf.Model.FrequencyPenalty = parameters.FrequencyPenaltyP50
	conf.Model.MinP = parameters.MinPP50
	conf.Model.PresencePenalty = parameters.PresencePenaltyP50
	conf.Model.RepetitionPenalty = parameters.RepetitionPenaltyP50
	conf.Model.Temperature = parameters.TemperatureP50
	conf.Model.TopA = parameters.TopAP50
	conf.Model.TopK = parameters.TopKP50
	conf.Model.TopP = parameters.TopPP50
	conf.Model.ModelReq = openai.ChatCompletionRequest{
		Model:            conf.Model.ModelName,
		MaxTokens:        conf.MaxTokens,
		Temperature:      float32(conf.Model.Temperature),
		FrequencyPenalty: float32(conf.Model.FrequencyPenalty),
		PresencePenalty:  float32(conf.Model.PresencePenalty),
		TopP:             float32(conf.Model.TopP),
	}
	return conf
}

func getEnv(key, defaultValue string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	log.Printf("Warning: Failed to parse %s. Using default value.", key)
	return defaultValue
}

func getEnvAsIntList(name string) []int64 {
	valueStr := os.Getenv(name)
	if valueStr == "" {
		log.Println("Missing required environment variable, " + name)
		var emptyArray []int64
		return emptyArray
	}
	var values []int64
	for _, str := range strings.Split(valueStr, ",") {
		value, err := strconv.ParseInt(strings.TrimSpace(str), 10, 64)
		if err != nil {
			log.Printf("Invalid value for environment variable %s: %v", name, err)
			continue
		}
		values = append(values, value)
	}
	return values
}

func getStrAsIntList(name string) []int64 {
	valueStr := viper.GetString(name)
	if valueStr == "" {
		log.Println("Missing required environment variable, " + name)
		var emptyArray []int64
		return emptyArray
	}
	var values []int64
	for _, str := range strings.Split(valueStr, ",") {
		value, err := strconv.ParseInt(strings.TrimSpace(str), 10, 64)
		if err != nil {
			log.Printf("Invalid value for environment variable %s: %v", name, err)
			continue
		}
		values = append(values, value)
	}
	return values
}

func getEnvAsInt(name string, defaultValue int) int {
	valueStr := os.Getenv(name)
	if valueStr == "" {
		log.Printf("Environment variable %s not set, using default value: %d", name, defaultValue)
		return defaultValue
	}
	value, err := strconv.Atoi(valueStr)
	if err != nil {
		log.Printf("Error parsing environment variable %s: %v. Using default value: %d", name, err, defaultValue)
		return defaultValue
	}
	return value
}

func getEnvAsFloat(name string, defaultValue float32) float32 {
	valueStr := os.Getenv(name)
	if valueStr == "" {
		return defaultValue
	}

	value, err := strconv.ParseFloat(valueStr, 32)
	if err != nil {
		log.Printf("Warning: Failed to parse %s as float: %v. Using default value.", name, err)
		return defaultValue
	}
	return float32(value)
}

func printConfig(c *Config) {
	if c == nil {
		fmt.Println("Config is nil")
		return
	}
	v := reflect.ValueOf(*c)
	t := v.Type()

	for i := 0; i < v.NumField(); i++ {
		field := v.Field(i)
		fieldName := t.Field(i).Name
		
		if field.Kind() == reflect.Struct {
			fmt.Printf("%s:\n", fieldName)
			printStructFields(field)
		} else {
			fmt.Printf("%s: %v\n", fieldName, field.Interface())
		}
	}
}

func printStructFields(v reflect.Value) {
	t := v.Type()
	for i := 0; i < v.NumField(); i++ {
		field := v.Field(i)
		fieldName := t.Field(i).Name
		fmt.Printf("  %s: %v\n", fieldName, field.Interface())
	}
}
