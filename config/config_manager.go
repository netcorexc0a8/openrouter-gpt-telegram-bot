// config/manager.go
package config

import (
	"log"
	"sync"

	"github.com/fsnotify/fsnotify"
	"github.com/spf13/viper"
)

type Manager struct {
	config    *Config
	mutex     sync.RWMutex
	listeners []chan<- Config
}

func NewManager(configPath string) (*Manager, error) {
	// Set default values

	viper.SetConfigFile(configPath)
	viper.AutomaticEnv()

	if err := viper.ReadInConfig(); err != nil {
		log.Fatalf("Error reading config file, %s", err)
		return nil, err
	}

	manager := &Manager{
		listeners: make([]chan<- Config, 0),
	}

	// Initial config load
	config, err := Load()
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
		return nil, err
	}
	manager.config = config

	// Setup file watcher
	viper.WatchConfig()
	viper.OnConfigChange(func(e fsnotify.Event) {
		manager.reloadConfig()
	})

	return manager, nil
}

func (m *Manager) reloadConfig() {
	newConfig, err := Load()
	if err != nil {
		log.Printf("Failed to reload config: %v", err)
		return
	}

	m.mutex.Lock()
	m.config = newConfig
	m.mutex.Unlock()

	// Notify all listeners
	for _, listener := range m.listeners {
		listener <- *newConfig
	}
}

func (m *Manager) Subscribe() <-chan Config {
	ch := make(chan Config, 1)
	m.listeners = append(m.listeners, ch)
	return ch
}

func (m *Manager) GetConfig() *Config {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	return m.config
}
