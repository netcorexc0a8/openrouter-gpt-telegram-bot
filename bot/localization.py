import json
import os
from typing import Dict, Optional


class Localization:
    """
    Class to handle localization loading and string retrieval.
    Loads language strings from JSON files in the lang directory.
    """
    
    def __init__(self, lang_dir: str = "lang", default_lang: str = "EN"):
        """
        Initialize the Localization class.
        
        Args:
            lang_dir: Directory containing language JSON files
            default_lang: Default language code to use if specified language is not available
        """
        # Check if lang_dir is relative and prepend the correct path
        if not os.path.isabs(lang_dir):
            # Look for the lang directory in the current working directory or relative to the script
            if os.path.exists(os.path.join(os.getcwd(), lang_dir)):
                self.lang_dir = os.path.join(os.getcwd(), lang_dir)
            elif os.path.exists(os.path.join(os.path.dirname(__file__), '..', lang_dir)):
                self.lang_dir = os.path.join(os.path.dirname(__file__), '..', lang_dir)
            else:
                self.lang_dir = lang_dir
        else:
            self.lang_dir = lang_dir
            
        self.default_lang = default_lang
        self.translations: Dict[str, Dict[str, str]] = {}
        self.current_language: str = default_lang
        
        # Load available translations
        self.load_translations()
    
    def get_lang_dir(self) -> str:
        """
        Determine the correct language directory path, checking multiple possible locations.
        """
        # Check if lang_dir is relative and try different possible locations
        if not os.path.isabs(self.lang_dir):
            # Try different possible paths where the lang directory might be located
            possible_paths = [
                os.path.join(os.getcwd(), self.lang_dir),
                os.path.join(os.path.dirname(__file__), '..', self.lang_dir),
                os.path.join('/app', self.lang_dir),  # Common in Docker containers
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', self.lang_dir),
                self.lang_dir # Fallback to original path
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    return path
        
        return self.lang_dir
    
    def load_translations(self) -> None:
        """
        Load all available language files from the lang directory.
        """
        # Define supported languages based on available files
        languages = []
        
        # Look for JSON files in the language directory
        lang_dir = self.get_lang_dir()
        for filename in os.listdir(lang_dir):
            if filename.endswith('.json'):
                lang_code = filename[:-5].upper()  # Remove '.json' and convert to uppercase
                languages.append(lang_code)
        
        # Load each language file
        for lang in languages:
            self.load_language(lang)
    
    def load_language(self, lang_code: str) -> None:
        """
        Load a specific language from its JSON file.
        
        Args:
            lang_code: Language code (e.g., 'EN', 'RU')
        """
        lang_dir = self.get_lang_dir()
        file_path = os.path.join(lang_dir, f"{lang_code}.json")
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                self.translations[lang_code] = json.load(f)
        else:
            # If the specific language file doesn't exist, load the default language
            lang_dir = self.get_lang_dir()
            default_path = os.path.join(lang_dir, f"{self.default_lang}.json")
            if os.path.exists(default_path):
                with open(default_path, 'r', encoding='utf-8') as f:
                    self.translations[lang_code] = json.load(f)
            else:
                # If even the default file doesn't exist, create an empty dict
                self.translations[lang_code] = {}
    
    def set_language(self, lang_code: str) -> None:
        """
        Set the current language for translations.
        
        Args:
            lang_code: Language code to set as current
        """
        # Ensure the language is loaded
        if lang_code.upper() not in self.translations:
            self.load_language(lang_code.upper())
        
        self.current_language = lang_code.upper()
    
    def get(self, key: str, lang: Optional[str] = None) -> str:
        """
        Get a localized string by key.
        
        Args:
            key: The key to translate (e.g., "commands.start")
            lang: Optional language code to use instead of the current language
            
        Returns:
            Translated string or the key if not found
        """
        # Use the specified language or fall back to current language
        translation_lang = lang.upper() if lang else self.current_language
        
        # Ensure the language is loaded
        if translation_lang not in self.translations:
            self.load_language(translation_lang)
        
        # Split the key by dots to navigate nested dictionaries
        keys = key.split('.')
        result = self.translations[translation_lang]
        
        # Navigate through the nested structure
        for k in keys:
            if isinstance(result, dict) and k in result:
                result = result[k]
            else:
                return key  # Return the original key if translation not found
        
        # Return the final value if it's a string, otherwise return the key
        return result if isinstance(result, str) else key


# Global instance for easy access
_localization_instance = None


def get_localization_instance() -> Localization:
    """
    Get the global localization instance.
    
    Returns:
        Localization instance
    """
    global _localization_instance
    if _localization_instance is None:
        _localization_instance = Localization()
    return _localization_instance


def translate(key: str, lang: Optional[str] = None) -> str:
    """
    Convenience function to translate a key using the global localization instance.
    
    Args:
        key: The key to translate (e.g., "commands.start")
        lang: Optional language code to use instead of the current language
        
    Returns:
        Translated string or the key if not found
    """
    localization = get_localization_instance()
    return localization.get(key, lang)