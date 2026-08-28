import configparser
import os
from pathlib import Path


class Config:
    """Configuration manager for OntoVis."""

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / ".config"

        self.config_path = Path(config_path)
        self.config = configparser.ConfigParser()

        if self.config_path.exists():
            self.config.read(self.config_path)
        else:
            raise FileNotFoundError(
                f"Config file not found at {self.config_path}. "
                f"Please copy .config.example to .config and add your API keys."
            )

    def get_llm_config(self):
        """Get LLM configuration.

        Returns:
            dict: LLM configuration with base_url, api_key, and model
        """
        if 'llm' not in self.config:
            raise ValueError("LLM configuration not found in .config file")

        llm_config = {
            'base_url': self.config.get('llm', 'base_url'),
            'api_key': self.config.get('llm', 'api_key'),
            'model': self.config.get('llm', 'model', fallback='anthropic.claude-sonnet-4-5-20250929-v1:0')
        }

        if not llm_config['api_key'] or llm_config['api_key'] == 'your-api-key-here':
            raise ValueError("Please set a valid API key in the .config file")

        return llm_config

    def get_api_key(self):
        """Get the LLM API key.

        Returns:
            str: API key
        """
        return self.get_llm_config()['api_key']

    def get_model_name(self):
        """Get the LLM model name.

        Returns:
            str: Model name
        """
        return self.get_llm_config()['model']
