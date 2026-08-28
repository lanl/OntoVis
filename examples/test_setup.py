"""
Quick test to verify the setup without making API calls.
"""

from ontovis import Config

try:
    config = Config()
    print("✓ Config file loaded successfully")

    llm_config = config.get_llm_config()
    print(f"✓ Base URL: {llm_config['base_url']}")
    print(f"✓ Model: {llm_config['model']}")
    print(f"✓ API Key: {'*' * 20}{llm_config['api_key'][-4:] if len(llm_config['api_key']) > 4 else '****'}")
    print("\nConfiguration is ready! You can now use the VisionAgent.")

except FileNotFoundError as e:
    print("✗ Config file not found")
    print("\nSetup instructions:")
    print("1. Copy .config.example to .config")
    print("   $ cp .config.example .config")
    print("2. Edit .config and add your Anthropic API key")

except ValueError as e:
    print(f"✗ Configuration error: {e}")
    print("\nPlease check your .config file and ensure:")
    print("- API key is set correctly")
    print("- All required fields are present")

except Exception as e:
    print(f"✗ Unexpected error: {e}")
