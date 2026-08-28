"""
Diagnostic script to help identify the correct endpoint configuration.
"""

from ontovis import Config
import anthropic

def test_endpoint_variations():
    """Test different endpoint URL variations."""
    config = Config()
    llm_config = config.get_llm_config()

    base_urls_to_try = [
        llm_config['base_url'],  # Original
        llm_config['base_url'].rstrip('/v1'),  # Without /v1
        f"{llm_config['base_url'].rstrip('/v1')}/messages",  # Direct to messages
        llm_config['base_url'].replace('/v1', ''),  # No version
    ]

    models_to_try = [
        llm_config['model'],  # Original
        llm_config['model'].split('.')[-1],  # Just the model part
        "claude-sonnet-4-5",  # Standard format
    ]

    print("Testing different endpoint configurations...")
    print("=" * 60)

    for base_url in base_urls_to_try:
        for model in models_to_try:
            print(f"\nTrying:")
            print(f"  URL: {base_url}")
            print(f"  Model: {model}")

            try:
                client = anthropic.Anthropic(
                    api_key=llm_config['api_key'],
                    base_url=base_url
                )

                message = client.messages.create(
                    model=model,
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Hi"}]
                )

                print(f"  ✓ SUCCESS!")
                print(f"  Response: {message.content[0].text}")
                print(f"\n✓✓✓ WORKING CONFIGURATION FOUND ✓✓✓")
                print(f"Update your .config file with:")
                print(f"base_url = {base_url}")
                print(f"model = {model}")
                return True

            except anthropic.NotFoundError as e:
                print(f"  ✗ 404 Not Found")
            except anthropic.AuthenticationError as e:
                print(f"  ✗ Authentication Error (but endpoint exists!)")
                print(f"    This means the URL/model combo is correct,")
                print(f"    but API key might be invalid")
            except Exception as e:
                print(f"  ✗ Error: {type(e).__name__}: {str(e)[:100]}")

    print("\n" + "=" * 60)
    print("No working configuration found.")
    print("\nPlease check with your LANL AI Portal documentation for:")
    print("1. The correct base URL format")
    print("2. The correct model ID format")
    print("3. Whether you need special headers or authentication")
    return False


if __name__ == "__main__":
    test_endpoint_variations()
