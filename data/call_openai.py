import os
import time
import requests
import json
from pathlib import Path
import openai
from openai import AzureOpenAI

def get_oauth_token(p_token_url, p_client_id, p_client_secret, p_scope):
    file_name = "py_llm_oauth_token.json"
    try:
        base_path = Path(__file__).parent
        file_path = Path.joinpath(base_path, file_name)
    except Exception as e:
        print(f"Error occurred while setting file path: {e}")
        return None
    try:
        # Check if the token is cached
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                token = json.load(f)
        else:
            # Get a new token from the OAuth server
            response = requests.post(
                p_token_url,
                data={"grant_type": "client_credentials", "client_id": p_client_id,
                      "client_secret": p_client_secret, "scope": p_scope}
            )
            response.raise_for_status()
            token = response.json()
            with open(file_path, "w") as f:
                json.dump(token, f)
    except Exception as e:
        print(f"Error occurred while getting OAuth token: {e}")
        return None

    try:
        # Check if the token is expired
        expires_in = time.time() + token["expires_in"]
        if time.time() > expires_in:
            # Refresh the token
            token = get_oauth_token(p_token_url, p_client_id,
                                    p_client_secret, p_scope)
    except Exception as e:
        print(f"Error occurred while while getting OAuth token: {e}")
        return None

    authToken = token["access_token"]
    return authToken

def get_config_list():
    # Define your credentials and URL
    client_id = "nvssa-prd-AoqM2A3gY2AE_p-q-Z2Nj53vcoYWjhaJMp8iT6L1h7k"
    client_secret = "ssap-KdxFMWX2XoKD7ip26Tg"
    # Please use this URL for retrieving token https://prod.api.nvidia.com/oauth/api/v1/ssa/default/token
    token_url = "https://prod.api.nvidia.com/oauth/api/v1/ssa/default/token"
    # Please use this Scope for Azure OpenAI: azureopenai-readwrite
    scope = "azureopenai-readwrite"
    token = get_oauth_token(token_url, client_id, client_secret, scope)
    print('Token: ' + token)
    print(token)
    # Define OPENAI Variables and URL
    openai.api_type = "azure"
    openai.api_base = "https://prod.api.nvidia.com/llm/v1/azure/"
    openai.api_version = "2023-03-15-preview"
    openai.api_key = token
    client = AzureOpenAI(
        #api_version=openai.api_base,
        api_version="2023-07-01-preview",
        api_key=token,
        # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
        azure_endpoint="https://prod.api.nvidia.com/llm/v1/azure/",
    )
    config_list = [{
        # "model": "meta/llama3-70b-instruct",
        "model": "gpt-4o",
        # "model": args.planner_model,
        "api_key": token,
        "api_type": "azure",
        "base_url": "https://prod.api.nvidia.com/llm/v1/azure/",
        "api_version": "2023-07-01-preview",
    }]
    return config_list, client
    
config_list, LLM_client = get_config_list()

response = LLM_client.chat.completions.create(
            model=config_list[0]['model'],
            messages = [
                {"role": "system", "content":  "You are a helpful assistant."},
]
        )

print(f"response: {response}")