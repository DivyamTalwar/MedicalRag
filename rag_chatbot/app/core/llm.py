import requests
import json
from . import config

class OmegaLLM:
    def __init__(self):
        self.endpoint = config.LLM_ENDPOINT
        self.api_key = config.MODELS_API_KEY
        self.model = "omega"

    def invoke(self, prompt: str):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post(self.endpoint, headers=headers, json=data, timeout=600)
        response.raise_for_status()
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0 and "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
            return {"content": result["choices"][0]["message"]["content"]}
        else:
            raise ValueError("Unexpected response format from LLM API")

CustomLLM = OmegaLLM
