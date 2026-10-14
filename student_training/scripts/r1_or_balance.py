"""Print the OpenRouter credit balance in USD (number only). Used by r1_paraphrase_pipeline.sh to cap the rewrite round."""
import json
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
req = urllib.request.Request("https://openrouter.ai/api/v1/credits", headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]})
d = json.load(urllib.request.urlopen(req))["data"]
print(f'{d["total_credits"] - d["total_usage"]:.2f}')
