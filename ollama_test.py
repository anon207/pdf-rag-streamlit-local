import requests

url = "http://localhost:11434/api/generate" 
payload = { "model": "llama3:8b", "prompt": "Reply with exactly: OLLAMA_OK", "stream": False }

r = requests.post(url, json=payload, timeout=120) 
r.raise_for_status() 
print(r.json()["response"].strip())