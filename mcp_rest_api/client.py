# client.py
import requests

url = "http://127.0.0.1:5000/data"

try:
    response = requests.get(url)
    response.raise_for_status()  # raises an error for 4xx/5xx responses
    data = response.json()
    print("Response from server:", data)
except requests.exceptions.RequestException as e:
    print("Error:", e)
