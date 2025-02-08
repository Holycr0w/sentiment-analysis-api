import requests

url = "http://127.0.0.1:5000/predict"  # URL of the Flask API
data = {"text": "I love this product!"}  # Input text for sentiment analysis

response = requests.post(url, json=data)  # Send POST request
print(response.json())  # Print the API response
