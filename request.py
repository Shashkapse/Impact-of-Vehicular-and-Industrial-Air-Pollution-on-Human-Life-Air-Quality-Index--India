import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Imperial Extra-Urban':47})

print(r.json())
