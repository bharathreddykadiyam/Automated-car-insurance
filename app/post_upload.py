import requests

url = 'http://127.0.0.1:8080/assessment'
files = {'file': open('static/uploads/demo_test.jpg','rb')}
resp = requests.post(url, files=files)
print('Status', resp.status_code)
print(resp.text[:2000])
