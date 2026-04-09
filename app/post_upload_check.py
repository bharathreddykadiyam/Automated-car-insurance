import requests, re
url = 'http://127.0.0.1:8080/assessment'
files = {'file': open('static/uploads/demo_test.jpg','rb')}
resp = requests.post(url, files=files)
html = resp.text
print('Status', resp.status_code)
print('Contains model-error text:', 'Model files not loaded' in html)
# print around Results section
idx = html.find('<h4>Results:')
if idx >= 0:
    print(html[idx:idx+800])
else:
    print('Results section not found')
