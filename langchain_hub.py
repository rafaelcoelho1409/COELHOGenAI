from lxml import html
import requests
headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"}
page = requests.get('https://smith.langchain.com/hub?page=2', headers = headers)
tree = html.fromstring(page.content)
print(page.content)
result = tree.xpath('//h4')
print(result)
