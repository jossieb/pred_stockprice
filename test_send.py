import os
import sys
import local
import requests
from urllib.parse import urljoin


symbol = "AD.AS"
username = "stillhaveit"
api_token = local.pan_key
pythonanywhere_host = "www.pythonanywhere.com"
api_base = "https://{pythonanywhere_host}/api/v0/user/{username}/".format(
    pythonanywhere_host=pythonanywhere_host,
    username=username,
)

resp = requests.post(
    urljoin(
        api_base,
        "files/path/home/{username}/static/AD.AS_nextday_trend.png".format(
            username=username
        ),
    ),
    files={"content": r"data_output/AD.AS_nextday_trend.png"},
    headers={"Authorization": "Token {api_token}".format(api_token=api_token)},
)
print(resp.status_code)

resp = requests.post(
    urljoin(
        api_base,
        "files/path/home/{username}/static/AD.AS_predict_reality.png".format(
            username=username
        ),
    ),
    files={"content": r"data_output/AD.AS_predict_reality.png"},
    headers={"Authorization": "Token {api_token}".format(api_token=api_token)},
)
print(resp.status_code)
