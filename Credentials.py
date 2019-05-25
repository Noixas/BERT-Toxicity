import json
import os

def get_youtube_credential():
    with open('Private/credentials.json') as creds:    
        credentials = json.load(creds)
    return credentials['youtube_api']['DEVELOPER_KEY']
print(get_youtube_credential())