#%%
#Get videos from channel

# -*- coding: utf-8 -*-

# Sample Python code for youtube.playlistItems.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/guides/code_samples#python
import os

import Credentials 

import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
def get_latest_vids(channel_id):
# Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
     
    playlist_id = 'UU'+channel_id[2:]
    # Get credentials and create an API client
    DEVELOPER_KEY = get_youtube_credential()

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version,  developerKey = DEVELOPER_KEY)

    request = youtube.playlistItems().list(
        part="contentDetails",
        maxResults=50,
        playlistId=playlist_id
    )
    response = request.execute()

    print(response)
    return response

#%%
chan = 'UC-lHJZR3Gqxm24_Vd_AJ5Yw'
response = get_latest_vids(chan)
response
#%%
import json

with open('latest_vid_pewdiepie_content_27_05_19.json', 'w') as outfile:  
    json.dump(response, outfile, indent=4)
#%%
vid_ids = [vid['snippet']['resourceId']['videoId'] for vid in response['items']]
#%%
vid_ids
#%% [markdown]
# # Get comments in thread below a parent comment

#%%

# -*- coding: utf-8 -*-

# Sample Python code for youtube.comments.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/guides/code_samples#python
# https://developers.google.com/youtube/v3/docs/comments/list?apix_params=%7B%22part%22%3A%22id%2C%20snippet%22%2C%22parentId%22%3A%22UgzMz-eeTG_yrMs5eFN4AaABAg%22%7D

import os
import Credentials 
import googleapiclient.discovery

def get_comments_from_thread(parent_id):
    # Disable OAuthlib's HTTPS verification when running locally.
    # DO NOT leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
#     input your developer key below
    
    DEVELOPER_KEY = get_youtube_credential()

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = DEVELOPER_KEY)

    request = youtube.comments().list(
        part="id, snippet",
        parentId=parent_id,
        maxResults=100
    )
    response = request.execute()

    return response

#%% [markdown]
# # Get comment threads below a video
# get_comments_from_thread(555)
#%%
# -*- coding: utf-8 -*-

# Sample Python code for youtube.commentThreads.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/guides/code_samples#python
# https://developers.google.com/youtube/v3/docs/commentThreads/list?apix_params=%7B%22part%22%3A%22id%2Csnippet%22%2C%22id%22%3A%22UgzMz-eeTG_yrMs5eFN4AaABAg%22%7D
# ogq4Cy7F9BQ
import os


import pprint
import googleapiclient.discovery

def get_comment_threads_from_video(video_id, pageToken=None):
    # Disable OAuthlib's HTTPS verification when running locally.
    # DO NOT leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = get_youtube_credential()
   
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = DEVELOPER_KEY)

    request = youtube.commentThreads().list(
        part="id,snippet",
        videoId=video_id,
        maxResults=100,
        pageToken = pageToken
    )
    response = request.execute()

    return response
#%%
# get_comment_threads_from_video()
#%%    
# -*- coding: utf-8 -*-

# Sample Python code for youtube.comments.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/guides/code_samples#python
# https://developers.google.com/youtube/v3/docs/comments/list?apix_params=%7B%22part%22%3A%22id%2C%20snippet%22%2C%22parentId%22%3A%22UgzMz-eeTG_yrMs5eFN4AaABAg%22%7D

import os

import googleapiclient.discovery

def get_comments_from_thread(parent_id):
    # Disable OAuthlib's HTTPS verification when running locally.
    # DO NOT leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
#     input your developer key below
    DEVELOPER_KEY = get_youtube_credential()

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = DEVELOPER_KEY)

    request = youtube.comments().list(
        part="id, snippet",
        parentId=parent_id,
        maxResults=100
    )
    response = request.execute()

    return response




#%%
comments_list_per_vid = list()
vids_count = 0
try:
    for vid_id in vid_ids:  
        print(vids_count)      
        comments_count = 0
        response = get_comment_threads_from_video(vid_id)
        first = True
        nextPage = False

        videoId = response['items'][0]['snippet']['videoId']
        comments_list = list()
        for comment in response['items']:
            comments_list.append(comment['snippet']['topLevelComment']['snippet']['textOriginal'])
            comments_count += 1
        nextPage = response.get('nextPageToken')
        i = 1
        comments_per_video = 4000
        while nextPage is not None and i <comments_per_video/100 :
            i += 1
            print("nextPage",i,nextPage)
            response = get_comment_threads_from_video(vid_id, nextPage)
            for comment in response['items']:
                comments_list.append(comment['snippet']['topLevelComment']['snippet']['textOriginal'])
                comments_count += 1

            nextPage = response.get('nextPageToken')
            # items.extend(response['items'])    
        comments = {"videoId":videoId,"commentsCount":comments_count,"comments":comments_list}; comments_count
        comments_list_per_vid.append(comments)
        vids_count += 1
except:
    print("some error")
#%%
print(nextPage)


#%%
comments_list_per_vid

#%%
import json

with open('comments_videos_pewdiepie_4500.json', 'w') as outfile:  
    json.dump(comments_list_per_vid, outfile, indent=4)


#%%
print(comments_count)


# #%%
# import json

# with open('example_comments.json', 'w') as outfile:  
#     json.dump(response, outfile, indent=4)


# #%%
# with open('youtube_comments.json') as json_file:  
#     data = json.load(json_file)


