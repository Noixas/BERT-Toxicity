
#%%
import pandas as pd

from textblob import TextBlob

df_train = pd.read_csv('train.csv')
# df_train.dropna(axis=0, inplace=True)


#%%
result = []

for i in range(0, 90000):
    polarity = TextBlob(df_train.comment_text[i]).sentiment.polarity
    comment = {'id':int(df_train.id[i]), 'polarity': polarity, 'comment_text':df_train.comment_text[i]}
    result.append(comment)
    



#%%
import json
pd.Series(result).to_json(orient='values')
with open('train_comment_polarity.json', 'w') as outfile:  
    json.dump(result, outfile, indent=4)
