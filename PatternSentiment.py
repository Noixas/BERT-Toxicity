from pattern.en import sentiment
from textblob import TextBlob
import json


with open('comments_videos_pewdiepie_1500.json', 'r', encoding='utf-8') as json_file:  
    data = json.load(json_file)

result =[]
for vid in data:
    score = []
    p_sum_polarity =0
    p_sum_sub = 0
    tb_sum_polarity =0
    tb_sum_sub = 0
    for comment in vid['comments']:
        comment = (comment.encode('ascii', 'ignore')).decode("utf-8")
        p_polarity = sentiment(comment)[0]
        p_subjectivity = sentiment(comment)[1]
        tb_polarity = TextBlob(comment).sentiment.polarity
        tb_subjectivity = TextBlob(comment).sentiment.subjectivity
        d = {'comment':comment, 'p_polarity': p_polarity, 'p_subjectivity': p_subjectivity,'tb_polarity': tb_polarity, 'tb_subjectivity': tb_subjectivity}
        p_sum_polarity+= p_polarity
        p_sum_sub += p_subjectivity
        tb_sum_polarity+= tb_polarity
        tb_sum_sub += tb_subjectivity
        score.append(d)
    result.append({"videoId": vid['videoId'], "comments":score, "avg_p_polarity": p_sum_polarity/1500, "avg_p_subjectivity": p_sum_sub/1500,"avg_tb_polarity": tb_sum_polarity/1500, "avg_tb_subjectivity": tb_sum_sub/1500})



    with open('polarity_pewdiepie_strip.json', 'w') as outfile:  
        json.dump(result, outfile, indent=4)

