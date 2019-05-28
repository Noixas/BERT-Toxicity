# Sentiment Analysis on the comments of Pewdiepie videos
## Introduction
Pewdiepie has become the biggest English speaking channel on Youtube with currently more than 95 million subscribers. With this huge amount of audience, it is interesting to see how the audience reacts to his videos.

In our paper we analyse the comments of 50 recent videos of Pewdiepie and we evaluate the polarity and toxicity leveraging libraries like TextBlob and the pre trained model BERT.

## Datasets
- 200,000 top-threaded comments scraped from Pewdiepie videos with Youtube API (4,000 per video, 50 videos)
- [Jigsaw Unintended Bias in Toxicity Classification dataset](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/)   that contains sentences with a score of toxicity with multiple labels like (”toxic”, ”severe toxic”, ”obscene”, ”threat”, ”insult”, ”identity hate”) indicating the type of toxicity.From this dataseth we used the first 90,000 entries of the training set to fine tune BERT

## Experiment

In order to analyse the comments, we used the [TextBlob](https://textblob.readthedocs.io/en/dev/) and [Pattern](https://www.clips.uantwerpen.be/pattern) libraries to score the sentiment polarity per comment and then averaged them per video being our general polarity score for such video. Then we fine tuned BERT using [Pytorch](https://github.com/pytorch/pytorch) and the [JUBTC](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/) dataset in Google Colab to score the videos toxicity (Our tuned model yielded a .908 accuracy value in the validation set). 

## Results
### Polarity results
Categorized videos with the top 5 highest and least polarity (Higher is more positive):

Rank | Video ID | Category | Pattern Polarity | TextBlob Polarity
---- | ---------| -------- | ---------------- | ------------
1 | qPnTTA8BC8A | Book review | 0.4933 | 0.4956
2 | C2fRC55rA8w | Travel vlog | 0.3267 | 0.3277
3 | PGbAWTqUuxQ | Hameplay    | 0.3185 | 0.3218
4 | QNLARCvIATo | Travel vlog | 0.2999 | 0.3009
5 | OEUsKLW1th4 | Gameplay    | 0.2640 | 0.2656
46 | WOSC6uGtBFw | Meme review| 0.0935 | 0.0964 
47 | rdaQsl9jqmw | Gameplay   | 0.0901 | 0.0899
48 | wFxCAWqvmBE | Meme review| 0.0628 | 0.0635 
49 | zYZ1Fd7iH90 | Cringe Tue.| 0.0581 | 0.0587 
50 | DCkydkdhL8M | Meme review| 0.0422 |  0.0448

### Toxicity results
Categorized  top  5  toxic  videos  and  least  5 toxic videos (Higher is more toxic):

Rank | Video ID | Category |  Toxicity
---- | ---------| -------- |  ------------
1 | JLREgYXXdB8 | Cringe Tue. | 0.2964
2 | eHYkTUmsJlY | Pew news | 0.1592
3 | JxAUHg8AguA | Cringe Tue. | 0.1536
4 | 4QnLRnKwFM0 | Pew news | 0.1501
5 | 3m4mF9-7L-Y | Pew news | 0.1368
46 | rc1VR54nHV0 | Collab. | 0.0612
47 | OEUsKLW1th4 | Gameplay | 0.0604
48 | wFxCAWqvmBE | Meme re. | 0.0522
49 | C2fRC55rA8w | Travel vlog | 0.0498
50 | qPnTTA8BC8A | Book re. | 0.0482 

## Analysis

- Comments are biased, for example the gameplay of ”Happy Wheel” is the 3rd most positive video while the gameplay of ”The Walking Dead” is in 49th place. The word 'happy' occurred a lot more times since it’s part of the game name which increases the polarity score while the oppositve happens with the word 'dead'.

    Rank | 'Happy' frequency | 'Dead' frequency |  Polarity ( Pattern)
    ---- | ------------------| ---------------- |  ------------
    3 | 438 | 130 | 0.3185 
    49 | 63 | 173 | 0.0581 
    
-  From the full results,  we  found that book review videos are more positive than other categories and also travel vlogs tend to have  higher polarity while meme reviews tend to have lower polarity. The polarity of a gameplay can differ drastically based on the game.
- Pew news and Cringe Tuesday categories remained in the most toxic videos, there could be multiple explanations to this, one of that we found in our results is that the model is biased and categorize wrongly certain sentences. For example the most toxic video "I broke my ass" contains misclassified comments like "I love you and your broken ass" with really toxic score of 0.9687.

- While  toxicity  and  polarity  are  two  different  attributes we found that 5 of the top 10 positive videos are also in the top 10 least toxic videos.  Furthermore 4 of the most negative videos are in the top 10 most toxic videos. The difference in the top10 list can be mainly explain due to the bias andthe different focus of the algorithms where the po-larity of a comment can be low if is sad while itcould remains as not toxic.

## Conclusion

Based on the results, we can conclude that gener-ally the comments of Pewdiepie’s videos are morepositive than negative, and in 80% of the samplevideos, less than 10% of the comments are toxic(Table 7). We also found out that the sentiment po-larity and toxicity somewhat correlates in the top 10% percentile. Finally after analysing the resultswe  discovered  that  the  models  weren’t  unbiasedand further research is recommended.




## Resources
 [Bert fined tunned weights (>400mbs)](https://drive.google.com/open?id=1uECXp5FRwcAKGGcT30R3UM07mUMMXG20)

  