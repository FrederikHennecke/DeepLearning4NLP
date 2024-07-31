### ressources
- ACL (NAACL)
- use [elicit](https://elicit.com/)

## general idea:
- pooling layers nach jedem layer
- lr scheduler verwenden (sofia)
- habt ihr Lust RLHF zu implementieren

## task specific improvements
### sentiment specific
- sentiment scores von [Vader](https://ojs.aaai.org/index.php/ICWSM/article/view/14550/14399) (rule-based model for general sentiment  analysis) als feature vor dem letzten fully connected layer einfügen
- [This](http://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html) is a dataset containing 191,731 tweets classified into four sentiment catagories (anger, fear, joy, sadness)
- I have a balanced dataset of classified tweets into sensitivity [1=sensitive, 0=not sensitive] (n=2870). I am unsure whether we are allowed to use it for our purposses. I will ask. We have used it in comination with BERT and recieved an accuracy of 86% on a test split. I still have the notebook. Didn't test it on our task. Looks promissing considering that currently we have 52.2% accuracy. [Authors](https://petsymposium.org/popets/2019/popets-2019-0059.pdf)
- [they](https://colab.research.google.com/github/DhavalTaunk08/NLP_scripts/blob/master/sentiment_analysis_using_roberta.ipynb#scrollTo=WTdfPjhFqExX) achieve almost 70% accuracy on the same dataset using a fine tuned model called RoBERTa.
- Considering that we have more data set available for finetuning we could train more layers, e.g. another attention layer solely for sentiment classification. I don't know much about this tho and it might not be a good idea.

###
- 
