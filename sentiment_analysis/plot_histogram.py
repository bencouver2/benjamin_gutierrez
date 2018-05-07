import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('sentiment_data.csv', sep=',',header=None, index_col =0)

data.plot(kind='bar')
plt.ylabel('Frequency')
plt.xlabel('Sentiment Score of a Tweet')
plt.title('Sentiment Scores for ten minutes of Tweets in English')

plt.show()
