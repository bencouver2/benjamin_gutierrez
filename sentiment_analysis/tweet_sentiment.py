"""
Benjamin Gutierrez Garcia, Evaluacion de Sentiemiento usando Twitter, Mayo 2018
bencouver@gmail.com
""
import sys,codecs,json,re

from pprint import pprint
from collections import Counter

#Headers are off by default i.e. number of lines of each file and Hello World
header_info = 0 

#We have levels 0, 1 and 2 for debug
debug = 0


scores = {} # initialize an empty dictionary
text = []
sentiment =0

def hw():
    print 'Hello, world!'

def lines(fp):
    print str(len(fp.readlines()))

def read_dic(scores,print_scores):
    afinnfile = open(sys.argv[1])
    for line in afinnfile:
      term, score  = line.split("\t")  # The file is tab-delimited. "\t" means "tab character"
      scores[term] = int(score)  # Convert the score to an integer, this creates the dict
      if print_scores == 1:
       print scores.items() # Print every (term, score) pair in the dictionary

def eval_sentiment(scores,data):
    sentiment = 0
    for key in scores.keys(): 
       key1=[]
       key1=key.split()
       if len(key1) != 1:
        if re.search(r'\b' + key + r'\b', data["text"].encode('utf-8')):
             sentiment = sentiment+scores[key]
             if debug == 2: print('    ',' kFFound!',key,'Score:',scores[key])
       else:
        words = []
        words=(data["text"].encode('utf-8')).split()
        if key in words:
             sentiment = sentiment+scores[key]
             if debug == 2: print('    ',' kFound!',key,'Score:',scores[key])
    return sentiment 


def exact_Match(phrase, word):
    b = r'(\s|^|$)' 
    res = re.match(b + word + b, phrase, flags=re.IGNORECASE)
    return bool(res)

def main():
    sent_file = open(sys.argv[1])
    tweet_file = open(sys.argv[2])

    if header_info == 1:
     hw()
     lines(sent_file)
     lines(tweet_file)
    
    #Read the dictionary provided 
    print_scores=0
    read_dic(scores,0)

    data = {} # initialize an empty dictionary
    with codecs.open(sys.argv[2],'rU','utf-8') as f:
      for line in f:
        data = json.loads(line)
        #
        #http://support.gnip.com/sources/twitter/data_format.html
        #https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/intro-to-tweet-json
        #
        #Tweet must be valiated before evaluating the sentiment. Very messy data
        #although dictionary structure helps a lot. First i check that the text field exists,
        #so its a valid tweet and sentiment can be evaluated. If not sentiment=0. Then I
        #filter out non-english messages, which receive a sentiment=0 score. If the language 
        #field is empty sentiment=0. Then we proceed to evaluate the sentiment of the tweet text.
        if 'text' in data: 
          if 'lang' in data:
              if (data['lang']=='en'):
                sentiment = eval_sentiment(scores,data)
                if debug == 1 or debug == 2:
                   print('ENGLISH',data["lang"],'Ubicacion:',data["place"],'text:',data["text"],'fuente:',data["source"],'sentiment=',sentiment)
                else:
                   print(sentiment)
              else:
                sentiment = 0
                if debug == 1 or debug == 2:
                   print('Not English',data["lang"],'sentiment=',0)
                else:
                   print(sentiment) 
          else:
           if debug == 1 or debug == 2:
              print('INVALID Language Field,sentiment=0')
           else:
              sentiment = 0
              print(sentiment)
        else:
          if debug == 1 or debug == 2:
             print('HAS NO TEXT,sentiment=0')
          else:
             sentiment = 0
             print(sentiment)
#      if debug == 1 or debug == 2:
#        print(data.keys()) 



if __name__ == '__main__':
    main()
