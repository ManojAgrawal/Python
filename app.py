
# coding: utf-8

# In[1]:

from flask import Flask,jsonify,json
from flask import request
from flask import render_template
import sys
import numpy
import nltk
import nltk.data
import collections
import json
#import yesno
from bs4 import BeautifulSoup
from pycorenlp import StanfordCoreNLP
import os
import sys
import nltk.data
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
lemma = WordNetLemmatizer()


app = Flask(__name__)

sent_detector = nltk.data.load("tokenizers/punkt/english.pickle")
nlp = StanfordCoreNLP('http://localhost:9000')    


# Hardcoded word lists
yesnowords = ["can", "could", "would", "is", "does", "has", "was", "were", "had", "have", "did", "are", "will", "wa"]
commonwords = ["the", "a", "an", "is", "are", "were", "."]
questionwords = ["who", "what", "where", "when", "why", "how", "whose", "which", "whom"]

# Process article file
article = open("beatles.txt", 'r')
article = BeautifulSoup(article, "lxml").get_text()
article = ''.join([i if ord(i) < 128 else ' ' for i in article])
article = article.replace("\n", " . ")
article = sent_detector.tokenize(article)

# Take in a tokenized question and return the question type and body
def processquestion(qwords):
    
    # Find "question word" (what, who, where, etc.)
    questionword = ""
    qidx = -1

    for (idx, word) in enumerate(qwords):
        if word.lower() in questionwords:
            questionword = word.lower()
            qidx = idx
            break
        elif word.lower() in yesnowords:
            return ("YESNO", qwords)

    if qidx < 0:
        return ("MISC", qwords)

    if qidx > len(qwords) - 3:
        target = qwords[:qidx]
    else:
        target = qwords[qidx+1:]
    type = "MISC"

    # Determine question type
    if questionword in ["who", "whose", "whom"]:
        type = "PERSON"
    elif questionword == "where":
        type = "PLACE"
    elif questionword == "when":
        type = "TIME"
    elif questionword == "how":
        if target[0] in ["few", "little", "much", "many"]:
            type = "QUANTITY"
            target = target[1:]
        elif target[0] in ["young", "old", "long"]:
            type = "TIME"
            target = target[1:]

    # Trim possible extra helper verb
    if questionword == "which":
        target = target[1:]
    if target[0] in yesnowords:
        target = target[1:]
    
    # Return question data
    return (type, target)

def answeryesno(article, question):
    prev = "no"
    questionstr = ' '.join(question)
    questionstr = questionstr.lower()
    question = nltk.pos_tag(question)
    answer = "no"
    keyword = ""
    for (word,pos) in question:
        if (pos == 'NN' or pos == 'NNS' or pos == 'NNP' or pos == 'NNPS'):
            keyword = word.lower()
            answer = "no"
    for sentence in article:
       # print sentence
        if answer == "yes":
            break
        s = nltk.word_tokenize(sentence.lower())
        if keyword in s:
            #print sentence
            answer = "yes"
            for (word,pos) in question:
                if answer == 'no': 
                    break
                if (pos != '.') and (word.lower() not in s) and (pos != 'DT') and (word != 'does') and (word != 'do'):
                    answer = 'no'                    
                    #print word, pos
                    if pos[0] == 'V':
                        tempword = nltk.stem.wordnet.WordNetLemmatizer().lemmatize(word,'v')                       
                        for (w,p) in nltk.pos_tag(s):
                            if p[0] == 'V':
                                tempword2 = nltk.stem.wordnet.WordNetLemmatizer().lemmatize(w,'v')
                                if tempword == tempword2:
                                    answer = 'yes'
                    elif word in article[0]:                       
                        answer = "yes"
                if prev == "yes":
                    if (word == "no" or word =="not"):
                        answer = "no"
                if pos[0] == 'V':
                    prev = "yes"
                else:
                    prev = "no"

    #print questionstr,answer
    return answer                   


@app.route('/')
def my_form():
    return render_template("my-form.html")

@app.route('/', methods=['POST'])
def my_form_post():
    
    question = request.form['text']
    
    done = False

# Tokenize question
    qwords = nltk.word_tokenize(question.lower().replace('?', ''))
    questionPOS = nltk.pos_tag(qwords)
    qwords = [lemma.lemmatize(q) for q in qwords]

# Process question
    (type, target) = processquestion(qwords)
    
    if type == "YESNO":
        answer = answeryesno(article, qwords)
        return answer
        
    # Get sentence keywords
    searchwords = set(target).difference(commonwords)
    dict = collections.Counter()
        
# Find most relevant sentences
    for (i, sent) in enumerate(article):
        sentwords = nltk.word_tokenize(sent.lower())
        sentwords = [lemma.lemmatize(s) for s in sentwords]
        wordmatches = set(filter(set(searchwords).__contains__, sentwords))
        dict[sent] = len(wordmatches)
           
    answer = []
    for (sentence, matches) in dict.most_common(5):
        parse = nlp.annotate(sentence,
                    properties={
                        'annotators': 'ner',
                        'outputFormat': 'json',
                        'timeout': 1000,
                       })
        sentencePOS = nltk.pos_tag(nltk.word_tokenize(sentence))
        done = False
    # Attempt to find matching substrings
        searchstring = ' '.join(target)
        if searchstring in sentence.lower():
#            startidx = sentence.lower().index(target[0])
#            endidx = sentence.lower().index(target[-1])
            answer.append(sentence)
            done = True
    
    # Check if solution is found
        if done:
            continue

    # Check by question type
#    answer = ""
#       for worddata in parse["sentences"][0]["words"]:
        for worddata in parse["sentences"][0]["tokens"]:    
        # Mentioned in the question
#        if worddata["word"] in searchwords:
#            continue
            if done == False:
        
                if type == "PERSON":
                    if worddata["ner"] == "PERSON":
                        answer.append(sentence)
                        done = True
#            elif done:
#                break
    # Check if solution is found
                if done:
                    continue

                if type == "PLACE":
                    if worddata["ner"] == "LOCATION":
                        answer.append(sentence)
                        done = True
#            elif done:
#                break
# Check if solution is found
                if done:
                    continue

                if type == "QUANTITY":
                    if worddata["ner"] == "NUMBER":
                        answer.append(sentence)
                        done = True
#            elif done:
#                break
# Check if solution is found
                if done:
                    continue

                if type == "TIME":
                    if worddata["ner"] == "NUMBER":
                        answer.append(sentence)
                        done = True
#            elif done:
#                answer = sentence
#                break
    
    
    if done:
#    print(answer)
        return jsonify({'matched phrases':answer})
    if not done:
        (answer, matches) = dict.most_common(1)[0]
        return jsonify({'note': "couldn't find exact matches", 'some close matches':answer})            

if __name__ == '__main__':
#    app.debug = True
    app.run()


# In[ ]:



