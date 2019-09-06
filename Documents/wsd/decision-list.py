
#Programming Assignment II - Word Sense Disambiguation(WSD) usinng decision list
#Team Members: Neha Patil, Neha Nagarkar
#Team Name: Power Team
#Class: AIT 690
# Date: April 4, 2019
# This is decision-list.py file where we have implemented decision list classifier to perform WSD
# Accuracy of the model using features is 80.95% as compared to that of the accuracy of the most frequent baseline is 60%
# We are obtaining left and right unigrams as well as bigrams around the word 'line' as features. Obtaining the log score for each of these features and then using the maximum log likelihood to obtain the best sense for the words in testing set.
# Decision list displays absolute log score
# The run time of the program is approximately 81 seconds
# The program prints the decision list into the my-decision-list.txt file
# The program also prints the classified classes alongwith the instance id which should be taken in -> my-line=answers.txt
# The scorer.py file is used for calculating accuracy and confusion matrix
# We have read the unigrams and bigrams in different lists and we are calculating log score using function calculation_logScore
# The calculated log score is stored into a dataframe for later use
# The test corpora is then read into a dataframe which has all the unigrams and bigrams (on both sides) with respect to target word
# These unigrams and bigrams are then passed through a number of tests and checked for a match
# The sense which has maximum log score is assigned for the target word.

import re
import pandas as pd
import xml.etree.ElementTree as ET
import math
import sys
import os

argList = sys.argv

cwd = os.getcwd()

########################################################################################################
## Reading the xml training file and pre-processing the data
## pre-processing includes lowering all the characters, removing spaces and removing special characters.

tree = ET.parse(argList[1])
root = tree.getroot()
dfcols = ['id', 'senseid', 'line']
df_xml = pd.DataFrame()
doc_id_list=[]
l=[]
x1={}
doc={}
doc['id']=''
doc['senseid']=''
doc['line']=''
for child in root.findall('lexelt'):
    for child1 in child.findall('instance'):
            doc={}
            id1=child1.get('id')
            x=child1.find('answer')
            y=x.get('senseid')
            doc['id']=id1
            doc['senseid']=y
            for txt in child1.findall('context'):
                line=[]
                for s1 in txt.findall('s'):
                    s=ET.tostring(s1,encoding='unicode',method='xml')
                    x1=re.sub('<s>', '', s)
                    x2=re.sub('</s>', '', x1)
                    line.append(x2)
                doc['line']=line
            l.append(doc)
l1 = pd.DataFrame(l)
sense_list=l1['senseid']
## The next block of code is normalizing the input
def normalize_document(doc):   ## function for pre-processing
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    return doc
docs=[]
for line in l1['line']:
    new_list=[]
    for sentence in line:
        new_list.append(normalize_document(sentence))
    docs.append(new_list)   
new_l=[]
for i in docs:
    x=''
    for j in i:
        x=x+' '+''.join(j)
    y=x.strip()
    new_l.append(y)    
text=''
for j in new_l:
    text=text+' '+''.join(j)
text=text.strip()
## The block below is used for finding and making a list of words that precede 
##and follow the target word (line/lines) in the training corpora
dict_bigram=[]
for i in docs:
    bigram=[]
    for j in i:
        if re.match(r".*headlines?head.*",j):
            r1 = re.search(r"(?:[a-zA-Z'-]+[^a-zA-Z'-]+){0,2}headlines?head(?:[^a-zA-Z'-]+[a-zA-Z'-]+){0,2}", j)
            bigram.append(r1.group())
        else:
            continue
    dict_bigram.append(bigram)    
final_dict=[]
for i in dict_bigram:
    if i != []:
        for j in i:
            if re.match(r".*headlinehead.*",j):
                x=re.sub(r"(.*)headlinehead(.*)",r"\1line\2",j)
                final_dict.append(x)
            else:
                x=re.sub(r"(.*)headlineshead(.*)",r"\1lines\2",j)
                final_dict.append(x)
    else:
        final_dict.append('chat line')
len(final_dict)
unigram_left=[]
for i in final_dict:
    r1 = re.search(r"(?:[a-zA-Z'-]+[^a-zA-Z'-]+){0,1}lines?", i)
    x=r1.group()
    r2 = re.sub(r"(.*)\slines?",r"\1",x)
    unigram_left.append(r2.strip())
unigram_right=[]
for i in final_dict:
    r1 = re.search(r"lines?(?:[^a-zA-Z'-]+[a-zA-Z'-]+){0,1}", i)
    x=r1.group()
    r2 = re.sub(r"lines?\s(.*)",r"\1",x)
    unigram_right.append(r2.strip())  
bigram_left=[]
for i in final_dict:
    r1 = re.search(r"(?:[a-zA-Z'-]+[^a-zA-Z'-]+){0,2}lines?", i)
    x=r1.group()
    r2=re.sub(r"(.*)\slines?",r"\1",x)
    bigram_left.append(r2.strip())
bigram_right=[]
for i in final_dict:
    r1 = re.search(r"lines?(?:[^a-zA-Z'-]+[a-zA-Z'-]+){0,2}", i)
    x=r1.group()
    r2=re.sub(r"lines?\s(.*)",r"\1",x)
    bigram_right.append(r2.strip()) 
###############################################################################
## Reading xml test file and pre-processing the test corpus
## pre-processing includes lowering all the characters, removing spaces and removing special characters.
treet = ET.parse(argList[2])
roott = treet.getroot()
lt=[]
doc1={}
doc1['id']=''
doc1['line']=''
for child in roott.findall('lexelt'):
    for child1 in child.findall('instance'):
            doc1={}
            id1=child1.get('id')
            doc1['id']=id1
            for txt in child1.findall('context'):
                line=[]
                for s1 in txt.findall('s'):
                    s=ET.tostring(s1,encoding='unicode',method='xml')
                    x1=re.sub('<s>', '', s)
                    x2=re.sub('</s>', '', x1)
                    line.append(x2)
                doc1['line']=line
            lt.append(doc1)
lt1 = pd.DataFrame(lt)
docst=[]
for line in lt1['line']:
    new_list=[]
    for sentence in line:
        new_list.append(normalize_document(sentence))
    docst.append(new_list)
new_lt=[]
for i in docst:
    x=''
    for j in i:
        x=x+' '+''.join(j)
    y=x.strip()
    new_lt.append(y)
df_test=pd.DataFrame()
df_test['lines']=new_lt
## The block below is used for finding and making a list of words that precede 
##and follow the target word (line/lines) in the test corpora    
dict_gramt=[]
for i in docst:
    gram=[]
    for j in i:
        if re.match(r".*headlines?head.*",j):
            r1 = re.search(r"(?:[a-zA-Z'-]+[^a-zA-Z'-]+){0,6}headlines?head(?:[^a-zA-Z'-]+[a-zA-Z'-]+){0,6}", j)
            gram.append(r1.group())
        else:
            continue
    dict_gramt.append(gram)     
final_dict1=[]
for i in dict_gramt:
    if i != []:
        for j in i:
            if re.match(r".*headlinehead.*",j):
                x=re.sub(r"(.*)headlinehead(.*)",r"\1line\2",j)
                final_dict1.append(x)
            else:
                x=re.sub(r"(.*)headlineshead(.*)",r"\1lines\2",j)
                final_dict1.append(x)
unigram_leftt=[]
for i in final_dict1:
    r1 = re.search(r"(?:[a-zA-Z'-]+[^a-zA-Z'-]+){0,1}lines?", i)
    x=r1.group()
    r2 = re.sub(r"(.*)\slines?",r"\1",x)
    unigram_leftt.append(r2.strip())
unigram_rightt=[]
for i in final_dict1:
    r1 = re.search(r"lines?(?:[^a-zA-Z'-]+[a-zA-Z'-]+){0,1}", i)
    x=r1.group()
    r2 = re.sub(r"lines?\s(.*)",r"\1",x)
    unigram_rightt.append(r2.strip()) 
bigram_leftt=[]
for i in final_dict1:
    r1 = re.search(r"(?:[a-zA-Z'-]+[^a-zA-Z'-]+){0,2}lines?", i)
    x=r1.group()
    r2=re.sub(r"(.*)\slines?",r"\1",x)
    bigram_leftt.append(r2.strip())
bigram_rightt=[]
for i in final_dict1:
    r1 = re.search(r"lines?(?:[^a-zA-Z'-]+[a-zA-Z'-]+){0,2}", i)
    x=r1.group()
    r2=re.sub(r"lines?\s(.*)",r"\1",x)
    bigram_rightt.append(r2.strip()) 
prevt=[]
for i in final_dict1:
    r1 = re.search(r"(?:[a-zA-Z'-]+[^a-zA-Z'-]+){0,6}lines?", i)
    x=r1.group()
    r2=re.sub(r"(.*)\slines?",r"\1",x)
    l=r2.split()
    prevt.append(l)
df_test['unigram_rightt']=unigram_rightt
df_test['unigram_leftt']=unigram_leftt
df_test['bigram_leftt']=bigram_leftt
df_test['bigram_rightt']=bigram_rightt
df_test['lines']=new_lt
df_test['id']=lt1['id']


#### Calculating count of each class in the training corpus####################
phone_count=0
for i in range(len(l1)):
    if l1['senseid'][i]=='phone':
        phone_count=phone_count+1
product_count=0
for i in range(len(l1)):
    if l1['senseid'][i]=='product':
        product_count=product_count+1
#################################################################################
### function for calculating log
        
def calculation_logScore(word_list,sense_list):
    df=pd.DataFrame()
    df['senseid']=sense_list
    df['word']=word_list
    log_val=0
    count=0
    prob_val=0
    df_new={}
    df_new['senseid']=[]
    df_new['word']=[]
    df_new['count']=[]
    for i in range(len(df)):
        count=0
        for j in range(len(df)):
            if df['senseid'][j]==df['senseid'][i] and df['word'][i]==df['word'][j]:
                count=count+1
        x=df['senseid'][i]
        y=df['word'][i]
        df_new['count'].append(count)
        df_new['senseid'].append(x)
        df_new['word'].append(y)
    data=pd.DataFrame(df_new)
    data=data.drop_duplicates()
    data=data.reset_index()
    data=data.drop(columns=['index'])
    mel_count=0
    list1=[]
    word=''
    for i in data['word']:
        word=i
        mel_count=text.count(word)
        list1.append(mel_count)
    data['word_freq']=list1
    prob_val=0
    list2=[]
    for i in range(len(data)):
        for j in range(len(data)):
            if data['word'][i]==data['word'][j] and data['senseid'][i]!=data['senseid'][j]:
                prob_val=data['count'][i]/data['word_freq'][i]
            elif data['word'][i]==data['word'][j] and data['senseid'][i] == 'phone':
                prob_val=data['count'][i]/phone_count
            elif data['word'][i]==data['word'][j] and data['senseid'][i] == 'product':
                prob_val=data['count'][i]/product_count          
        list2.append(prob_val)
        len(data)
    data['prob']=list2
    list3=[]
    for i in range(len(data)):
        for j in range(len(data)):
            if data['word'][i]==data['word'][j] and data['senseid'][i]!=data['senseid'][j]:
                log_val=abs(math.log(data['prob'][i]/data['prob'][j]))
            elif data['prob'][i]<=0.0:
                log_val=0.0
            else:
                log_val=abs(math.log(data['prob'][i]))
        list3.append(log_val)
    data['log']=list3
    return data
##### calculation of log score for each feature and storing in a dataframe with corresponding features
new_unigram_left=pd.DataFrame()
new_unigram_right=pd.DataFrame()
new_bigram_left=pd.DataFrame()
new_bigram_right=pd.DataFrame()
new_unigram_left=calculation_logScore(unigram_left,sense_list)
new_unigram_right=calculation_logScore(unigram_right,sense_list)
new_bigram_left=calculation_logScore(bigram_left,sense_list)
new_bigram_right=calculation_logScore(bigram_right,sense_list)

#####################################################################################################
# Here we are Assigning appropriate sense to each target word in the test corpus using a no. of tests
################## Writing the rules of decision list to a text file

sense=[]
idlist=[]
f1= open(argList[3],"w+")
for i in range(len(df_test)):
    s1=''
    s2=''
    s3=''
    s4=''
    score1=0.0
    score2=0.0
    score3=0.0
    score4=0.0
    for j in range(len(new_unigram_left)):
        if df_test['unigram_leftt'][i]==new_unigram_left['word'][j]:
            s1=new_unigram_left['senseid'][j]
            score1=new_unigram_left['log'][j]
            break
    for j in range(len(new_unigram_right)):
        if df_test['unigram_rightt'][i]==new_unigram_right['word'][j]:
            s2=new_unigram_right['senseid'][j]
            score2=new_unigram_right['log'][j]
            break
    for j in range(len(new_bigram_left)):
        if df_test['bigram_leftt'][i]==new_bigram_left['word'][j]:
            s3=new_bigram_left['senseid'][j]
            score3=new_bigram_left['log'][j]
            break
    for j in range(len(new_bigram_right)):
        if df_test['bigram_rightt'][i]==new_bigram_right['word'][j]:
            s4=new_bigram_left['senseid'][j]
            score4=new_bigram_left['log'][j]
            break
    if s1=='' and s2=='' and s3=='' and s4=='':
        if re.match(r".*products?.*",df_test['lines'][i]):
            sense.append('product')
            f1.write("If new word, check if the test data has product word in it - assign sense product\n")
            f1.write("---------------------------------------------------------------------------------------------------\n")
        else:
            sense.append('phone')
            f1.write("If new word, check if the test data has phone word in it - assign sense phone\n")
            f1.write("---------------------------------------------------------------------------------------------------\n")
    elif s1!='' or s2!='' and s3=='' and s4=='':
        if re.match(r".*products?.*",df_test['lines'][i]):
            sense.append('product')
            f1.write("If the word exists, check if the test data has product word in it - assign sense product\n")
            f1.write("Log Likelihood:%f\n"%score1)
            f1.write("---------------------------------------------------------------------------------------------------\n")
        elif re.match(r".*(tele)?phones?.*",df_test['lines'][i]):
            sense.append('phone')
            f1.write("If the word exists, check if the test data has phone word in it - assign sense phone\n")
            f1.write("Log Likelihood:%f\n"%score1)
            f1.write("---------------------------------------------------------------------------------------------------\n")
        else:
            largest_log=max(score1,score2)
            if largest_log==score1: 
                sense.append(s1)
                f1.write("Sense: "+s1+"\n")
                f1.write("Log score:%f\n"%largest_log)
                f1.write("---------------------------------------------------------------------------------------------------\n")
            else: 
                sense.append(s2)
                f1.write("Sense: "+s2+"\n")
                f1.write("Log score:%f\n"%score2)
                f1.write("---------------------------------------------------------------------------------------------------\n")
    else:
        if s1==s2 and s2==s3 and s3==s4:
                sense.append(s1)
                f1.write("If all relevant words from test data have " + s1 + " sense from training, then assign " + s1+"\n")
                f1.write("Log Likelihood:%f\n"%score1)
                f1.write("---------------------------------------------------------------------------------------------------\n")
        elif s1==s2 and s2 != s3 and s2==s4:
                if score2 > score3:
                    sense.append(s2)
                    f1.write("If few words from test data have different sense from training, then assign the one with greater log likelihood - assign" + s2+"\n")
                    f1.write("Log Likelihood:%f\n"%score2)
                    f1.write("---------------------------------------------------------------------------------------------------\n")
                else:
                    sense.append(s3)
                    f1.write("If few words from test data have different sense from training, then assign the one with greater log likelihood - assign" + s3+"\n")
                    f1.write("Log Likelihood:%f\n"%score3)
                    f1.write("---------------------------------------------------------------------------------------------------\n")
        elif s1!=s2 and s2 == s3 and s2==s4:
                if score2 >score1:
                    sense.append(s2)
                    f1.write("If few words from test data have different sense from training, then assign the one with greater log likelihood - assign" + s3+"\n")
                    f1.write("Log Likelihood:%f\n"%score2)
                    f1.write("---------------------------------------------------------------------------------------------------\n")
                else:
                    sense.append(s1)
                    f1.write("If few words from test data have different sense from training, then assign the one with greater log likelihood - assign" + s3+"\n")
                    f1.write("Log Likelihood:%f\n"%score1)
                    f1.write("---------------------------------------------------------------------------------------------------\n")
        elif s1==s2 and s2 != s3 and s3==s4:
                if score2 >score3:
                    sense.append(s2)
                    f1.write("If few words from test data have different sense from training, then assign the one with greater log likelihood - assign" + s3+"\n")
                    f1.write("Log Likelihood:%f\n"%score2)
                    f1.write("---------------------------------------------------------------------------------------------------\n")
                else:
                    sense.append(s3)
                    f1.write("If few words from test data have different sense from training, then assign the one with greater log likelihood - assign" + s3+"\n")
                    f1.write("Log Likelihood:%f\n"%score3)
                    f1.write("---------------------------------------------------------------------------------------------------\n")
        else:
                largest_log=max(score1,score2,score3,score4)
                if largest_log==score1: 
                    sense.append(s1)
                    f1.write("Check for maximum log likelihood for relevant words from training, then assign the one with greater log likelihood - assign" + s1+"\n")
                    f1.write("Log Likelihood:%f\n"%score1)
                    f1.write("---------------------------------------------------------------------------------------------------\n")
                elif largest_log==score2: 
                    sense.append(s2)
                    f1.write("Check for maximum log likelihood for relevant words from training, then assign the one with greater log likelihood - assign" + s2+"\n")
                    f1.write("Log Likelihood:%f\n"%score2)
                    f1.write("---------------------------------------------------------------------------------------------------\n")
                elif largest_log==score3: 
                    sense.append(s3)
                    f1.write("Check for maximum log likelihood for relevant words from training, then assign the one with greater log likelihood - assign" + s3+"\n")
                    f1.write("Log Likelihood:%f\n"%score3)
                    f1.write("---------------------------------------------------------------------------------------------------\n")
                else:
                    sense.append(s4)
                    f1.write("If few words from test data have different sense from training, then assign the one with greater log likelihood - assign" + s4+"\n")
                    f1.write("Log Likelihood:%f\n"%score4)
                    f1.write("---------------------------------------------------------------------------------------------------\n")
    idlist.append(df_test['id'][i])
f1.close()

model=pd.DataFrame()
model['sense']=sense
model['id']=idlist
sense_val=''
id_val=''
for i in range(len(model)):
        sense_val=model['sense'][i]
        id_val=model['id'][i]
        print('<answer instance="'+id_val+'" senseid="'+sense_val+'"/>')
    


