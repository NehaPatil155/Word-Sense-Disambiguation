#Programming Assignment II - POS Tagging
#Team Members: Neha Patil, Neha Nagarkar
#Team Name: Power Team
#Class: AIT 690
# Date: March 19, 2019

#To run the tagger program from command line - provide the program name as tagger.py with arguments: pos-train.txt pos-test.txt > pos-test-with-tags.txt
#This program takes approx 15 minutes to run
#This is tagger for tagging words from training and test corpus
#This code includes two parts: first for tagging training data and second for tagging test data
#Initially the tagging was done using just most likely probability and later on rules were added
#Before adding rules, the accuracy of tagging on test data was around 70.56% before addition of rules
#Rule 1:- If the word ends with 'ers', 'es' or 's' for eg., years, numbers, ties then the tag is NNS
#Rule 2:- When first letter is capital and ends with a dot then the tag is NNP
#Rule 3:- If the word is 'the' or 'a' then assign tag 'DT'
#Rule 4:- If the word is 'but' 'But' 'and' then the tag is 'CC'
#Rule 5:- If the word contains '.' then the tag is NNP
#Rule 6:- If the word is '%' then the tag is NN
#after the addition of rules, accuracy is 80.78%
#We have directly assigned the tags because 90-95% of the times in training dataset, these wors
#words are tagged with the corresponding tag
import re
import pandas as pd
import os
import sys
argList = sys.argv

cwd = os.getcwd()
##Part I
## Reading the training file and pre-processing the input training file includes removing unnecessary characters such as space, [, ],etc.
f=open(argList[1],'r')
text=f.read()
text1=re.sub(r'\n',' ',text)
token=text1.split(' ')
list1 = []
for i in token:
    if i==' ' or i=='[' or i==']' or i=='':
        continue
    else:
        list1.append(i)
tagged_sentence1=[]
#checking for character \/ in the training set and splitting the words with appropriate tags
for i in list1:
    if re.match(r"(.*)\/(.*)/(.*)",i):
        s=re.sub(r"(.*\/.*)/(.*)",r"\1 \2",i)
        x=s.split(" ")
    else:
        x=i.split('/')
    tagged_sentence1.append(x)
tagged_sentence=tagged_sentence1  
for i in range(len(tagged_sentence1)):
    if re.match(r"(.*)|(.*)",tagged_sentence1[i][1]):
        tagged_sentence[i][1]=re.sub(r"(.*)\|(.*)",r"\1",tagged_sentence1[i][1])
    else:
        tagged_sentence[i][1]=tagged_sentence1[i][1]     
# finding unique words, tags and their count from the training dataset 
words=[]
unique_words=[]
for i in range(len(tagged_sentence)):
    words.append(tagged_sentence[i][0])
for i in words:
    if i not in unique_words:
        unique_words.append(i)
tags=[]
unique_tags=[]
for i in range(len(tagged_sentence)):
    tags.append(tagged_sentence[i][1])
for i in tags:
    if i not in unique_tags:
        unique_tags.append(i)
freq={}
freq_table=[]
freq['word']=''
freq['count']=0
count=0
for i in unique_words:
    count=0
    freq['word']=i
    for j in tagged_sentence:
        if i==j[0]:
            count=count+1
        freq['count']=count
    freq_table.append(freq.copy())
f_countTable=pd.DataFrame(freq_table)

freq1={}
freq_table1=[]
freq1['tags']=''
freq1['count']=0
count=0
for i in unique_tags:
    count=0
    freq1['tags']=i
    for j in tagged_sentence:
        if i==j[1]:
            count=count+1
        freq1['count']=count
    freq_table1.append(freq1.copy())
f_tagsTable1=pd.DataFrame(freq_table1)

#############################################################################################################
##This block of code is to find the probability tables for word-tag, previous tag-current tag combinations
def match_func(x):
    for i in range(len(f_tagsTable1)):
        if f_tagsTable1['tags'][i]==x:
            return (f_tagsTable1['count'][i])

tagged_frame=pd.DataFrame(tagged_sentence)
table1=pd.crosstab(tagged_frame[0],tagged_frame[1])
table1.columns[0]
ptable1=table1.copy()
for col in unique_tags:
    for row in unique_words:
        if table1[col][row]==0:
            continue
        else:
            variable1=table1[col][row]/match_func(col)
            ptable1.replace(ptable1[col][row],variable1,inplace=True)   
prev=[]
current=[]
for i in range(len(tagged_sentence)):
    prev1=tagged_sentence[i-1][1]
    current1=tagged_sentence[i][1]
    prev.append(prev1)
    current.append(current1)

df=pd.DataFrame(columns=['prev','current'])
df['prev']=prev
df['current']=current
table2=pd.crosstab(df['prev'],df['current'])     
ptable=table2.copy()
for col in unique_tags:
    for row in unique_tags:
            variable=round(table2[col][row]/match_func(row),4)
            ptable.replace(ptable[col][row],variable,inplace=True)

##############################################################################################
##Part II
########## The next part of the code is to tag the test corpus.#######################
##This part of the code just pre-processes the the pos-test file.
#Before adding rules, the accuracy of tagging on test data was around 70.56% before addition of rules
#Rule 1:- If the word ends with 'ers', 'es' or 's' for eg., years, numbers, ties then the tag is NNS
#Rule 2:- When first letter is capital and ends with a dot then the tag is NNP
#Rule 3:- If the word is 'the' or 'a' then assign tag 'DT'
#Rule 4:- If the word is 'but' 'But' 'and' then the tag is 'CC'
#Rule 5:- If the word contains '.' then the tag is NNP
#Rule 6:- If the word is '%' then the tag is NN
#Rule 7:- If the word is 'for' then the tag is IN
#after the addition of rules, accuracy is 80.78%
#We have directly assigned the tags because 90-95% of the times in training dataset, these wors
#words are tagged with the corresponding tag
testfile=open(argList[2],'r')
lines=testfile.read()
lines1=re.sub(r'\n',' ',lines)
token1=lines1.split(' ')
listpt = []
for i in token1:
    if i==' ' or i=='[' or i==']' or i=='':
        continue
    else:
        listpt.append(i)    
tagged_frame1=pd.DataFrame(listpt)
#########The next part is where the tagging of test corpus is implemented
final_dictt=[]
finalt1={}
finalt1['words']=''
finalt1['tags']=''
list_poss_test=[]
possible_values={}
possible_values['tags']=''
possible_values['Prob']=0.0
for i in ptable1.columns:
    x=tagged_frame1[0][0]
    y=ptable1[i][x]
    if y != 0.0 or y != 0:
        possible_values['tags']=i
        possible_values['Prob']=y
        list_poss_test.append(possible_values.copy())
list_poss_test1=pd.DataFrame(list_poss_test)
if len(list_poss_test1)==1:
    finalt1['words']=tagged_frame1[0][0]
    finalt1['tags']=list_poss_test1['tags'][0]
    firsttag=list_poss_test1['tags'][0]
else:
    finalt1['words']=tagged_frame1[0][0]
    if list(list_poss_test1['Prob']==1.0)==True:
        for j in range(len(list_poss_test1)):
            if list_poss_test1['tags'][j]=='NNP' and finalt1['words'].capitalize() == finalt1['words']:
                finalt1['tags']=list_poss_test1['tags'][j]
                firsttag=list_poss_test1['tags'][j]
                break
            else:
                var=list(list_poss_test1.loc[list_poss_test1['Prob'].idxmax()])
                if len(var)>1:
                    tag1=var[1]
                else:
                    tag1=var[0]
                finalt1['tags']=tag1
                firsttag=tag1                
    else:
        var=list(list_poss_test1.loc[list_poss_test1['Prob'].idxmax()])
        if len(var)>1:
            tag1=var[1]
        else:
            tag1=var[0]
        finalt1['tags']=tag1
        firsttag=tag1
final_dictt.append(finalt1.copy())

## The user defined function to tag the words from test corpus using model the trained model

def tagt(w,t):
    if w in unique_words:
          list_possw=[]
          possible_valuesw={}
          possible_valuesw['tags']=''
          possible_valuesw['Prob']=0.0
          for i in ptable1.columns:
              y1=ptable1[i][w]
              if y1==0.0:
                  continue
              else:
                  possible_valuesw['tags']=i
                  possible_valuesw['Prob']=y1
                  list_possw.append(possible_valuesw.copy())
          list_possw1=pd.DataFrame(list_possw)
          return_tag1=None
          for i in range(len(list_possw1)):
              if (list_possw1['tags'][i]).lower()==w.lower():
                  return_tag1=list_possw1['tags'][i]
              elif (list_possw1['tags'][i]=='DT') and w in ('the','a'):
                  return_tag1=list_possw1['tags'][i]
              elif (list_possw1['tags'][i]=='CC') and w in ('But','but','and'):
                  return_tag1=list_possw1['tags'][i]
              elif (list_possw1['tags'][i]=='NN') and w == '%':
                  return_tag1=list_possw1['tags'][i]
              elif re.match(r"[A-Z]+(.)*-[A-Z]+(.)*",w):
                  if (list_possw1['tags'][i])=='NNP':
                      return_tag1=list_possw1['tags'][i]
              elif re.match(r"[aA-zZ]+[es|s|ers|rs]+",w):
                  if (list_possw1['tags'][i])=='NNS':
                      return_tag1=list_possw1['tags'][i]
              elif re.match(r"(\.)*(.)*(\.)*(.)+(\.)+",w):
                  if (list_possw1['tags'][i])=='NNP':
                      return_tag1=list_possw1['tags'][i]
          return_tag=None
          if len(list_possw1)==1:
              return_tag=list_possw1['tags']
              var=list(return_tag)
              if len(var)>1:
                  return_tag=var[1]
              else:
                  return_tag=var[0]
          else:
              listt=[]
              poss_valuest={}
              poss_valuest['Currtags']=''
              poss_valuest['Prob']=0.0
              y=[]
              for i in ptable.columns:
                  y=ptable[i][t]
                  poss_valuest['Currtags']=i
                  poss_valuest['Prob']=y
                  listt.append(poss_valuest.copy())
              listt1=pd.DataFrame(listt)
              new=[]
              new1={}
              new1['tags']=''
              new1['Prob']=0.0
              for i in range(len(list_possw1)):
                  new1['tags']=list_possw1['tags'][i]
                  for j in range(len(listt1)):
                      if list_possw1['tags'][i]==listt1['Currtags'][j]:
                          new1['Prob']=list_possw1['Prob'][i]*listt1['Prob'][j]
                          new.append(new1.copy())
              temp=pd.DataFrame(new)
              return_tag=temp.loc[temp['Prob'].idxmax(skipna=False)]
              var=list(return_tag)
              if len(var)>1:
                  return_tag=var[1]
              else:
                  return_tag=var[0]
          if return_tag1 is not None:
              return return_tag1
              return_tag1=None
          else:
              return return_tag
    else:
        default='NN'
        return default

for i in range(len(tagged_frame1)-1):
    w=tagged_frame1[0][i+1]
    finalt1['words']=w
    if i==0:
        t=firsttag
        curr=tagt(w,t)
        finalt1['tags']=curr
    else:
        t=curr
        curr1=tagt(w,t)
        finalt1['tags']=curr1
        curr=curr1
    final_dictt.append(finalt1.copy())
##In the below code a tagged file is generated for test corpus
    len(final_dictt)
tagged_framen=pd.DataFrame(final_dictt)
#####################################################################################
final_dictn=[]
final1={}
final1['words']=''
final1['tags']=''
list_poss=[]
possible_values={}
possible_values['tags']=''
possible_values['Prob']=0.0
for i in ptable1.columns:
    x=tagged_framen['words'][0]
    y=ptable1[i][x]
    if y != 0.0 or y != 0:
        possible_values['tags']=i
        possible_values['Prob']=y
        list_poss.append(possible_values.copy())
list_poss1=pd.DataFrame(list_poss)
if len(list_poss1)==1:
    final1['words']=tagged_framen['words'][0]
    final1['tags']=list_poss1['tags'][0]
    firsttag=list_poss1['tags'][0]
else:
    final1['words']=tagged_framen['words'][0]
    if list(list_poss1['Prob']==1.0)==True:
        for j in range(len(list_poss1)):
            if list_poss1['tags'][j]=='NNP' and final1['words'].capitalize() == final1['words']:
                final1['tags']=list_poss1['tags'][j]
                firsttag=list_poss1['tags'][j]
                break
            else:
                var=list(list_poss1.loc[list_poss1['Prob'].idxmax()])
                if len(var)>1:
                    tag1=var[1]
                else:
                    tag1=var[0]
                final1['tags']=tag1
                firsttag=tag1                
    else:
        var=list(list_poss1.loc[list_poss1['Prob'].idxmax()])
        if len(var)>1:
            tag1=var[1]
        else:
            tag1=var[0]
        final1['tags']=tag1
        firsttag=tag1
final_dictn.append(final1.copy())
##The user defined function(tag) tags the word in the training corpus using most 
##likely probability and with the help of some rules
#improving the tagging function using the probability of tag of next word
def tag(w,pt,nt):
    if w in unique_words:
        list_possw=[]
        possible_valuesw={}
        possible_valuesw['tags']=''
        possible_valuesw['Prob']=0.0
        for i in ptable1.columns:
            y1=ptable1[i][w]
            if y1==0.0:
                continue
            else:
                possible_valuesw['tags']=i
                possible_valuesw['Prob']=y1
                list_possw.append(possible_valuesw.copy())
        list_possw1=pd.DataFrame(list_possw)
        return_tag1=None
        for i in range(len(list_possw1)):
            if (list_possw1['tags'][i]).lower()==w.lower():
                return_tag1=list_possw1['tags'][i]
            elif (list_possw1['tags'][i]=='DT') and w in ('the','a'):
                return_tag1=list_possw1['tags'][i]
            elif (list_possw1['tags'][i]=='CC') and w in ('But','but','and'):
                return_tag1=list_possw1['tags'][i]
            elif (list_possw1['tags'][i]=='NN') and w == '%':
                  return_tag1=list_possw1['tags'][i]
            elif (list_possw1['tags'][i]=='IN') and w == 'for':
                  return_tag1=list_possw1['tags'][i]
            elif re.match(r"[A-Z]+(.)*-[A-Z]+(.)*",w):
                if (list_possw1['tags'][i])=='NNP':
                    return_tag1=list_possw1['tags'][i]
            elif re.match(r"[aA-zZ]+[es|s|ers|rs]+",w):
                if (list_possw1['tags'][i])=='NNS':
                    return_tag1=list_possw1['tags'][i]
            elif re.match(r"(\.)*(.)*(\.)*(.)+(\.)+",w):
                if (list_possw1['tags'][i])=='NNP':
                    return_tag1=list_possw1['tags'][i]
        return_tag=None
        if len(list_possw1)==1:
            return_tag=list_possw1['tags']
            var=list(return_tag)
            if len(var)>1:
                return_tag=var[1]
            else:
                return_tag=var[0]
        else:
            listt1=[]
            poss_valuest1={}
            poss_valuest1['Currtags']=''
            poss_valuest1['Prob']=0.0
            y=[]
            for i in ptable.columns:
                y=ptable[i][pt]
                poss_valuest1['Currtags']=i
                poss_valuest1['Prob']=y
                listt1.append(poss_valuest1.copy())
            list_ft1=pd.DataFrame(listt1)
            listt2=[]
            poss_valuest2={}
            poss_valuest2['Currtags']=''
            poss_valuest2['Prob']=0.0
            y2=[]
            for i in ptable.columns:
                y2=ptable[nt][i]
                poss_valuest2['Currtags']=i
                poss_valuest2['Prob']=y2
                listt2.append(poss_valuest2.copy())
            list_ft2=pd.DataFrame(listt2)
            new=[]
            new1={}
            new1['tags']=''
            new1['Prob']=0.0
            for i in range(len(list_possw1)):
                new1['tags']=list_possw1['tags'][i]
                for j in range(len(list_ft1)):
                    if list_possw1['tags'][i]==list_ft1['Currtags'][j]:
                        for k in range(len(list_ft2)):
                            if list_possw1['tags'][i]==list_ft2['Currtags'][k]:
                                new1['Prob']=list_possw1['Prob'][i]*list_ft1['Prob'][j]*list_ft1['Prob'][k]
                new.append(new1.copy())
            temp=pd.DataFrame(new)
            return_tag=temp.loc[temp['Prob'].idxmax(skipna=False)]
            var=list(return_tag)
            if len(var)>1:
                return_tag=var[1]
            else:
                return_tag=var[0]
        if return_tag1 is not None:
            return return_tag1
            return_tag1=None
        else:
            return return_tag
    else:
        default='NN'
        return default
    
for i in range(len(tagged_framen)-1):
    w=tagged_framen['words'][i+1]
    final1['words']=w
    if i==0:
        pt=firsttag
        nt=tagged_framen['tags'][i+2]
        curr=tag(w,pt,nt)
        final1['tags']=curr
    else:
        pt=curr
        nt=tagged_framen['tags'][i+1]
        curr1=tag(w,pt,nt)
        final1['tags']=curr1
        curr=curr1
    final_dictn.append(final1.copy())
##End of tagging
## We are generating a text file of training corpus tagged by our system.
a1=" "
final_string1=[]
pos_test=pd.DataFrame(final_dictn)
len(pos_test)
for i in range(len(pos_test)):
    x=pos_test['words'][i]+"/"+pos_test['tags'][i]
    final_string1.append(x)
for i in range(len(final_string1)):
    print(final_string1[i])




