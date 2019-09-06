# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 18:13:40 2019

@author: nehap
"""

# Programming Assignment IV - Information Retrieval
# Team Members: Neha Patil, Neha Nagarkar
# Team Name: Power Team
# Class: AIT 690
# Date: April 17, 2019
# This is ir-system.py file where we have implemented information retrieval based on TFIDF score 
# Provide ir-system.py and cran.qry from the command prompt. You must input a query and the system will 
# retrieve relevant documents and output it to cran-output.txt file, which will then be used to calculate precision and recall
# Here is an example of how to run the program - 
# If we input query 1, the system retrieves 10 relevant documents with document ids - 184, 12, 14, 665, 78, 1313, 1361, 1362, 62, 374
# Therefore, precision for first query = 3/10 = 0.3 where 3 is the number of relevant documents retrieved by the ir-system that matches with cranqrel documents
# And recall for first query = 3/29 = 0.103 where 29 is the number of relevant documents for first query in cranqrel
# The precision and recall for bag of words model is 0.21 and 0.31 respectively
# In order to improve the bag-of-words model we implemented 
# 
import re
import pandas as pd
import sys
import os

argList = sys.argv

cwd = os.getcwd()
import math
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer   
lemmatizer = WordNetLemmatizer()  
wpt = nltk.WordPunctTokenizer()
stopwords = nltk.corpus.stopwords.words('english')
# importing the input document file
f=open(argList[1],'r')
text=f.read()
list1=[]
list2=[]
list3=[]
list4=[]
list5=[]
text3=[]
document_df=pd.DataFrame()
# cleaning the input file
text1=text.split(".I")
for record in range(1,1401):
    x=text1[record].split(".T")
    x1=x[1].split(".A")
    y=x1[1].split(".B")
    y1=y[1].split(".W")
    list1.append(x[0].rstrip('\n').strip())
    list2.append(x1[0].strip('\n').strip())
    list3.append(y[0].strip('\n').strip())
    list4.append(y1[0].strip('\n').strip())
    list5.append(y1[1].strip('\n').strip())

document_df['Id']=list1
document_df['Title']=list2
document_df['Author']=list3
document_df['Publication']=list4
document_df['Text']=list5

temp={}
#function to remover special characters
def remove_string_special_characters(s):
    stripped = re.sub(r'[^a-zA-Z_\-\s]', '', s,re.I|re.A)
    stripped = re.sub('\s+', ' ', stripped)
    tokens = wpt.tokenize(stripped)
    filtered_tokens = [token for token in tokens if token not in stopwords]
    stripped = ' '.join(filtered_tokens)
    stripped = re.sub('\-', '', stripped)
    stripped = stripped.strip()
    return(stripped)


def get_doc(sent,i):
    doc_info = []
    count=0
    sent1=sent.split()
    for words in sent1:
        count=count+1
    temp = {'doc_id': i, 'doc_length': count}
    doc_info.append(temp)
    return doc_info

# function to create a word frequency list
def create_freq_dict(sents,id1):
    freqDict_list = []
    freqDict_list1 = []
    word=sents.split()
    for sent in word:
        if sent not in freqDict_list:
            freqDict_list.append(sent)       
    for i in freqDict_list:
        temp={}
        count=0
        for j in word:
            if i==j:
                count=count+1
        temp = {'doc_id':id1, 'word': i, 'count':count}
        freqDict_list1.append(temp)   
    return freqDict_list1
# function to calculate term frequency of a document
def computeTF(doc, freqDict_list,i):
    """
    tf = (frequency of the term in the doc/total number of terms in the doc)
    """
    TF_scores = []
    for tempDict in freqDict_list:
            temp = {'doc_id':tempDict['doc_id'] ,
                    'TF_score': tempDict['count']/doc[0]['doc_length'],
                    'key':tempDict['word'] }
            TF_scores.append(temp)
            temp={}
    return TF_scores
# function to calculate inverse document frequency    
def computeIDF(doc_info, elem,whole_list,i):
    """
    tf = ln(total number of docs/number of docs with term in it)
    """
    IDF_scores = []
    if len(whole_list)==1400:
        for dict1 in elem:
            count=0
            temp={}
            for j in whole_list:
                for k in j:
                    if dict1['word']==k['word']:
                        count=count+1
                        break
            if count>0:
                logvalue=1.0+math.log(1400/count)
            else:
                logvalue=1.0
            temp = {'doc_id': dict1['doc_id'],'IDF_score': logvalue,
                        'key':dict1['word'],'count':count}
            IDF_scores.append(temp)
    return IDF_scores  
# function to calculate TFIDF
def computeTFIDF(TF_scores, IDF_scores):
    TFIDF_scores = []
    for j in IDF_scores:
        temp={}
        for i in TF_scores:
            if j['key']==i['key'] and j['doc_id']==i['doc_id']:
                temp = {'doc_id': j['doc_id'],
                        'TFIDF_score': j['IDF_score']*i['TF_score'],
                        'key': i['key']}              
        TFIDF_scores.append(temp)
    return TFIDF_scores
list_TFIDF=[]
text_sents=''
text_sents_clean=''
doc_info=[]
freqDict_list=[]
for j in range(len(document_df)):
    text_sents_clean = remove_string_special_characters(document_df['Text'][j])
    doc = get_doc(text_sents_clean,j)
    doc_info.append(doc)
    freqDist = create_freq_dict(text_sents_clean,j)
    freqDict_list.append(freqDist)
# creating a list for TFIDF score for each document body
for i in range(len(document_df)):
    if document_df['Text'][i]!='':
        TF_scores = computeTF(doc_info[i], freqDict_list[i],i)
        IDF_scores = computeIDF(doc_info[i], freqDict_list[i],freqDict_list,i)
        TFIDF_scores = computeTFIDF(TF_scores, IDF_scores)
        list_TFIDF.append(TFIDF_scores)
    else:
        list_TFIDF.append('')
list_TFIDF2=[]
text_sents2=''
text_sents_clean2=''
doc_info2=[]
freqDict_list2=[]
# creating a lidt for TFIDF score for document title
for j in range(len(document_df)):
    text_sents_clean2 = remove_string_special_characters(document_df['Title'][j])
    doc = get_doc(text_sents_clean2,j)
    doc_info2.append(doc)
    freqDist = create_freq_dict(text_sents_clean,j)
    freqDict_list2.append(freqDist)

for i in range(len(document_df)):
    if document_df['Title'][i]!='':
        TF_scores = computeTF(doc_info2[i], freqDict_list2[i],i)
        IDF_scores = computeIDF(doc_info2[i], freqDict_list2[i],freqDict_list2,i)
        TFIDF_scores = computeTFIDF(TF_scores, IDF_scores)
        list_TFIDF2.append(TFIDF_scores)
    else:
        list_TFIDF2.append('')


###############################################################################
## Reading query file and pre-processing the extracted queries

f_qry=open(argList[2],'r')
text_qry=f_qry.read()
list1_qry=[]
list2_qry=[]
list3_qry=[]
df_qry=pd.DataFrame()
text1_qry=text_qry.split(".I")
# cleaning the query file
for record in range(1,226):
    x=text1_qry[record].split(".W")
    list1_qry.append(x[0].rstrip('\n').strip())
    list2_qry.append(x[1].strip('\n').strip())

for j in list2_qry:
    doc = re.sub(r'[^a-zA-Z_\-\s]', '', j, re.I|re.A)
    doc=re.sub(r"(\-)",' ',doc)
    doc=re.sub(r"(\b\s\b)+",' ',doc)
    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stopwords]
    doc = ' '.join(filtered_tokens)
    doc = doc.strip()
    list3_qry.append(doc)
    
df_qry['id']=list1_qry
df_qry['queries']=list2_qry
df_qry['qry']=list3_qry
len(df_qry)
qry_TFIDF=[]
text_sents1=''
text_sents_clean1=''
doc_info1=[]
freqDict_list1=[]
# creating a TFIDF score list for all queries
for j in range(len(df_qry)):
    text_sents_clean1 = remove_string_special_characters(df_qry['qry'][j])
    doc1 = get_doc(text_sents_clean1,j)
    doc_info1.append(doc1)
    freqDist1 = create_freq_dict(text_sents_clean1,j)
    freqDict_list1.append(freqDist1)
for i in range(len(df_qry)):
    TF_scores1 = computeTF(doc_info1[i], freqDict_list1[i],i)
    IDF_scores1 = computeIDF(doc_info1[i], freqDict_list1[i],freqDict_list,i)
    TFIDF_scores1 = computeTFIDF(TF_scores1, IDF_scores1)
    qry_TFIDF.append(TFIDF_scores1)

final_qry12=[]
for k in range(len(df_qry)):
    words=df_qry['qry'][k].split()
    final_qry=[]
    for i in words:
        for j in qry_TFIDF[k]:
            if i==j['key']:
                x=j['TFIDF_score']
        final_qry.append(x)
    final_qry12.append(np.asarray(final_qry))

#############################################################################
# function for calculating cosine similarity between a query and a document
def cosine_simiarity(a,b):
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    if normb !=0.0:
        cos = dot / (norma * normb)
        return cos
    else:
        cos = 0
        return cos
# function for calculating cosine similarity between all queries and document by creating a 2d array
def cos_all(all_poss1,all_poss2,a,u):
    cs_list=[]
    cs_list_t=[]
    idlist=[]
    idlist_t=[]
    for i in  range(len(all_poss1)):
        b= np.asarray(all_poss1[i])
        c=i+1
        cs=cosine_simiarity(a,b)
        if cs > 0:     
            cs_list.append(cs)
            idlist.append(c)
    for i in  range(len(all_poss2)):
        b= np.asarray(all_poss2[i])
        c=i+1
        cs=cosine_simiarity(a,b)
        if cs > 0:     
            cs_list_t.append(cs)
            idlist_t.append(c)
    data1=pd.DataFrame()
    data1['cosine']=cs_list
    data1['docid']=idlist
    sortedlist1=data1.sort_values(by='cosine', ascending=False, na_position='first')
    sortedlist1.reset_index(inplace=True)
    x=sum(data1['cosine'])/len(data1['cosine'])
    idlist1=[]
    cs_list1=[]
    for i in range(len(data1)):
        if data1['cosine'][i]>x:
            cs_list1.append(data1['cosine'][i])
            idlist1.append(data1['docid'][i])
    data2=pd.DataFrame()
    data2['cosine']=cs_list1
    data2['docid']=idlist1       
    sortedlist=data2.sort_values(by='cosine', ascending=False, na_position='first')
    sortedlist.reset_index(inplace=True)
    list10=[]
    if len(sortedlist)>10:
        for i in range(10):
            out={}
            out={'qryid':u+1,'docid':sortedlist['docid'][i]}
            list10.append(out)
    elif len(sortedlist1)>10:
        for i in range(10):
            out={}
            out={'qryid':u+1,'docid':sortedlist1['docid'][i]}
            list10.append(out)
    elif len(sortedlist1)<10:
        for i in range(len(sortedlist1)):
            out={}
            out={'qryid':u+1,'docid':sortedlist1['docid'][i]}
            list10.append(out)
    if len(cs_list_t)==0:
            return list10
    else:
        data=pd.DataFrame()
        data['cosine']=cs_list_t
        data['idlist']=idlist_t
        sorteddata=data.sort_values(by='cosine', ascending=False, na_position='first')
        sorteddata.reset_index(inplace=True)
        if sorteddata['cosine'][0]<x:
            return list10
        else:
            if sorteddata['cosine'][0]<sortedlist1['cosine'][9]:
                return list10
            else:
                for i in range(2):
                    out={}
                    out={'qryid':u+1,'docid':sorteddata['idlist'][i]}
                    list10.append(out)
                list12=[]
                for i in range(len(list10)):
                    for j in range(len(list10)):
                        if list10[i]['docid']!=list10[j]['docid']:
                            list12.append(list10[j])
                return list12
            
#  creating an array for the TFIDFsores of queries w.r.t document body             
def twd(words):
    twod_list = [] 
    for i in range (len(words)):
       new = []                  
       for j in range (len(list_TFIDF)):
            new.append(0.0)
       twod_list.append(new)
                                                             
    for i in range (len(words)):                                              
        for j in range (len(list_TFIDF)):
            for k in range (len(list_TFIDF[j])):
                var=0.0
                if words[i]==list_TFIDF[j][k]['key']:
                    if j == list_TFIDF[j][k]['doc_id']:          
                        var=list_TFIDF[j][k]['TFIDF_score']
                        break
            if var>0:
                twod_list[i][j]=var
    all_poss=[]
    for i in range(len(list_TFIDF)):
        temporary=[]
        for j in range(len(words)):
            temporary.append(twod_list[j][i])
        all_poss.append(np.asarray(temporary))
    return all_poss
#  creating an array for the TFIDFsores of queries w.r.t document body   
def twd_title(words):
    twod_list = [] 
    for i in range (len(words)):
       new = []                  
       for j in range (len(list_TFIDF2)):
            new.append(0.0)
       twod_list.append(new)
                                                             
    for i in range (len(words)):                                              
        for j in range (len(list_TFIDF2)):
            for k in range (len(list_TFIDF2[j])):
                var=0.0
                if words[i]==list_TFIDF2[j][k]['key']:
                    if j == list_TFIDF2[j][k]['doc_id']:          
                        var=list_TFIDF2[j][k]['TFIDF_score']
                        break
            if var>0:
                twod_list[i][j]=var
    all_poss=[]
    for i in range(len(list_TFIDF2)):
        temporary=[]
        for j in range(len(words)):
            temporary.append(twod_list[j][i])
        all_poss.append(np.asarray(temporary))
    return all_poss


output=[]
for u in range(len(df_qry)):
    words=df_qry['qry'][u].split()
    all_poss1=twd(words)
    all_poss2=twd_title(words)
    a=final_qry12[u]
    list11=cos_all(all_poss1,all_poss2,a,u)
    output.append(list11)

# output the relevant documents in cran-output.txt file

f=open(argList[3],'w')   
for i in range(len(output)):
    t=len(output[i])
    for j in range(t):
        qryid=output[i][j]['qryid']
        docid=output[i][j]['docid']
        f.write(str(qryid)+" "+str(docid)+"\n")

f.close()

