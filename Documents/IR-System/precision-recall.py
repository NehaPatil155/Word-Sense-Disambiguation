# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:19:23 2019

@author: nehap
"""

import pandas as pd
import sys
import os

argList = sys.argv

cwd = os.getcwd()
contents = ""
df=pd.DataFrame()
list1=[]
# inputting the cran-output file
with open(argList[1],'r') as f:
    for line in f.readlines():
        contents += line
        
list1=contents.split("\n")
# inputting the cranqrel file
rel=open(argList[2],'r')
rel_qry=rel.read()
list2=[]
list2=rel_qry.split("\n")
count=0
list_qry=[]
list_doc=[]

for j in range(len(list2)):
        x=list2[j]
        y=x.split()
        list_qry.append(y[0])
        list_doc.append(y[1])
        
df1=pd.DataFrame()
df1['qry']=list_qry
df1['doc']=list_doc
df1['qry']=df1['qry'].astype(str).astype(int)
list_qry1=[]
list_doc1=[]
for j in range(len(list1)):
    if list1[j]!='':
        x=list1[j]
        y=x.split()
        var=int(y[0])
        list_qry1.append(var)
        list_doc1.append(y[1])    
df2=pd.DataFrame()
df2['qry']=list_qry1
df2['doc']=list_doc1
 
count=0  
count_list=[]   
count_id_list=[]
count_pd=pd.DataFrame()
for i in range(1,226): 
    count=0
    for j in range(len(df2)):
        if i == df2['qry'][j]:
            count=count+1
    count_id_list.append(i)
    count_list.append(count)
count_pd['ID']=count_id_list
count_pd['count']=count_list
relevant_list=[]   

# loop for getting relevant document list by comparing cranqrel file with cran-output file
for i in range(1,226):
    rel=0
    for j in range(len(df1)):
        if i==df1['qry'][j]:
            for k in range(len(df2)):
                if df1['qry'][j]==df2['qry'][k]:
                    if df1['doc'][j]==df2['doc'][k]:
                        rel=rel+1
                        break
    relevant_list.append(rel)
# loop for getting the count of documents retrieved by the system
precision_list=[]
for i in range(0,225):
    if count_pd['count'][i]!=0:
        precision=relevant_list[i]/count_pd['count'][i]
        precision_list.append(precision)
    else:
        precision_list.append(0.0)
final_precision=sum(precision_list)/len(precision_list)
collection_count=[]
count1=0   
# loop for getting the count of documents in the collection i.e., cranqrel 
for i in range(1,226): 
    count1=0
    for j in range(len(df1)):
        if i == df1['qry'][j]:
            count1=count1+1
    collection_count.append(count1)
recall_list=[]
for i in range(0,225):
    if collection_count[i]!=0:
        recall=relevant_list[i]/collection_count[i]
        recall_list.append(recall)
    else:
        recall_list.append(0.0)
final_recall=sum(recall_list)/len(recall_list)
# writing the precision and recall values in mylogfile.txt displaying average as well individual values for queries
f=open(argList[3],'w') 
f.write("Average Precision for the system:-"+str(final_precision)+"\n")
f.write("Average Recall for the system:- "+str(final_recall)+"\n")
f.write("\nPrecision for respective Queries\n")
id1=1
for i in precision_list:
    f.write("Query Id:-"+str(id1)+" Precision:-"+str(i)+"\n")
    id1=id1+1
f.write("\nRecall for respective Queries\n")
id2=1
for i in recall_list:
    f.write("Query Id:-"+str(id2)+" Recall:-"+str(i)+"\n")
    id2=id2+1