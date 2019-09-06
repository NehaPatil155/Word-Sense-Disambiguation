
#Programming Assignment II - WSD
#Team Members: Neha Patil, Neha Nagarkar
#Team Name: Power Team
#Class: AIT 690
# Date: April 4, 2019
#This is scorer file used for finding accurracy and confusion matrix

import pandas as pd
import re
import sys
argList = sys.argv

#To run the scorer program from command line - provide the program name as scorer.py with arguments: pos-test-with-tags.txt pos-test-key.txt > pos-tagging-report.txt
#Providing argument two here i.e., line-test.txt 
#Processing the line-answers.txt 
list_of_lists=[]
with open(argList[2], "r") as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(',')]
        # in alternative, if you need to use the file content as numbers
        #inner_list = [int(elt.strip()) for elt in line.split(',')]
        list_of_lists.append(inner_list)
list_test=[]
for i in list_of_lists:
    for j in i:
       m=re.search(r'(senseid=")(.*)("\/>)', j)
       found = m.group(2)
       list_test.append(found)
instanceid=[]
for i in list_of_lists:
    for j in i:
       m=re.search(r'(instance=")(.*:)"\ssenseid.*', j)
       found = m.group(2)
       instanceid.append(found)
key=pd.DataFrame()
key['id']=instanceid
key['sense']=list_test
##################actually trained data############################
#Providing argument one here i.e., line-test.xml

list_of_lists1=[]
with open(argList[1], "r") as f1:
    for line in f1:
        inner_list = [elt.strip() for elt in line.split(',')]
        # in alternative, if you need to use the file content as numbers
        #inner_list = [int(elt.strip()) for elt in line.split(',')]
        list_of_lists1.append(inner_list)
len(list_of_lists1)
list_test1=[]
x=re.search(r'(senseid=")(.*)("\/>)','<answer instance="line-n.w7_122:11595:" senseid="phone"/>')
for i in list_of_lists1:
    for j in i:
       m1=re.search(r'(senseid=")(.*)("\/>)', j)
       found1 = m1.group(2)
       list_test1.append(found1)
instanceid1=[]

for i in list_of_lists1:
    for j in i:
       m2=re.search(r'(instance=")(.*:)"\ssenseid.*', j)
       found2 = m2.group(2)
       instanceid1.append(found2)
key1=pd.DataFrame()
key1['id']=instanceid1
key1['sense']=list_test1
count=0
for i in range(len(key1)):
    if key1['sense'][i]==key['sense'][i] and key1['id'][i]==key['id'][i] :
        count=count+1
accuracy=((count)/len(key))*100
print("Accuracy in percentage:-")
print(round(accuracy,2))   

from nltk.metrics import ConfusionMatrix
#confusion matrix shows number of misclassified tag, it is a two way array which gives 
#number of correctly classified tags is given in <> next to number incorrectly classified tags
ref  = list(key['sense'])
tagged = list(key1['sense'])
cm = ConfusionMatrix(ref, tagged) 
print("Confusion Matrix:-\n")      
print(cm)  
                
