#Programming Assignment II - POS Tagging
#Team Members: Neha Patil, Neha Nagarkar
#Team Name: Power Team
#Class: AIT 690
# Date: March 19, 2019
#This is scorer file used for finding accurracy and confusion matrix

import pandas as pd
import re
import sys
argList = sys.argv
#To run the scorer program from command line - provide the program name as scorer.py with arguments: pos-test-with-tags.txt pos-test-key.txt > pos-tagging-report.txt
#Providing argument two here i.e., pos-text-key.txt 
#Processing the pos-text-key.txt 
key_file=open(argList[2],'r')
data=key_file.read()
data1=re.sub(r'\n',' ',data)
words=data1.split(' ')
key_list = []
for i in words:
    if i==' ' or i=='[' or i==']' or i=='':
        continue
    else:
        key_list.append(i)       
finalkey_list1=[]
for i in key_list:
    if re.match(r"(.*)\/(.*)/(.*)",i):
        s=re.sub(r"(.*\/.*)/(.*)",r"\1 \2",i)
        x=s.split(" ")
    else:
        x=i.split('/')
    finalkey_list1.append(x)
finalkey_list=finalkey_list1  
for i in range(len(finalkey_list1)):
    if re.match(r"(.*)|(.*)",finalkey_list1[i][1]):
        finalkey_list[i][1]=re.sub(r"(.*)\|(.*)",r"\1",finalkey_list1[i][1])
    else:
        finalkey_list[i][1]=finalkey_list1[i][1]
    
key_frame=pd.DataFrame(finalkey_list)
len(key_frame)
##################actually trained data############################
#Providing argument one here i.e., pos-test-with-tags.txt
#Processing the pos-test-with-tags.txt 
#Processing the pos-test-with-tags.txt
file=open(argList[1],'r')
fdata=file.read()
fdata1=re.sub(r'\n',' ',fdata)
text=fdata1.split(' ')


test_list = []

for i in text:
    if i==' ' or i=='[' or i==']' or i=='':
        continue
    else:
        test_list.append(i)
        
finaltest_list1=[]
for i in test_list:
    if re.match(r"(.*)\/(.*)/(.*)",i):
        s=re.sub(r"(.*\/.*)/(.*)",r"\1 \2",i)
        x=s.split(" ")
    else:
        x=i.split('/')
    finaltest_list1.append(x)
finaltest_list=finaltest_list1 
len(finaltest_list) 
for i in range(len(finaltest_list1)):
    if re.match(r"(.*)|(.*)",finaltest_list1[i][1]):
        finaltest_list[i][1]=re.sub(r"(.*)\|(.*)",r"\1",finaltest_list1[i][1])
    else:
        finaltest_list[i][1]=finaltest_list[i][1]
    
test_frame=pd.DataFrame(finaltest_list1)
len(test_frame)

new_df=pd.DataFrame(key_frame)
new_df['model_tags']=test_frame[1]
count=0
for i in range(len(new_df)):
    if new_df[1][i]!=new_df['model_tags'][i]:
        count=count+1

accuracy=((len(new_df)-count)/len(new_df))*100
print("Accuracy in percentage:-")
print(round(accuracy,2))

from nltk.metrics import ConfusionMatrix
#confusion matrix shows number of misclassified tag, it is a two way array which gives 
#number of correctly classified tags is given in <> next to number incorrectly classified tags
ref  = list(new_df[1])
tagged = list(new_df['model_tags'])
cm = ConfusionMatrix(ref, tagged) 
print("Confusion Matrix:-\n")      
print(cm)  
                
