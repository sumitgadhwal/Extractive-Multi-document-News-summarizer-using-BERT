
import os
from summarizer.pre_processor import PreProcessor
from summarizer.centroid_embeddings import summarize
from summarizer.mmr_summarizer import MMRsummarize
from summarizer.sentenceOrdering import sentencePositioning
import nltk
from bs4 import BeautifulSoup


main_folder_path = os.getcwd() + "/Input/Input_DUC2002/input14"
Output_path=os.getcwd() +'/Output/Output_DUC2002/output14/system'


f=open(os.getcwd()+"/Output_NumberofSentences.txt")
summary_length=int(f.read())
print(summary_length)

files = os.listdir(main_folder_path)

body=''
#print(files)
f.close()


is_html=1  #1 means html else text

f=open(os.getcwd()+"/isHTML.txt")
is_html=int(f.read())
f.close()

##Reading the input from specified path
if is_html==1:
    for file1 in files:
        f=open(main_folder_path+"/"+file1)
        raw=f.read()
	
        lines = BeautifulSoup(raw).find_all('text')
        line=' '
        for itr in lines:
            line=line+itr.get_text()
	#print(line)
	
        body=body+line
        f.close()
	#print(line)
	#print('\n\n')
else:
    for file1 in files:
        f=open(main_folder_path+"/"+file1)
        line=f.read()
	
        
        body=body+line
        f.close()
	#print(line)
	#print('\n\n')



#applying k-means model, select around tripple of required sentences for summary to be filtered out further
print('\n\n1st stage: Preprocessing and K-means cluster algorithm in progress\n\n')
result = PreProcessor(body,summary_length)
full = ''.join(result)
print(full)
f = open(Output_path+'/task1_englishSyssum1.txt','w')
print (full,file=f)
f.close()





#applying centroid word embedding model, select  1.5 times of required sentences, will be removed out in next algorithm for removing redundancy
print('\n\n2nd stage: centroid word embedding algorithm in progress\n\n')
result2=summarize(full,summary_length)
print(result2)
f = open(Output_path+'/task1_englishSyssum2.txt','w')
print (result2,file=f)
f.close()




###applying MMR model to remove redundancy
print('\n\n3rd stage: MMR model algorithm in progress\n\n')
result3=MMRsummarize(result2,summary_length)
print(result3)
f = open(Output_path+'/task1_englishSyssum3.txt','w')
print (result3,file=f)
f.close()


#applying Sentenceordering model
print('\n\n 4th stage: Sentence Ordering algrorithm in progress\n\n')
result4=sentencePositioning(result3)
f = open(Output_path+'/task1_englishSyssum4.txt','w')
print (result4)
print (result4,file=f)
f.close()


