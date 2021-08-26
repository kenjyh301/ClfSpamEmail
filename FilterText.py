import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction import _stop_words
import re
import logging


#clean data
class FilterText():
    def __init__(self,dfName):
        logging.info("Init text file to filter")
        self.__dfText= pd.read_csv(dfName)

    def ShowData(self):
        print(self.__dfText)

    def LowerCase(self,line):
        lineFilter= line.lower()
        return lineFilter

    def RemoveSpecialChar(self,line):
        lineFilter= line.translate(str.maketrans(dict.fromkeys(string.punctuation)))
        return lineFilter

    def RemoveStopWords(self,line):
        lineToken= line.split()
        lineFilter= [i for i in lineToken if i not in _stop_words.ENGLISH_STOP_WORDS]
        return ' '.join(word for word in lineFilter)

    def RemoveHyperLink(self,line):
        lineFilter= re.sub(r"http\S+","",line)
        return lineFilter

    def Filter(self):
        for index,row in self.__dfText.iterrows():
            # print(index)
            sentence= row['sentence']
            sentence= self.LowerCase(sentence)
            sentence= self.RemoveSpecialChar(sentence)
            sentence= self.RemoveStopWords(sentence)
            sentence= self.RemoveHyperLink(sentence)
            self.__dfText.at[index,'sentence']=sentence
    
    def Save(self,fileName):
        pd.DataFrame.to_csv(self.__dfText,fileName)

    def GetData(self):
        return self.__dfText










