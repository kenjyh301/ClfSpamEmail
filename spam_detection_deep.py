import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from FilterText import FilterText
import logging
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from keras.layers import Dense,LSTM, Embedding, Dropout, Activation, Bidirectional

logging.basicConfig(level=logging.ERROR)


# df= pd.DataFrame({'sentence':['aa bb cc','dd ee ff']})
# sen= df['sentence']
# tokenizer= Tokenizer(num_words=20)
# tokenizer.fit_on_texts(sen)

#Prepare data
try:
    dfText= pd.read_csv('df_filter.csv')
except Exception as e:
    logging.error(str(e))
    filterText= FilterText("df.csv")
    filterText.Filter()
    filterText.Save('df_filter.csv')
    dfText= filterText.GetData()
X= dfText['sentence'].astype(str)
y= dfText['type']
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=None)

#Tokenize data
tokenizer= Tokenizer(num_words=20)
tokenizer.fit_on_texts(X)
X_train_feature= np.array(tokenizer.texts_to_sequences(X_train))
X_test_feature= np.array(tokenizer.texts_to_sequences(X_test))

#Padding data
X_train_feature= pad_sequences(X_train_feature,maxlen=20)
X_test_feature= pad_sequences(X_test_feature,maxlen=20)

#Label target output
labelEncoder= LabelEncoder()
y_train= labelEncoder.fit(y_train.values)
y_test= labelEncoder.fit(y_test.values)


#size of the output vector from each layer
embedding_vector_length = 32
#Creating a sequential model
model = tf.keras.Sequential()
#Creating an embedding layer to vectorize
model.add(Embedding(20, embedding_vector_length, input_length=20))
#Addding Bi-directional LSTM
model.add(Bidirectional(LSTM(64)))
# model.add(Bidirectional(LSTM(64,return_sequences=True, dropout=0.5)))
#Relu allows converging quickly and allows backpropagation
model.add(Dense(16, activation='relu'))
#Deep Learninng models can be overfit easily, to avoid this, we add randomization using drop out
model.add(Dropout(0.1))
#Adding sigmoid activation function to normalize the output
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
