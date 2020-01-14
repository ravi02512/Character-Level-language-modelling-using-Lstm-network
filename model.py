from __future__ import print_function
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking,LSTM
import numpy as np
import random
import sys
import io



def build_model(x,y):

    input_data = Input(shape=[x.shape[1],x.shape[2]],name='Input_layer_1')

    hidden1 = LSTM(256,activation="relu",name='Lstm_layer_1',return_sequences=True)(input_data)
    
    drop_out=Dropout(0.4,name='dropout_layer_1')(hidden1)
    
    hidden2 = LSTM(128,activation="relu",name='Lstm_layer_2',return_sequences=False)(drop_out)
    
    drop_out2=Dropout(0.2,name='dropout_layer_2')(hidden2)
    
    hidden3 = Dense(y.shape[1], activation="relu",name='Dense_layer_1')(drop_out2)
    
    output=Activation('softmax',name='Activation_layer_1')(hidden3)
    
    model = Model(inputs=[input_data], outputs=[output])
    
    model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

    return model