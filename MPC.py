#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 21:47:19 2019

@author: erico
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("../base/Base_Completa.csv", sep=';', encoding='utf-8', low_memory=False)

df = df[df.TAXA_PARTICIPACAO_5EF!=0]

q1 = df['MEDIA_5EF_LP'].quantile(q=0.25)
q4 = df['MEDIA_5EF_LP'].quantile(q=0.75)

df1Q = df[df['MEDIA_5EF_LP']<q1]
df4Q = df[df['MEDIA_5EF_LP']>q4]

df1Q.loc[:,'ROTULO'] = 0
df4Q.loc[:,'ROTULO'] = 1

treino_base_LP = pd.concat([df4Q , df1Q])

atributos = ["TAXA_PARTICIPACAO_5EF", "TP_DEPENDENCIA", "IN_AGUA_INEXISTENTE", "IN_ENERGIA_INEXISTENTE", "IN_ESGOTO_INEXISTENTE","IN_LABORATORIO_INFORMATICA", "IN_LABORATORIO_CIENCIAS", "IN_QUADRA_ESPORTES", "IN_BIBLIOTECA_SALA_LEITURA", "IN_PATIO_COBERTO", "IN_SALA_DIRETORIA", "IN_SALA_PROFESSOR", "IN_AUDITORIO", "IN_DEPENDENCIAS_PNE", "IN_EQUIP_TV","IN_EQUIP_COPIADORA", "IN_EQUIP_RETROPROJETOR", "IN_EQUIP_IMPRESSORA", "IN_EQUIP_SOM", "IN_COMPUTADOR", "IN_INTERNET", "CO_UF", "CO_REGIAO"]


X = treino_base_LP.loc[:, atributos ].values
y = treino_base_LP.loc[:,'ROTULO'].values

# 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
#X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=1) # , test_size = 0.3, random_state = 1


# 
import keras
from keras.models import Sequential
from keras.layers import Dense


# 
def crear_model():
	# define model
	model = Sequential()

	model.add(Dense(units = len(atributos), kernel_initializer = 'uniform', activation = 'relu', input_dim = len(atributos)))
	model.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
	model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) # softmax sigmoid

	opt = keras.optimizers.Adam(lr=0.001)
	# Compiling the ANN
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

# 
modelo = crear_model()

# summarize the model
modelo.summary()

epocas = 1

# Treina o modelo
historico = modelo.fit(X_train, y_train, batch_size = 1, epochs= epocas, validation_data=(X_test, y_test))


scores = modelo.evaluate(X_test, y_test, batch_size=128)
print("%s: %.2f%%" % (modelo.metrics_names[1], scores[1]*100))


# evaluate test accuracy
score = modelo.evaluate(X_train, y_train, verbose=0)
accuracy = 100*score[1]
print('Test accuracy train: %.4f%%' % accuracy)


# Predicting the Test set results
from sklearn.metrics import classification_report, confusion_matrix
# evaluate the network
print("[INFO] evaluating network...")
predictions = modelo.predict(X_test)
print(classification_report(X_test.argmax(axis=1), predictions.argmax(axis=1), target_names = ['Abaixo da media','acima da media'])) 

# Salvar modelo em arquivo
modelo.save('Escola_Modelo_MPC_{}.h5'.format(accuracy))
