# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import DataFrame, concat, read_csv
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.layers import Dense, LSTM
from keras.models import  Sequential
from keras import backend as K
from keras import optimizers
# Convertir Serie de Tiempo en Dataset Supervisado
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # Secuencia Input (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('Var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # Secuencia Forecast-Prediccion (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('Var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('Var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # Secuencias Juntas
    agg = concat(cols, axis=1)
    agg.columns = names
    # Eliminar outlayers NaN
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def graficaUnitaria(fix:16,fiy:8,historyXYZ,strEvaluacion,mostrar,nombreArchivo):
    figure(figsize=(fix,fiy))
    pyplot.grid(True)
    if(strEvaluacion=='accuracy'):
        pyplot.title("ACCURACY vs VAL_ACCURACY")
        pyplot.plot(historyXYZ.history['acc'], label='Train ACC')
        pyplot.legend()
        pyplot.plot(historyXYZ.history['val_acc'], label='Test VAL_aCC')
        pyplot.legend()
        pyplot.savefig(nombreArchivo, dpi=300)
        if(mostrar):
            pyplot.show()
    else:
        pyplot.title("LOSS vs VAL_LOSS")
        pyplot.plot(historyXYZ.history['loss'], label='Train LOSS')
        pyplot.plot(historyXYZ.history['val_loss'], label='Test VAL_LOSS')
        pyplot.legend()
        pyplot.savefig(nombreArchivo, dpi=300)
        if(mostrar):
            pyplot.show()
    





dataset = read_csv('aucorpTrain.csv', header=0, index_col=0)
dataset=dataset.iloc[:,2:]
values = dataset.values
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# Cast a un solo tipo de dato
values = values.astype('float32')
# Normalizacion
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# Datos a Modelo Aprendizaje Supervisado
reframed = series_to_supervised(scaled, 1, 1)
#Datos para entrenar
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)

# Train and Test Data
values = reframed.values
train = values[:68, :]
test = values[69:, :]
# Separacion Datos Input-Output
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# Reshape input para tener 3D [muestra, timesteps, caracteristicas]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# RED NEURONAL
modelC1 = Sequential()
modelC1.add(LSTM(12, dropout=0.0002, input_shape=(train_X.shape[1], train_X.shape[2])))
modelC1.add(Dense(1, activation='sigmoid'))
sgd = optimizers.SGD(lr=0.0002, decay=1e-7, momentum=0.9, nesterov=True)
modelC1.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
historyC1 = modelC1.fit(train_X, train_y, epochs=1000, batch_size=40, validation_data=(test_X, test_y),validation_split=0.2, verbose=0, shuffle=False)
modelC1.summary()


ypredictC1 = modelC1.predict(test_X)
print(ypredictC1)

test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
print(test_X)
# Inversion de Escala MinMax
inv_yhat = concatenate((ypredictC1, test_X[:, 1:]), axis=1)
print(inv_yhat)

inv_yhat = scaler.inverse_transform(inv_yhat)
