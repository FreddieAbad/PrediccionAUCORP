# PrediccionAUCORP
Prediccion de Valores en Series de Tiempo (Forecasting Time Series) usando MLP, LSTM - RNN

**IMPORTANTE**
Documento Final (Resumen):
https://github.com/FreddieAbad/PrediccionAUCORP/blob/master/Entrega/Prediccion%20de%20Ventas%20en%20una%20Serie%20de%20Tiempo%20-%20P.%20Aucay.ipynb

**CONFIGURACION**
Para correr el Jupyter Notebook se puede realizar por 2 configuraciones:
### Configuracion 1 
Entorno Conda para importar: 
(Para importar el entorno es necesario tener instalado con Anaconda3, la version de python y demas estan determinados en el siguiente entorno )
https://github.com/FreddieAbad/PrediccionAUCORP/blob/master/Entrega/Ambiente%20Conda/condaEnvironmentPrediccionAUCORP.yml

### Configuracion 2
#### Configuracion Directa
##### Dependencias:
- **Python 3.6** (Es fundamental contar con esta version ya que Py3.7 no corren algunas funciones, Py 2.4 tiene funciones deprecated)
- Keras
- Tensorflow
- Scikit Learn
- Scipy
- Pandas
- Numpy
- Jupyter Notebook for Anaconda.

##### Comando para instalar Tensorflow:
Si el computador tiene GPU
pip install --ignore-installed --upgrade tensorflow-gpu 
Si el computador NO tiene GPU
pip install --ignore-installed --upgrade tensorflow

##### Comando para instalar KERAS:
conda install mkl-service m2w64-toolchain pip install pydot keras

