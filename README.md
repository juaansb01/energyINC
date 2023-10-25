#  [energyINC](https://github.com/juaansb01/energyINC/blob/main/Energy_inc.ipynb)
Primer proyecto de DataScience con Machine Learning con un esquema de los pasos a seguir y documentado a través de un Jupyter Notebook

## Instalación 
Descargar la carpeta Aily - DS Challange - Churn - Data, en la que están contenidas las bases de datos con las que trabajamos a lo largo de todo el proyecto.

## Librerías necesarias para el proyecto
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss, roc_curve, auc, precision_recall_curve
import scipy.sparse as sp
import seaborn as sns
```
```
pip install -r requirements.txt
```
## Etapas del proyecto
* Extraer los datos a partir de archivos .csv
* Cargar y juntar las bases de datos usando Python
* Limpiar, transformar y formatear los datos para el análisis
* Analizar los datos
* Realizar las transformaciones necesarias de los datos para entrenar el modelo
* Entrenar distintos modelos ajustando sus parámetros
* Compartir las conclusiones obtenidas a partir del modelo elegido

## Autores
* **Juan Sánchez Blázquez** [juaansb01](https://github.com/juaansb01)
