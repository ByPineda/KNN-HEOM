#Importamos los datos ya agrupados
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np

datos = pd.read_csv('./dataset/breast-cancer-orange.csv', engine='python')
#Dejamos solo las columnas "recurrence" y Cluster
datos = datos[['recurrence', 'Cluster']]
#Ahora analizamos qué calse corresponde a cada Cluster (por mayoría)
print(datos.groupby('Cluster')['recurrence'].value_counts())

#Como podemos observar nos dice los Clusters seleccionados predice que no hay recurrencias en ambos Clusters.
#Ahora hacemos la matriz de confusión

#Creamos un nuevo df reemplazando los valores de la columna Clusters por los valores de la columna recurrence que correspondan
datos['Cluster'] = datos['Cluster'].replace({"C1": 'no-recurrence-events', "C2": 'no-recurrence-events'})
print("Datos: \n")
print(datos)

print("y_true: \n")
y_true = datos['recurrence']
print(y_true)

print("y_pred: \n")
y_pred = datos['Cluster']
print(y_pred)

#Creamos la matriz de confusión
confusion_matrix = confusion_matrix(y_true, y_pred,labels=["no-recurrence-events", "recurrence-events"])
print(confusion_matrix)

#Calculamos la precisión
accuracy = accuracy_score(y_true, y_pred)

#Imprimimos la precisión
print("Reporte de clasificación: \n", classification_report(y_true, y_pred,zero_division=1))