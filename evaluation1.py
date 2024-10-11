#Importamos los datos ya agrupados
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np

datos = pd.read_csv('./ K-MEANSGROUPS_2.csv', engine='python')
#Dejamos solo las columnas "Class" y Grupo
datos = datos[['Class', 'Grupo']]
#Ahora analizamos qué calse corresponde a cada grupo (por mayoría)
print(datos.groupby('Grupo')['Class'].value_counts())

#Como podemos observar nos dice los grupos seleccionados predice que no hay recurrencias en ambos grupos.
#Ahora hacemos la matriz de confusión

#Creamos un nuevo df reemplazando los valores de la columna grupos por los valores de la columna Class que correspondan
datos['Grupo'] = datos['Grupo'].replace({1: 'no-recurrence-events', 2: 'no-recurrence-events'})
print("Datos: \n")
print(datos)

print("y_true: \n")
y_true = datos['Class']
print(y_true)

print("y_pred: \n")
y_pred = datos['Grupo']
print(y_pred)

#Creamos la matriz de confusión
confusion_matrix = confusion_matrix(y_true, y_pred,labels=["no-recurrence-events", "recurrence-events"])
print(confusion_matrix)

#Calculamos la precisión
accuracy = accuracy_score(y_true, y_pred)

#Imprimimos la precisión
print("Reporte de clasificación: \n", classification_report(y_true, y_pred,zero_division=1))