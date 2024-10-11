'''
# Algoritmo de K-means / K - Medias
Algoritmo de agrupamiento no supervisado que agrupa los datos en K grupos distintos.
Creado para la materia "Mineria de Datos" de la Benemérita Universidad Autónoma de Puebla.
Docente: Pedro Tecanhuehue

'''
'''
## Librerias necesarias

'''
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

'''
## Funciones del algoritmo

'''

def calcular_distancias(datos,centroides):
    distancias_por_centroide = pd.DataFrame()
    i=1;
    #recorremos los centroides
    for _,centroide in centroides.iterrows():
        distancias = distancia_centroide(datos,centroide)#Obtenemos las distancias de cada centroide
        nombre_nueva_columna = f'Grupo{i}'
        #guardamos las distancias de cada centroide
        distancias_por_centroide[nombre_nueva_columna] = distancias
        i+=1
    return distancias_por_centroide
    
    
#Calculamos las distancias de un centroide a todos los objetos
def distancia_centroide(datos,centroide):
    #iteramos los indices
    distancias=[]
    suma_distancias = 0
    for _, datos_fila in datos.iterrows():
        #iteramos los atributos
        for columna in datos:
            dato = datos_fila[columna] #Obtenemos el dato a medir
            dato_centroide = centroide[columna]
            suma_distancias += HEOM(dato,dato_centroide,datos[columna],columna, datos)**2#Acumulamos la suma de distancias al cuadrado 
            #Acumulamos la suma de distancias al cuadrado 
        distancia = np.sqrt(suma_distancias) #se obtiene la raíz cuadrada de la suma de distancias
        suma_distancias = 0  #Limpiamos el valor de la suma 
        distancias.append(distancia)
    return distancias

def HEOM(dato, dato_centroide,datos_columna,columna, df):
    unique_values = df[columna].unique()
    # Si alguno de los datos de dato o dato_centroide es nulo, none, vacio o ? se asigna la distancia como 1
    if pd.isnull(dato) or pd.isnull(dato_centroide) or dato == '?' or dato == 'none' or dato == '':
        return 1
    # Verificar el tipo de datos de la columna
    if pd.api.types.is_numeric_dtype(datos_columna):
        return rn_diff(dato, dato_centroide, datos_columna)
    else:
        return overlap(dato, dato_centroide)

#Calcula la distancia de tipo númerico
def rn_diff(x, y, columna):
    return np.absolute(x - y) / rango(columna) 

#Calcula distancia de tipo Categórico o Nominal
def overlap(x,y):
    if (x == y):
        return 0
    return 1

#Calcula rango
def rango(columna):
    datos_atributo = np.array(columna) #obtiene todos los valores de un atributo a la vez
    # Filtramos solo datos con números
    datos_numericos_atributo = [cadena for cadena in datos_atributo if cadena is not None]
    # Calculamos máximo y mínimo
    minimo = np.amin(datos_numericos_atributo, axis=0) 
    maximo = np.amax(datos_numericos_atributo, axis=0)
    #calculamos rango 
    rango = maximo - minimo
    return rango

#Asigna a cada objeto al centroide más cercano y le asigna el grupo al que pertenece
def seleccionar_grupo_cercano(distancias):
    #Obtenemos la distancia minima en cada objeto
    minimos = distancias.min(axis=1) 
    grupos = []
    
    #iteramos las filas de las distancias
    for i_fila, datos_fila in distancias.iterrows():
        for columna in distancias:
            dato = datos_fila[columna] 
            #Si el dato de las distancias es igual a la distamcia minima, guardamos el grupo al que pertenece
            if (dato == minimos[i_fila]):
                grupos.append(columna)
                break
            
        
    return grupos

#Recalcula los nuevos centroides
def recalcular_centroides(datos,grupos):
    datos_con_grupo = datos.copy() #copiamos el dataframe original para no alterarlo
    datos_con_grupo['Grupo'] = grupos #agregamos la columna grupo a los datos
    datos_agrupados = datos_con_grupo.groupby('Grupo') # Separamos los grupos
    #Preparamos un dataframe a partir del original
    datos_original = pd.DataFrame(datos)
    # Obtenemos las columnas del DataFrame original
    columnas = datos_original.columns
    # Creamos un nuevo DataFrame con las mismas columnas pero sin datos
    nuevos_centroides = pd.DataFrame(columns=columnas)
    # Creamos un nuevo DataFrame para guardar cada atributo en una sola fila

    #iteramos los grupos
    for nombre_grupo, grupo in datos_agrupados:
        #iteramos las columnas de los grupos
        fila = len(nuevos_centroides)
        for columna in grupo:
            #Si la columna de los grupos es el numero de grupo lo ignoramos para no guardar ese valor
            if columna.startswith('Grupo'):
                continue
            # Verificar el tipo de datos de la columna
            if pd.api.types.is_numeric_dtype(columna):
                nuevos_centroides.loc[fila, columna] = promedio_atributo_numerico(grupo[columna])
            else:
                nuevos_centroides.loc[fila, columna] = promedio_atributo_categorico(grupo[columna]) 
    return nuevos_centroides

#Calcula el promedio de un atributo de tiopo numerico
def promedio_atributo_numerico(columna):
    datos_atributo = np.array(columna) #obtiene todos los valores de un atributo a la vez
    # Filtramos solo datos con números
    datos_numericos_atributo = [cadena for cadena in datos_atributo if cadena is not None]
    #devolvemos el promedio
    promedio = np.sum(datos_numericos_atributo)
    promedio = promedio / len(datos_atributo)
    promedio_con_dos_decimales = round(promedio, 2)
    return promedio_con_dos_decimales

#Calcula el promedio de un atributo de tipo categorico
def promedio_atributo_categorico(categorias):
    moda_categoria = {}
    for categoria in categorias:
        valor_categoria = categoria
        if valor_categoria in moda_categoria:
            moda_categoria[valor_categoria] += 1
        else:
            moda_categoria[valor_categoria] = 1
    moda = sorted(moda_categoria.items(), key=lambda x: x[1], reverse=True)
    return moda[0][0]

def agregar_grupo_a_datos(datos, grupos):
    datos_con_grupo = datos.copy() #copiamos el dataframe original para no alterarlo
    datos_con_grupo ['Grupo'] = grupos #agregamos la columna grupo a los datos
    #quitamos la palabra grupo de la columna de grupos
    
    datos_con_grupo ['Grupo'] = datos_con_grupo['Grupo'].apply(eliminar_palabra_grupo)
    return datos_con_grupo

#Elimina la palabra grupo para que solo se guarde el numero de grupo al que pertenece
def eliminar_palabra_grupo(valor):
    return valor.replace('Grupo', '')

#Verifica si se repite el centroide
def verificar_centroides(centroides, centroides_anteriores):
    if centroides.equals(centroides_anteriores):
        return True
    
    if isinstance(centroides_anteriores, pd.Series):
        centroides_anteriores = pd.DataFrame(centroides_anteriores).T

    if isinstance(centroides_anteriores, list):
        centroides_anteriores = pd.DataFrame(centroides_anteriores)

    for _,fila in centroides.iterrows():
        esta_la_fila = (centroides_anteriores.values == fila.values).all(axis=1).any()
        if esta_la_fila:
            continue
        else:
            return False
    
    return True

'''
# Algoritmo de K-means

'''
# Función que implementa el algoritmo k-MEANS
def k_means(datos, k):
    print("Iniciando algoritmo K-MEANS")
    #Seleccionamos los centroides iniciales
    print("Seleccionando centroides iniciales para k = ", k)
    centroides = datos.sample(n=k) # Inicializar los centroides de forma aleatoria
    print(centroides)
    print("\n")
    #Iteraciones para reasignar centroides
    while True:
        #Calculamos las distancias de cada objeto a cada centroide
        print("Calculando distancias de cada objeto a cada centroide")
        distancias = calcular_distancias(datos,centroides)
        print(distancias)
        #Asignamos cada objeto al centroide más cercano y obtenemos el grupo al que pertenece
        print("Asignando objetos a centroides (grupo cercano)")
        grupos = seleccionar_grupo_cercano(distancias)
        #print(grupos)
        centroides_anteriores = centroides.copy()
        print("Recalculando centroides")
        centroides = recalcular_centroides(datos,grupos)#Recalculamos los centroides
        print(centroides)
        #Establecemos las condiciones de paro
        if verificar_centroides(centroides, centroides_anteriores):
            print("Centroides iguales, terminando algoritmo")
            break
    
    return agregar_grupo_a_datos(datos,grupos)
    

'''
# Ejecución del algoritmo

'''

# Cargamos el conjunto de datos
breast_cancer = fetch_ucirepo(id=14) 
datos  = breast_cancer.data.original


k_grupos = [2,4,6]
archivos = []
for i in range(len(k_grupos)):
        nombre_archivo = f' K-MEANSGROUPS_{k_grupos[i]}.csv'#Nombramos el archivo donde se guardará el resultado
        archivos.append(nombre_archivo)#Guardamos el nombre del archivo para mostrarlo después
        #Ejecutamos el algoritmo k-MEANS
        print(f"Realizando agrupamiento sobre {k_grupos[i]} posibles grupos. Por favor espere un momento.")
        resultado = k_means(datos,k_grupos[i])
        #Guardamos los resultados en el archivo correspondiente
        resultado.to_csv(nombre_archivo, index=False)
        #Mostramos los nombres de los archivos
        nombres = ""
        for i in range(len(archivos)):
            nombres = nombres + " " + archivos[i]
        print("Ejecución finalizada.\n\n")

