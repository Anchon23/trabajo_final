# 1. Lectura y exploración inicial del dataset:
# Importamos las librearias necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# leer el dataset con la libreria pandas
# creamos un try para que si no encuentra el fichero nos muestre un mensaje de error
try:
    datos = pd.read_csv('C:/Users/angel/OneDrive/Documentos/trabajo.csv', sep=',')
except FileNotFoundError:
    print("El fichero no existe")
except Exception as e:
    print("Se ha producido un error al leer el fichero: ", type(e).__name__)

# Mostrar las primeras y últimas filas del dataset.
print(datos.head())
print(datos.tail())

# Obtener información general del dataset, como el número de filas y columnas, y los tipos de datos.
num_filas, num_columnas = datos.shape
print(datos.info())
print("Número de filas:", num_filas)
print("Número de columnas:", num_columnas)

# 2. Limpieza y preprocesamiento de datos:
# qutar valores NaN usando una función personalizada que use la moda/media para remplazarlos.
# Si tenemos valores NaN, se reemplazarán por la moda/media de la columna.

def fillna_custom(col):
    if col.dtype == 'object':
        fill_value = col.mode()[0]
    else:
        fill_value = col.mean()
    return col.fillna(fill_value)
try:
    datos = datos.apply(fillna_custom, axis=0)
except Exception as e:
    print("Se ha producido un error al reemplazar los valores NaN: ", type(e).__name__)

# Renombrar las columnas para facilitar su manipulación. (Las traducimos de ingles a español)
datos.rename(columns={'Unnamed': 'id', 'Gender': 'genero', 'EthnicGroup': 'Grupo_etnico_estudiante', 'ParentEduc': 'educacion_padres', 
                      'LunchType': 'tipo_almuerzo', 'TestPrep': 'curso_preparacion', 'ParentMaritalStatus': 'estado_civil_padres', 
                      'PracticeSport': 'frequencia_deporte', 'IsFirstChild': 'esprimerniño', 'NrSiblings':'numeros_hermanos', 
                      'TransportMeans': 'medio_transporte','WklyStudyHours': 'horas.semanales_autoaprendizaje', 'MathScore': 'puntuacion_mates',
                      'ReadingScore': 'puntuacion_lectura', 'WritingScore': 'puntuacion_escrita'}, inplace=True)

# 3. Análisis exploratorio de datos
# Funciones personalizadas para calcular estadísticas
def calcular_media(columna):
    return np.mean(columna) 

def calcular_mediana(columna):
     return np.median(columna) 

def calcular_desv_est(columna):
    return np.std(columna)

# creamos un bucle para calcular las estadisticas de las columnas numericas
# si las variables no son numericas nos mostrara un mensaje de error
try:
    columnas_numericas = datos.select_dtypes(include=[np.number]).columns.tolist()

    for col in columnas_numericas:
        print(f"Estadísticas para la columna '{col}':")
        print(f"Media: {calcular_media(datos[col]):.2f}")
        print(f"Mediana: {calcular_mediana(datos[col]):.2f}")
        print(f"Desviación estándar: {calcular_desv_est(datos[col]):.2f}")
except Exception as e:
    print("Se ha producido un error al calcular las estadísticas, es posible las variable no sean numericas: ", type(e).__name__)

# grafico de barras de la puntuacion de matematicas, lectura y escritura por genero
pv1=pd.pivot_table(datos,index='genero',values=['puntuacion_mates','puntuacion_lectura','puntuacion_escrita'])
plt.style.use('ggplot')
p1=pv1.plot(kind='barh',y=['puntuacion_mates','puntuacion_lectura','puntuacion_escrita'],edgecolor='black',linewidth=2,figsize=(12,8),title='Hombres vs Mujeres')
p1.bar_label(p1.containers[0], label_type='edge',padding=0.5,bbox={"boxstyle": "round", "pad": 0.6, "facecolor": "white", "edgecolor": "#1c1c1c", "linewidth" : 2.5, "alpha": 1})
p1.bar_label(p1.containers[1], label_type='edge',padding=0.5,bbox={"boxstyle": "round", "pad": 0.6, "facecolor": "white", "edgecolor": "#1c1c1c", "linewidth" : 2.5, "alpha": 1})
p1.bar_label(p1.containers[2], label_type='edge',padding=0.5,bbox={"boxstyle": "round", "pad": 0.6, "facecolor": "white", "edgecolor": "#1c1c1c", "linewidth" : 2.5, "alpha": 1})
plt.show()

# boxplot de puntuacion total en funcion del nivel de estudios de los padres
# creamos un daataframe a partir del dataset original
df3 = datos.copy()

# caculamos la puntuacion total haciendo la media de las puntuaciones de matematicas, lectura y escritura
try:
    df3['puntuacion_total'] = (df3['puntuacion_mates'] + df3['puntuacion_lectura'] + df3['puntuacion_escrita']) / 3
except ValueError as e:
    print("Se ha producido un error al calcular la puntuación total: ", type(e).__name__)

# creamos una nueva columna con el nivel de estudios de los padres pasado a numerico
df3["formacion_padres"] = df3["educacion_padres"].map({"high school": 1, "some college": 2, "associate's degree": 3, "bachelor's degree": 4, "master's degree": 5})

# creamos el boxplot
plt.boxplot([
    df3[df3["educacion_padres"] == "high school"]["puntuacion_total"],
    df3[df3["educacion_padres"] == "some college"]["puntuacion_total"],
    df3[df3["educacion_padres"] == "associate's degree"]["puntuacion_total"],
    df3[df3["educacion_padres"] == "bachelor's degree"]["puntuacion_total"],
    df3[df3["educacion_padres"] == "master's degree"]["puntuacion_total"]],
    labels=["secundaria", "algunos estudios universitarios", "técnico superior", "licenciado", "posgrado"],
    boxprops=dict(color='black'),
    medianprops=dict(color='black'), vert=True, patch_artist=True)

# añadimos la media de cada grupo
try:
    for i in range(5):
        median = round(df3[df3["formacion_padres"] == i+1]["puntuacion_total"].median(), 2)
        plt.text(i+1, median+2, str(median), color='black', ha='center')
except Exception as e:
    print("Se ha producido un error al añadir las medias: ", type(e).__name__)

# añadimos el titulo y las etiquetas
plt.title("Boxplot de puntuacion total en funcion del nivel de estudios de los padres")
plt.xlabel("nivel de estudios de los padres")
plt.ylabel("puntuacion total")
plt.show()

# Gráfico circular de grupos étnicos del estudiantes
# calculamos el numero de estudiantes de cada grupo
try:
    group_a = datos.loc[datos['Grupo_etnico_estudiante']=='group A'].count()[0]
    group_b = datos.loc[datos['Grupo_etnico_estudiante']=='group B'].count()[0]
    group_c = datos.loc[datos['Grupo_etnico_estudiante']=='group C'].count()[0]
    group_d = datos.loc[datos['Grupo_etnico_estudiante']=='group D'].count()[0]
    group_e = datos.loc[datos['Grupo_etnico_estudiante']=='group E'].count()[0]
except Exception as e:
    print("Se ha producido un error al calcular el número de estudiantes de cada grupo: ", type(e).__name__)

# grafico circular
plt.pie([group_a, group_b, group_c, group_d, group_e], labels = ['group_A','group_B','group_C','group_D','group_E'],autopct='%.2f%%')
plt.title('Gráfico circular de grupos étnicos')
plt.show()

# matriz de correlacion utilizando las variables numericas
corr_df = datos.corr()
print("la correlacion de las variables cualitativas es:")
print(corr_df, "\n")
plt.figure(figsize=(8, 6))
sns.heatmap(corr_df, annot=True)
plt.show()

# 4. Manipulación de datos usando condicionales y bucles:
# filtrar dataset por numero de hermanos mayor a 6 y nivel de estudios de los padres bachelor's degree
print(datos.loc[(datos['numeros_hermanos'] > 6) & (datos['educacion_padres'] == "bachelor's degree"),["numeros_hermanos","educacion_padres"]])

# filtrar dataset por genero mujer y puntuacion total mayor a 77
print(datos.loc[(datos['genero'] == "female") & (datos['puntuacion_mates'] > 77),["genero","puntuacion_mates"]])

# mostrar por pantalla los datos del alumno con Unnamed: 0 igual a 13478
print(datos.loc[13478])

# filtrar dataset por grupo etnico igual a Group A y puntuacion en lectura ordenada de menor a mayor
print(datos.loc[datos['Grupo_etnico_estudiante'] == "group A",["Grupo_etnico_estudiante","puntuacion_lectura"]].sort_values(by=['puntuacion_lectura']))

# mostrar por pantalla el porcentaje de estudiantes que obtuvieron una puntuacion superior a 90 en cada asignatura
print("puntuacion superior a 90 en matematicas: ", round((datos['puntuacion_mates'] > 90).mean()*100,2), "%")
print("puntuacion superior a 90 en lectura: ", round((datos['puntuacion_lectura'] > 90).mean()*100,2), "%")
print("puntuacion superior a 90 en escritura: ", round((datos['puntuacion_escrita'] > 90).mean()*100,2), "%")

# nueva columna creada a patir de la suma de la puntuacion lectora y escrita
df4 = datos.copy()
df4['comprensión.lectora.y.expresión.escrita'] = (df4['puntuacion_lectura'] + df4['puntuacion_escrita']) / 2
print(df4.head())
