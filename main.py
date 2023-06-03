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

try:
    def fillna_custom(col):
        if col.dtype == 'object':
            fill_value = col.mode()[0]
        else:
            fill_value = col.mean()
        return col.fillna(fill_value)
    datos = datos.apply(fillna_custom, axis=0)
except Exception as e:
    print("Se ha producido un error al reemplazar los valores NaN: ", type(e).__name__)

# Renombrar las columnas para facilitar su manipulación. (Las traducimos de ingles a español)
datos.rename(columns={'Unnamed: 0': 'id', 'Gender': 'genero', 'EthnicGroup': 'Grupo_etnico_estudiante', 'ParentEduc': 'educacion_padres', 
                      'LunchType': 'tipo_almuerzo', 'TestPrep': 'curso_preparacion', 'ParentMaritalStatus': 'estado_civil_padres', 
                      'PracticeSport': 'frequencia_deporte', 'IsFirstChild': 'esprimerniño', 'NrSiblings':'numeros_hermanos', 
                      'TransportMeans': 'medio_transporte','WklyStudyHours': 'horas_semanales_autoaprendizaje', 'MathScore': 'puntuacion_mates',
                      'ReadingScore': 'puntuacion_lectura', 'WritingScore': 'puntuacion_escrita'}, inplace=True)

# Seleccionar solo columnas numéricas
columnas_numericas = datos.select_dtypes(include=[int, float]).columns

# Calcular mediana y media de las columnas numéricas
medianas = datos[columnas_numericas].median()
medias = datos[columnas_numericas].mean()

# Crear DataFrame con mediana y media
estadisticas_df = pd.DataFrame({'Columna': columnas_numericas, 'Mediana': medianas, 'Media': medias})

# Imprimir el DataFrame en forma de tabla
print(estadisticas_df.to_string(index=False))

# 3. Análisis exploratorio de datos
# creamos un bucle para calcular las estadisticas de las columnas numericas
# si las variables no son numericas nos mostrara un mensaje de error
try:
    columnas_numericas = datos.select_dtypes(include=[np.number]).columns.tolist()

    for col in columnas_numericas:
        print(f"Estadísticas para la columna '{col}':")
        print(f"Media: {np.mean(datos[col]):.2f}")
        print(f"Mediana: {np.median(datos[col]):.2f}")
        print(f"Desviación estándar: {np.std(datos[col]):.2f}")
except Exception as e:
    print("Se ha producido un error al calcular las estadísticas, es posible las variable no sean numericas: ", type(e).__name__)

# 4. Visualización de datos
# Cuatro gráficos circulares que muestran el porcentaje de personas según diferentes variables.
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

variables = ['frequencia_deporte', 'estado_civil_padres', 'educacion_padres', 'Grupo_etnico_estudiante']
etiquetas = ['Frecuencia de deporte', 'Estado civil de los padres', 'Educación de los padres', 'Grupo étnico del estudiante']

for i, ax in enumerate(axes.flat):
    variable = variables[i]
    etiqueta = etiquetas[i]
    data = datos[variable].value_counts()
    porcentajes = data / data.sum() * 100
    
    ax.pie(porcentajes, labels=data.index, autopct='%1.1f%%', startangle=90)

    ax.set_title(etiqueta)

plt.suptitle('Porcentaje de personas según diferentes variables', fontsize=14)
plt.tight_layout()
plt.show()

# grafico de barras de la puntuacion de matematicas, lectura y escritura por genero
pv1 = pd.pivot_table(datos, index='genero', values=['puntuacion_mates', 'puntuacion_lectura', 'puntuacion_escrita'])
plt.style.use('ggplot')
p1 = pv1.plot(kind='barh', y=['puntuacion_mates', 'puntuacion_lectura', 'puntuacion_escrita'], edgecolor='black', linewidth=2, figsize=(12, 8), title='Hombres vs Mujeres')

containers = p1.containers

for container in containers:
    p1.bar_label(container, label_type='edge', padding=0.5, bbox={"boxstyle": "round", "pad": 0.6, "facecolor": "white", "edgecolor": "#1c1c1c", "linewidth": 2.5, "alpha": 1})

plt.show()

# boxplot de puntuacion total en funcion del nivel de estudios de los padres
df3 = datos.copy()

# caculamos la puntuacion total haciendo la media de las puntuaciones de matematicas, lectura y escritura
try:
    df3['puntuacion_total'] = (df3['puntuacion_mates'] + df3['puntuacion_lectura'] + df3['puntuacion_escrita']) / 3
except ValueError as e:
    print("Se ha producido un error al calcular la puntuación total: ", type(e).__name__)

# creamos una nueva columna con el nivel de estudios de los padres pasado a numerico
df3["formacion_padres"] = df3["educacion_padres"].map({"high school": 1, "some college": 2, "associate's degree": 3, "bachelor's degree": 4, "master's degree": 5})
data = []
labels = ["secundaria", "algunos estudios universitarios", "técnico superior", "licenciado", "posgrado"]
for i in range(5):
    data.append(df3[df3["formacion_padres"] == i+1]["puntuacion_total"])
plt.boxplot(data, labels=labels, boxprops=dict(color='black'), medianprops=dict(color='black'), vert=True, patch_artist=True)

# añadimos la media de cada grupo
try:
    for i, d in enumerate(data):
        median = round(d.median(), 2)
        plt.text(i+1, median+2, str(median), color='black', ha='center')
except Exception as e:
    print("Se ha producido un error al añadir las medias: ", type(e).__name__)

plt.title("Boxplot de puntuacion total en funcion del nivel de estudios de los padres")
plt.xlabel("nivel de estudios de los padres")
plt.ylabel("puntuacion total")
plt.show()


# Gráfico de barras de la puntuación media de cada asignatura en función de la frecuencia de deporte
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

puntuaciones = ['puntuacion_lectura', 'puntuacion_mates', 'puntuacion_escrita']
frecuencias = ['regularly', 'sometimes', 'never']
colores = ['red', 'green', 'blue']

for i, ax in enumerate(axes):
    data = datos[datos['frequencia_deporte'] == frecuencias[i]]
    promedios = data[puntuaciones].mean()
    promedios.plot(kind='bar', ax=ax, color=colores[i])

    ax.set_xlabel('Frecuencia de deporte: ' + frecuencias[i])
    ax.set_ylabel('Puntuación promedio')
    ax.set_xticklabels(puntuaciones, rotation=0)
    
fig.suptitle('Puntuaciones en función de la frecuencia de deporte', fontsize=14)

plt.tight_layout()

plt.show()

# matriz de correlacion utilizando las variables numericas
# eliminar una columna de las variables cuantitativas
datos.drop(['id'], axis=1, inplace=True)
corr_df = datos.corr()
print("la correlacion de las variables cuantitativas es:")
print(corr_df, "\n")
plt.figure(figsize=(8, 6))
sns.heatmap(corr_df, annot=True)
plt.show()


# Graph II, III, IV: Correlation with MathScore, ReadingScore, WritingScore
for score_column in ['puntuacion_mates', 'puntuacion_lectura', 'puntuacion_escrita']:
    # Create a heatmap of the correlations with the target column
    corr = data.corr()
    target_corr = corr[score_column].drop(score_column)

    # Sort correlation values in descending order
    target_corr_sorted = target_corr.sort_values(ascending=False)

    plt.figure(figsize=(5, 10))
    sns.heatmap(target_corr_sorted.to_frame(), cmap="coolwarm", annot=True, fmt='.2f')
    plt.title(f'Correlation with {score_column}')
    plt.show()

# 4. Manipulación de datos usando condicionales y bucles:
# filtrar dataset por numero de hermanos mayor a 6 y nivel de estudios de los padres bachelor's degree
print(datos.loc[(datos['numeros_hermanos'] > 6) & (datos['educacion_padres'] == "bachelor's degree"),["numeros_hermanos","educacion_padres"]])

# filtrar dataset por genero mujer y puntuacion total mayor a 77
print(datos.loc[(datos['genero'] == "female") & (datos['puntuacion_mates'] > 77),["genero","puntuacion_mates"]])

# mostrar por pantalla los datos del alumno con un id igual a 13478
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
