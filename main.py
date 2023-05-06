import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# leer el dataset con la libreria pandas
datos = pd.read_csv('C:/Users/angel/OneDrive/Documentos/trabajo.csv', sep=',')

# Mostrar las primeras y últimas filas del dataset.
print(datos.head())
print(datos.tail())

# Obtener información general del dataset, como el número de filas y columnas, y los tipos de datos.
num_filas, num_columnas = datos.shape
print(datos.info())
print("Número de filas:", num_filas)
print("Número de columnas:", num_columnas)

# qutar valores NaN usando una función personalizada que use la moda/media para remplazarlos. En caso de no haber filas con NaN, 
# todo dato que esté por encima o por debajo de dos desviaciones típicas de la media tendrá que imputarse por la moda/media.

def fillna_custom(col):
    if col.dtype == 'object':
        fill_value = col.mode()[0]
    else:
        fill_value = col.mean()
    return col.fillna(fill_value)

datos = datos.apply(fillna_custom, axis=0)

# Renombren las columnas para facilitar su manipulación.
datos.rename(columns={'Unnamed': 'id', 'Gender': 'genero', 'EthnicGroup': 'Grupo_etnico_estudiante', 'ParentEduc': 'educacion_padres', 
                      'LunchType': 'tipo_almuerzo', 'TestPrep': 'curso_preparacion', 'ParentMaritalStatus': 'estado_civil_padres', 
                      'PracticeSport': 'frequencia_deporte', 'IsFirstChild': 'esprimerniño', 'NrSiblings':'numeros_hermanos', 
                      'TransportMeans': 'medio_transporte','WklyStudyHours': 'horas.semanales_autoaprendizaje', 'MathScore': 'puntuacion_mates',
                      'ReadingScore': 'puntuacion_lectura', 'WritingScore': 'puntuacion_escrita'}, inplace=True)
print(datos.head())

# Funciones personalizadas para calcular estadísticas
def calcular_media(columna):
    return np.mean(columna) 

def calcular_mediana(columna):
     return np.median(columna) 

def calcular_desv_est(columna):
    return np.std(columna)

columnas_numericas = datos.select_dtypes(include=[np.number]).columns.tolist()

for col in columnas_numericas:
    print(f"Estadísticas para la columna '{col}':")
    print(f"Media: {calcular_media(datos[col]):.2f}")
    print(f"Mediana: {calcular_mediana(datos[col]):.2f}")
    print(f"Desviación estándar: {calcular_desv_est(datos[col]):.2f}")

# Crear gráficos para visualizar la distribución de los datos, como histogramas y diagramas de caja.
# pv1=pd.pivot_table(datos,index='genero',values=['puntuacion_mates','puntuacion_lectura','puntuacion_escrita'])
# plt.style.use('ggplot')
# p1=pv1.plot(kind='barh',y=['puntuacion_mates','puntuacion_lectura','puntuacion_escrita'],edgecolor='black',linewidth=2,figsize=(12,8),title='Hombres vs Mujeres')
# p1.bar_label(p1.containers[0], label_type='edge',padding=0.5,bbox={"boxstyle": "round", "pad": 0.6, "facecolor": "white", "edgecolor": "#1c1c1c", "linewidth" : 2.5, "alpha": 1})
# p1.bar_label(p1.containers[1], label_type='edge',padding=0.5,bbox={"boxstyle": "round", "pad": 0.6, "facecolor": "white", "edgecolor": "#1c1c1c", "linewidth" : 2.5, "alpha": 1})
# p1.bar_label(p1.containers[2], label_type='edge',padding=0.5,bbox={"boxstyle": "round", "pad": 0.6, "facecolor": "white", "edgecolor": "#1c1c1c", "linewidth" : 2.5, "alpha": 1})
# plt.show()

# boxplot de puntuacion total en funcion del nivel de estudios de los padres
df3 = datos.copy()
df3['puntuacion_total'] = (df3['puntuacion_mates'] + df3['puntuacion_lectura'] + df3['puntuacion_escrita']) / 3

df3["formacion_padres"] = df3["educacion_padres"].map({"high school": 1, "some college": 2, "associate's degree": 3, "bachelor's degree": 4, "master's degree": 5})

# plt.boxplot([
#     df3[df3["educacion_padres"] == "high school"]["puntuacion_total"],
#     df3[df3["educacion_padres"] == "some college"]["puntuacion_total"],
#     df3[df3["educacion_padres"] == "associate's degree"]["puntuacion_total"],
#     df3[df3["educacion_padres"] == "bachelor's degree"]["puntuacion_total"],
#     df3[df3["educacion_padres"] == "master's degree"]["puntuacion_total"]],
#     labels=["secundaria", "algunos estudios universitarios", "técnico superior", "licenciado", "posgrado"],
#     boxprops=dict(color='red'),
#     medianprops=dict(color='red'), vert=True, patch_artist=True)

# for i in range(5):
#     median = round(df3[df3["formacion_padres"] == i+1]["puntuacion_total"].median(), 2)
#     plt.text(i+1, median+2, str(median), color='black', ha='center')
    
# plt.title("Boxplot de puntuacion total en funcion del nivel de estudios de los padres")
# plt.xlabel("nivel de estudios de los padres")
# plt.ylabel("puntuacion total")
# plt.show()

# Gráfico circular de grupos étnicos
group_a = datos.loc[datos['Grupo_etnico_estudiante']=='group A'].count()[0]
group_b = datos.loc[datos['Grupo_etnico_estudiante']=='group B'].count()[0]
group_c = datos.loc[datos['Grupo_etnico_estudiante']=='group C'].count()[0]
group_d = datos.loc[datos['Grupo_etnico_estudiante']=='group D'].count()[0]
group_e = datos.loc[datos['Grupo_etnico_estudiante']=='group E'].count()[0]

# plt.pie([group_a, group_b, group_c, group_d, group_e], labels = ['group_A','group_B','group_C','group_D','group_E'],autopct='%.2f%%')
# plt.title('Gráfico circular de grupos étnicos')
# plt.show()

# grafico matriz de correlacion
matriz_correlacion = datos.corr()

# plt.matshow(matriz_correlacion, cmap='coolwarm')
# for i in range(matriz_correlacion.shape[0]):
#     for j in range(matriz_correlacion.shape[1]):
#         plt.annotate(f'{matriz_correlacion.iloc[i, j]:.2f}', xy=(j, i), horizontalalignment='center', verticalalignment='center')
# plt.xticks(range(len(matriz_correlacion.columns)), matriz_correlacion.columns, rotation=90)
# plt.yticks(range(len(matriz_correlacion.columns)), matriz_correlacion.columns)
# plt.show()


df2 = datos.copy()
df2 = df2.drop(columns='Unnamed: 0', axis=1)
df2['genero']=pd.factorize(df2.genero)[0]
df2['Grupo_etnico_estudiante']=pd.factorize(df2.Grupo_etnico_estudiante)[0]
df2['educacion_padres']=pd.factorize(df2.educacion_padres)[0]
df2['tipo_almuerzo']=pd.factorize(df2.tipo_almuerzo)[0]
df2['curso_preparacion']=pd.factorize(df2.curso_preparacion)[0]
df2['estado_civil_padres']=pd.factorize(df2.estado_civil_padres)[0]
df2['frequencia_deporte']=pd.factorize(df2.frequencia_deporte)[0]
df2['esprimerniño']=pd.factorize(df2.esprimerniño)[0]
df2['medio_transporte']=pd.factorize(df2.medio_transporte)[0]

corr_df = df2.corr()
# print("la correlacion de las variables cualitativas es:")
# print(corr_df, "\n")
# plt.figure(figsize=(8, 6))
# sns.heatmap(corr_df, annot=True)
# plt.show()

# filtrar dataset por numero de hermanos mayor a 6 y nivel de estudios de los padres bachelor's degree
print(datos.loc[(datos['numeros_hermanos'] > 6) & (datos['educacion_padres'] == "bachelor's degree"),["numeros_hermanos","educacion_padres"]])