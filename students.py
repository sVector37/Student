# -*- coding: utf-8 -*-
# Импорт необходимых библиотек
import numpy as np
import pandas as pd
import scipy
#import h2o
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Инициализация H2O
#h2o.init()
#h2o.cluster_info()
#h2o.ls()

xlsx = pd.ExcelFile('C:\\data\\db\\Feedback.xlsx')
df_fb = pd.read_excel(xlsx, 'forma')
# Очистка данных
df_fb = df_fb.dropna()
# Проверка и просмотр строк и названий столбцов после импорта
df_fb.head()
df_fb.tail()
df_fb.dtypes()
df_fb.info()
df_fb.isnull().sum()
df_fb.columns
df_fb.describe()
df_fb.type.value_counts()
sns.countplot(data=df_fb,x="type", hue='type',palette="Paired")
plt.show()
df_fb.recipient.value_counts()
sns.countplot(data=df_fb,x="recipient", hue='recipient',palette="Paired")
plt.show()
sns.histplot(data=df_fb, y="recipient", hue='recipient',palette="Paired")
plt.show()

#from pymongo import MongoClient
# Подключение к СУБД
client = MongoClient()
db = client.DWH_student
# Формирование коллекций БД и заполнение их соответствующими (отфильтрованными по одному из параметров) фрагментами датафреймов
df_recipient = df['recipient']
df_text = df['text']
df_type = df['type']

xlsx = pd.ExcelFile('C:\\data\\db\\2session2024m.xlsx')
df = pd.read_excel(xlsx, 'DoT')
# Очистка данных
df = df.dropna()
# Проверка и просмотр строк и названий столбцов после импорта
df.head()
#df.tail()
#df.dtypes()
df.info()
df.isnull().sum()
df.columns
df.describe()
df.direction.value_counts()
plt.figure(figsize=(20, 12)) 
sns.countplot(data=df,y="direction", hue='direction',palette="Paired")
plt.show()

sns.barplot(data=df,x="indicator1",y="direction", hue='direction',palette="Paired")
plt.show()
sns.barplot(data=df,x="indicator2",y="direction", hue='direction',palette="Paired")
plt.show()
sns.barplot(data=df,x="indicator3",y="direction", hue='direction',palette="Paired")
plt.show()
sns.barplot(data=df,x="indicator4",y="direction", hue='direction',palette="Paired")
plt.show()

xlsx = pd.ExcelFile('C:\\data\\db\\Subj.xlsx')
df = pd.read_excel(xlsx, 'mcourses')
# Очистка данных
df = df.dropna()
# Проверка и просмотр строк и названий столбцов после импорта
df.head()
#df.tail()
#df.dtypes()
df.info()
df.isnull().sum()
df.columns
df.describe()
df.ball.value_counts()
plt.figure(figsize=(20, 12)) 
sns.countplot(data=df,x="ball", hue='groupnumber',palette="Paired")
plt.show()

xlsx = pd.ExcelFile('C:\\data\\db\\Subjects.xlsx')
df = pd.read_excel(xlsx, 'courses')


# Воспользуемся возможностями seaborn для анализа
bike = pd.read_excel(xlsx, 'Transactions')
sns.pairplot(bike);
sns.pairplot(data=bike, aspect=.85, hue='brand', size=12);

sns.set(font_scale=1.15)
plt.figure(figsize=(12,6))
sns.heatmap(bike.corr(), cmap='RdBu_r', annot=True, vmin=-1, vmax=1);

# Теперь воспользуемся возможностями matplotlib.pyplot

# Пример: отображение распределения велосипедов по производителям
make_counts = df['brand'].value_counts()
print(make_counts)
# Визуализация распределения велосипедов по производителям
plt.figure(figsize=(12, 6))
plt.title('Распределение велосипедов по производителям', size=12)
plt.xticks(rotation=90)
plt.xlabel('Производитель', size=8)
plt.ylabel('Количество велосипедов', size=12)
plots = sns.barplot(x=make_counts.values, y=make_counts.index, orient='h')
plt.show()

xlsx = pd.ExcelFile('C:\\data\\db\\lms.xlsx')
df = pd.read_excel(xlsx, 'bLMS')
df.head()
#df=df.rename(columns={"race/ethnicity":"race_ethnicity"})
df.tail()
df.describe()
df.dtypes
df.info()
df.isnull().sum()
# Очистка данных df = df.dropna()
df.columns

plt.figure(figsize=(20, 12)) 
sns.catplot(x="ball",y="direction",data=df, palette="ocean",kind="bar")
plt.xticks(rotation=90)
plt.show()

# Строим графики встроенными возможностями Pandas Basic
df.plot.bar('conspect', 'ball', color="red")
df.plot.scatter('presentation','ball', color="red")
df.plot.scatter('tests','ball', color="green")
df.plot.scatter('forum_chat','ball')

# Строим функцию линейной регрессии
from scipy import polyval, stats
fit_output = stats.linregress(df[['conspect','ball']])
slope, intercept, r_value, p_value, slope_std_error = fit_output
print(slope, intercept, r_value, p_value, slope_std_error)
# Рисуем график линейной регрессии
import matplotlib.pyplot as plt
plt.plot(df[['conspect']], df[['ball']],'o', label='Data')
plt.plot(df[['conspect']], intercept + slope*df[['ball']], 'r', linewidth=3, label='Linear regression line')
plt.ylabel('Средний балл промежуточной аттестации')
plt.xlabel('Полный конспект у курсов в LMS, %')
plt.legend()
plt.show()

fit_output = stats.linregress(df[['presentation','ball']])
slope, intercept, r_value, p_value, slope_std_error = fit_output
print(slope, intercept, r_value, p_value, slope_std_error)
# Рисуем график линейной регрессии
import matplotlib.pyplot as plt
plt.plot(df[['presentation']], df[['ball']],'o', label='Data')
plt.plot(df[['presentation']], intercept + slope*df[['ball']], 'r', linewidth=3, label='Linear regression line')
plt.ylabel('Средний балл промежуточной аттестации')
plt.xlabel('Полный комплект презентаций курса в LMS, %')
plt.legend()
plt.show()

fit_output = stats.linregress(df[['tests','ball']])
slope, intercept, r_value, p_value, slope_std_error = fit_output
print(slope, intercept, r_value, p_value, slope_std_error)
# Рисуем график линейной регрессии
import matplotlib.pyplot as plt
plt.plot(df[['tests']], df[['ball']],'o', label='Data')
plt.plot(df[['tests']], intercept + slope*df[['ball']], 'r', linewidth=3, label='Linear regression line')
plt.ylabel('Средний балл промежуточной аттестации')
plt.xlabel('Комплект тестов для самоподготовки в LMS, %')
plt.legend()
plt.show()

fit_output = stats.linregress(df[['forum_chat','ball']])
slope, intercept, r_value, p_value, slope_std_error = fit_output
print(slope, intercept, r_value, p_value, slope_std_error)
# Рисуем график линейной регрессии
import matplotlib.pyplot as plt
plt.plot(df[['forum_chat']], df[['ball']],'o', label='Data')
plt.plot(df[['forum_chat']], intercept + slope*df[['ball']], 'r', linewidth=3, label='Linear regression line')
plt.ylabel('Средний балл промежуточной аттестации')
plt.xlabel('Тесное общение в форуме и чате в LMS, %')
plt.legend()
plt.show()

# Воспользуемся возможностями seaborn для анализа

# Подготовка данных для кластеризации
X = df[['conspect', 'ball']]
# Преобразование категориальных данных в числовые
X = pd.get_dummies(X, columns=['conspect'], drop_first=True)
# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Обучение модели кластеризации
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)
# Оценка модели
labels = kmeans.labels_
silhouette_avg = silhouette_score(X_scaled, labels)
print(f'Silhouette Score: {silhouette_avg}')
# Визуализация результатов кластеризации
df['Cluster'] = labels
plt.figure(figsize=(12, 6))
sns.scatterplot(x='conspect', y='ball', hue='Cluster', data=df, palette='viridis')
plt.title('Кластеризация курсов в LMS по наличию полного конспекта')
plt.xlabel('Полный конспект у курсов в LMS, %')
plt.ylabel('Баллы аттестации')
plt.xticks(rotation=90)
plt.show()

# Подготовка данных для кластеризации
X = df[['presentation', 'ball']]
# Преобразование категориальных данных в числовые
X = pd.get_dummies(X, columns=['presentation'], drop_first=True)
# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Обучение модели кластеризации
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)
# Оценка модели
labels = kmeans.labels_
silhouette_avg = silhouette_score(X_scaled, labels)
print(f'Silhouette Score: {silhouette_avg}')
# Визуализация результатов кластеризации
df['Cluster'] = labels
plt.figure(figsize=(12, 6))
sns.scatterplot(x='presentation', y='ball', hue='Cluster', data=df, palette='viridis')
plt.title('Кластеризация курсов в LMS по наличию презентаций всего курса')
plt.xlabel('Полный набор презентаций у курсов в LMS, %')
plt.ylabel('Баллы аттестации')
plt.xticks(rotation=90)
plt.show()

# Подготовка данных для кластеризации
X = df[['tests', 'ball']]
# Преобразование категориальных данных в числовые
X = pd.get_dummies(X, columns=['tests'], drop_first=True)
# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Обучение модели кластеризации
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)
# Оценка модели
labels = kmeans.labels_
silhouette_avg = silhouette_score(X_scaled, labels)
print(f'Silhouette Score: {silhouette_avg}')
# Визуализация результатов кластеризации
df['Cluster'] = labels
plt.figure(figsize=(12, 6))
sns.scatterplot(x='tests', y='ball', hue='Cluster', data=df, palette='viridis')
plt.title('Кластеризация курсов в LMS по наличию тестов для самопроверки')
plt.xlabel('Полный набор тестов в курсах LMS, %')
plt.ylabel('Баллы аттестации')
plt.xticks(rotation=90)
plt.show()

# Запускаем алгоритм стохастического градиента
from sklearn.model_selection import train_test_split
from sklearn import linear_model, metrics

train_data = df[['conspect','presentation','tests','forum_chat','video']]
train_labels = df[['ball']]
test_size = 1.0, random_state = 0
model = linear_model.SGDClassifier(alpha=0.001, max_iter=100, random_state = 0)
model.fit(train_data, train_labels)
model_predictions = model.predict(test_data)
print(metrics.accuracy_score(test_labels, model_predictions))
print(metrics.classification_report(test_labels, model_predictions))

# Подготовка данных для кластеризации
X = df[['conspect', 'presentation', 'tests', 'forum_chat', 'ball']]
# Преобразование категориальных данных в числовые
X = pd.get_dummies(X, columns=['ball'], drop_first=True)
# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Обучение модели кластеризации
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)
# Оценка модели
labels = kmeans.labels_
silhouette_avg = silhouette_score(X_scaled, labels)
print(f'Silhouette Score: {silhouette_avg}')
# Визуализация результатов кластеризации
df['Cluster'] = labels
plt.figure(figsize=(12, 6))
sns.scatterplot(x='conspect', y='ball', hue='Cluster', data=df, palette='viridis')
plt.title('Кластеризация по модулям учебных курсов в LMS')
plt.xlabel('Модели')
plt.ylabel('Баллы аттестации')
plt.xticks(rotation=90)
plt.show()






