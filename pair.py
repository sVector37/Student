# -*- coding: utf-8 -*-
#Created on Sun Nov 17 16:31:01 2024

import numpy as np
import pandas as pd
import h2o
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
