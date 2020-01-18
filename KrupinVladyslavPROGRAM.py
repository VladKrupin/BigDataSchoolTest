import pandas as pd
import numpy as np

#імпорт даних
tabular=pd.read_csv('tabular_data.csv')
hashed=pd.read_csv('hashed_data.csv')
train=pd.read_csv('train_target.csv')
test=pd.read_csv('test_target.csv')

#перетворення списку з переліком захешованих значень категоріальної змінної
#у матрицю розміром кількість унікальних значень категорії на кількість користувачів
id_n=1
user_char=''
hashed_val=[]
for i in range(len(hashed)):
  if id_n in np.unique(hashed['ID']):
    if id_n==hashed['ID'][i]:
      user_char+=hashed['HASH'][i]+' '
    else:
      hashed_val.append(user_char)
      user_char=hashed['HASH'][i]+' '
      id_n+=1
  else:
    hashed_val.append('')
    id_n+=1
hashed_val.append(user_char)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=None)
cat_var=cv.fit_transform(hashed_val).toarray() 

#заміна NaN значень на медіану  
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN', strategy='median', axis=0)
tabular_new=pd.DataFrame(imputer.fit_transform(tabular))

#знаходження середнього значення числових показників за 3 періоди для кожного користувача
tabular_new=tabular_new.groupby(1)[[i for i in range(2,45)]].mean()

#об'єднання таблиці з числовими та таблиці з категоріальними значеннями
X_set=np.concatenate((tabular_new, cat_var), axis=1)

#розбиття датасету на тренувальний та тестовий
X_train=X_set[0:3871,:]
X_test=X_set[3871:,:]

#створення цільової функції
Y_train=train.iloc[:,1].values

#нормалізація числових характеристик з метою приведення до однієї розмірності
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train[:,0:43]=sc_x.fit_transform(X_train[:,0:43])
X_test[:,0:43]=sc_x.transform(X_test[:,0:43])

#зменшення кількості змінних
from sklearn.decomposition import PCA
pca=PCA(n_components=59)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
explained_variance=pca.explained_variance_ratio_

#створення та навчання класифікатора на тренувальному наборі
from xgboost import XGBClassifier
classifier=XGBClassifier(n_estimators=124, random_state=0)
classifier.fit(X_train, Y_train)

#крос-валідація класифікатора на тренувальній вибірці
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier, X=X_train, y=Y_train, cv=10)
accuracies.mean() #0.71

#прогнозування ймовірностей належності до класу 0 або 1
Y_pred_prob=pd.DataFrame(classifier.predict_proba(X_test))
test['SCORE']=Y_pred_prob[1]

#збереження результатів
test.to_csv('KrupinVladyslav_test.txt', index=False)