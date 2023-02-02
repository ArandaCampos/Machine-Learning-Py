import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("estudos_notas.csv")

i = dataset.iloc[:,1:2]
print(i)
X, y = dataset.iloc[:,1:2].values, dataset.iloc[:,-1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

model =  PolynomialFeatures(degree = 2)

X_poly = model.fit_transform(X_train)

model.fit(X_poly, y_train)
model.fit(X_train, y_train)

model_linear = LinearRegression()
model_linear.fit(X_poly, y_train)

plt.figure(figsize=(15,8))
plt.plot(X_train, y_train, "^", label="Dados de treinamento")
plt.plot(X_test, model_linear.predict(model.fit_transform(X_test)), "Dr", label="Dados de teste")
plt.title("Tempo de estudos x Pontuação (Regressão Polinomial)")
plt.xlabel("Tempo de estudos (min)")
plt.ylabel("Pontuação")
plt.legend()
plt.show()
