import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Carregamento dos dados
df = pd.read_csv("experiencia_salario.csv")

# Separação de var independente (experiência) e dependente (salário)
X, y = df.iloc[:, :-1].values, df.iloc[:, 1].values

# Separação CRUZADA de variáveis de teste (1/3) e trainamento (2/3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Instanciar a classe
linear_regression = LinearRegression()

# Treinar o modelo .fit()
linear_regression.fit(X_train, y_train)

# Predição do modelo para os dados de teste
y_predict = linear_regression.predict(X_test)

# Visualização dos dados
plt.figure(figsize=(15,8))
plt.plot(X, y, "^", label="Dados de trainamento")
plt.plot(X_test, y_test, "Dr", label="Dados de testes")
plt.plot(X_test, y_predict, color="blue")
plt.title("Experiências x Salários")
plt.xlabel("Anos de experiências")
plt.ylabel("Salários (R$)")
plt.legend()
plt.show()
