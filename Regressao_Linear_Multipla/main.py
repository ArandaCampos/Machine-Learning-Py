import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Carregamento de dados
dataset = pd.read_csv("aptos.csv")

# Separação das variáveis independentes (X) e dependentes (y)
X, y = dataset.iloc[:,:-1].values, dataset.iloc[:,-1].values

# Separação CRUZADA das variáveis de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Treinamento do model
model = LinearRegression()
model.fit(X_train, y_train)

# Predição do modelo diante das variáveis de teste
y_predict = model.predict(X_test)

# Relatório
for i in range(len(y_predict)):
    erro = (y_predict[i] - y_test[i]) / y_test[i] * 100
    if erro < 0: erro *= -1
    print('Predição: {:.1f}\tReal: {:.2f}\tErro: {:.2f}%'.format(y_predict[i], y_test[i], erro))
