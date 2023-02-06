import matplotlib.pyplot as plt
from sklearn import datasets, tree

# Carregamento de dados e separação de features (característas) e target (etiquetas)
dataset = datasets.load_iris()
X, y = dataset.data, dataset.target

# Instância do modelo e treinamento
model = tree.DecisionTreeClassifier()
model.fit(X, y)

# Visualização da árvore de decisão
figure = plt.figure(figsize=(10,8))
tree.plot_tree(model, feature_names=dataset.feature_names, class_names=dataset.target_names, filled=True)
plt.show()

# Visualização da classificação no plano cartesiano
plt.figure(figsize=(15,10))
plt.scatter(dataset.data[:, 0], dataset.data[:, 2], c=dataset.target)
plt.xlabel(dataset.feature_names[0])
plt.ylabel(dataset.feature_names[2])
plt.tight_layout()
plt.show()
