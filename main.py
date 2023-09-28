# importando as bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# carregar o banco de dados
data = pd.read_csv('banco_de_dados.csv')

# dividir os dados em variaveis
X = data.drop('vendas', axis=1)
Y = data['vendas']

# treinamento e teste
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

# treinar modelo regressão linear
model = LinearRegression()
model.fit(X_train, Y_train)

# fazer a predição
Y_pred = model.predict(X_test)

# calcular o coeficiente de determinação (R²)
r2 = r2_score(Y_test, Y_pred)
print('Coificiente de Determinação (R²):', r2)

# Associar as variaveis de entrada com os coeficientes  do modelo
associacoes = pd.DataFrame({'Variavel': X.columns, 'Coeficiente': model.coef_})
print(associacoes)
