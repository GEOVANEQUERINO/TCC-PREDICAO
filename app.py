#importando bibliotecas
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():

# carregar o banco de dados
data = pd.read_csv('banco_de_dados.csv')

# dividir os dados em variaveis entrada (X) e saida (Y)
X = data.drop('vendas', axis=1)
Y = data ['vendas']

# dividir os dados para treinamento e teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# treinamento modelo regressão linear
model = LinearRegression()
model.fit(X_train, Y_train)

# fazendo a predição com os dados enviados do formulario
entrada = [float(request.form['var1']), float(request.form['var2']), float(request.form['var3'])]
predicao = model.predict([entrada])

# associar as variaveis de entrada com os coeficientes do modelo
associacoes = pd.DataFrame({'Variavel' : X.columns, 'Coeficiente' : model.coef_})

return render_template('result.html', predicao = predicao [0], associacoes = associacoes.to_html())

if __name__ == '__main__':
    app.run(debug = True)
