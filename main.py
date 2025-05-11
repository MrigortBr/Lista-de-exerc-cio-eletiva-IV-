import pandas as pd
import unicodedata
import re
from tabulate import tabulate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import plotly.graph_objects as go
import plotly.express as px

class DataLoad:
    # Constructor principal
    def __init__(self, data):
        self.__data = None  # atributo "privado"
        self.load_data(data)

    def load_data(self, data):
        try:
            self.__data = pd.read_csv(data)
            print("Dados carregados com sucesso.")
            print(self.__data.head())
        except FileNotFoundError:
            print("Arquivo não encontrado.")

    #1 - Crie uma função que receba um DataFrame e um dicionário de renomeação e retorne o DataFrame com os nomes de colunas alterados.
    def renomear_colunas(self, renomeacoes: dict):
        if self.__data is not None:
            self.__data = self.__data.rename(columns=renomeacoes)
            print("Colunas renomeadas com sucesso.")
        else:
            print("Nenhum dado carregado para renomear.")

    #2 - Crie uma função que remova outliers de uma coluna numérica, usando o critério de desvio padrão ou outro critério justificado.
    def remover_outliers(self, coluna: str):
        Q1 = self.__data[coluna].quantile(0.25)
        Q3 = self.__data[coluna].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        df_filtrado = self.__data[(self.__data[coluna] >= limite_inferior) & (self.__data[coluna] <= limite_superior)]
        print(df_filtrado)

    #3 - Crie uma função que padronize o texto de uma coluna: tudo em minúsculo, sem acento e sem espaços duplicados.
    def padronizar_texto(self, coluna: str):
        # Função para remover acentos
        def remover_acentos(texto: str) -> str:
            nfkd = unicodedata.normalize('NFKD', texto)
            return ''.join([c for c in nfkd if not unicodedata.combining(c)])

        # Verifica se a coluna existe
        if coluna in self.__data.columns:
            self.__data[coluna] = self.__data[coluna].str.lower()  # Converte para minúsculo
            self.__data[coluna] = self.__data[coluna].apply(remover_acentos)  # Remove acentos
            self.__data[coluna] = self.__data[coluna].str.replace(r'\s+', ' ', regex=True)  # Remove espaços duplicados
            self.__data[coluna] = self.__data[coluna].str.strip()  # Remove espaços no início e no fim
            print(f"Coluna '{coluna}' padronizada com sucesso.")
        else:
            print(f"A coluna '{coluna}' não existe no DataFrame.")

    #4 - Crie uma função que retorne média, mediana, desvio padrão, mínimo e máximo de uma coluna numérica selecionada.
    def estatisticas_coluna(self, coluna: str):
        if coluna in self.__data.columns:
            media = self.__data[coluna].mean()
            mediana = self.__data[coluna].median()
            desvio_padrao = self.__data[coluna].std()
            minimo = self.__data[coluna].min()
            maximo = self.__data[coluna].max()

            print(f"Estatísticas da coluna '{coluna}':")
            print(f"Média: {media}")
            print(f"Mediana: {mediana}")
            print(f"Desvio Padrão: {desvio_padrao}")
            print(f"Mínimo: {minimo}")
            print(f"Máximo: {maximo}")
        else:
            print(f"A coluna '{coluna}' não existe no DataFrame.")

    #5 - Crie uma função que transforme uma variável categórica em valores numéricos usando um método à sua escolha (ex: one-hot, map, label encoder).
    def transformar_categoria(self, coluna: str, metodo: str = 'label'):
        if coluna in self.__data.columns:
            if metodo == 'label':
                # Label Encoding: Atribui um número único a cada categoria
                self.__data[coluna] = self.__data[coluna].astype('category').cat.codes
                print(f"Coluna '{coluna}' transformada usando Label Encoding.")
            
            elif metodo == 'onehot':
                # One-Hot Encoding: Cria novas colunas para cada categoria
                self.__data = pd.get_dummies(self.__data, columns=[coluna], prefix=[coluna])
                print(f"Coluna '{coluna}' transformada usando One-Hot Encoding.")
            
            else:
                print(f"Método '{metodo}' não reconhecido. Use 'label' ou 'onehot'.")
        else:
            print(f"A coluna '{coluna}' não existe no DataFrame.")

    #6 - Crie uma função que normalize os dados (min-max scaling) ou padronize (z-score) uma ou mais colunas numéricas.
    def normalizar_ou_padronizar(self, colunas: list, metodo: str = 'minmax'):
        if not all(col in self.__data.columns for col in colunas):
            print("Algumas colunas não existem no DataFrame.")
            return
        
        # Cria os objetos dos escaladores
        if metodo == 'minmax':
            scaler = MinMaxScaler()  # Normalização Min-Max
            print(f"Normalizando as colunas {colunas} com Min-Max Scaling.")
        
        elif metodo == 'zscore':
            scaler = StandardScaler()  # Padronização Z-Score
            print(f"Padronizando as colunas {colunas} com Z-Score.")

        else:
            print(f"Método '{metodo}' não reconhecido. Use 'minmax' ou 'zscore'.")
            return
        
        # Aplica a transformação nas colunas selecionadas
        self.__data[colunas] = scaler.fit_transform(self.__data[colunas])
        print(f"Colunas '{colunas}' transformadas com sucesso.")

    #7 - Crie uma função que receba uma coluna e exiba um histograma e um boxplot, lado a lado, com Plotly.
    def exibir_histograma_e_boxplot(self, coluna: str):
        if coluna in self.__data.columns:
            # Histograma
            histograma = go.Histogram(
                x=self.__data[coluna],
                nbinsx=20,
                name='Histograma',
                marker=dict(color='rgba(255, 99, 132, 0.7)'),
            )

            # Boxplot
            boxplot = go.Box(
                y=self.__data[coluna],
                boxmean='sd',  # Mostrar a média e os desvios padrão no boxplot
                name='Boxplot',
                marker=dict(color='rgba(99, 255, 132, 0.7)'),
            )

            # Layout
            layout = go.Layout(
                title=f'Histograma e Boxplot da coluna {coluna}',
                xaxis=dict(title=coluna),
                yaxis=dict(title="Valores"),
                showlegend=False,
                height=500,
                width=1000,
            )

            # Exibição lado a lado
            fig = go.Figure(data=[histograma, boxplot], layout=layout)
            fig.show()
        else:
            print(f"A coluna '{coluna}' não existe no DataFrame.")

    #8 - Crie uma função que exiba a contagem e o percentual de valores nulos por coluna.
    def contar_nulos(self):
        nulos = self.__data.isnull().sum()  # Contagem de valores nulos por coluna
        percentual_nulos = (nulos / len(self.__data)) * 100  # Percentual de valores nulos
        
        # Exibe o resultado
        resultado = pd.DataFrame({
            'Contagem de Nulos': nulos,
            'Percentual de Nulos (%)': percentual_nulos
        })
        
        print("Contagem e Percentual de Valores Nulos por Coluna:")
        print(resultado)
    
    #9 - Crie uma função que classifique valores de uma coluna em categorias (ex: "baixo", "médio", "alto"), com base em regras personalizadas.
    def classificar_coluna(self, coluna: str, limites: dict):
        if coluna in self.__data.columns:
            def classificar(valor):
                for categoria, intervalo in limites.items():
                    if intervalo[0] <= valor <= intervalo[1]:
                        return categoria
                return 'outros'  # Caso o valor não se encaixe em nenhum intervalo

            # Aplica a função de classificação na coluna
            self.__data[f'{coluna}_classificado'] = self.__data[coluna].apply(classificar)
            print(f"Coluna '{coluna}' classificada com sucesso.")
        else:
            print(f"A coluna '{coluna}' não existe no DataFrame.")

    #10 - Crie uma função que agrupe uma variável numérica em faixas (ex: idade em jovens, adultos, idosos) usando `pd.cut()` ou lógica condicional.
    def agrupar_em_faixas(self, coluna: str, faixas: list, labels: list):
        if coluna in self.__data.columns:
            # Usando pd.cut() para agrupar os dados
            self.__data[f'{coluna}_faixa'] = pd.cut(self.__data[coluna], bins=faixas, labels=labels, right=False)
            print(f"Coluna '{coluna}' agrupada em faixas com sucesso.")
        else:
            print(f"A coluna '{coluna}' não existe no DataFrame.")

    def print_data(self):
        print(self.__data)

    #Listar dados mais de forma mais agradavel
    def print_data_tabulate(self):
        print(tabulate(dl.get_data(), headers='keys', tablefmt='pretty', showindex=False))
    

    def get_data(self):
        return self.__data


# Cria o objeto e carrega os dados
dl = DataLoad("./data/base_dados.csv")

faixas = [0, 18, 60, 100]  # Faixas de idade: até 18, 19 a 60, acima de 60
labels = ['Jovem', 'Adulto', 'Idoso']  # Rótulos para as faixas
# Acessa os dados atualizados
dl.agrupar_em_faixas('Idade', faixas, labels)

dl.print_data_tabulate()




