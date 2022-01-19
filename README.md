# Machine-learning-Modelos de estatistica

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)
[![author](https://img.shields.io/badge/author-RafaelGallo-red.svg)](https://github.com/RafaelGallo?tab=repositories) 
[![](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-374/) 
[![](https://img.shields.io/badge/R-3.6.0-red.svg)](https://www.r-project.org/)
[![](https://img.shields.io/badge/ggplot2-white.svg)](https://ggplot2.tidyverse.org/)
[![](https://img.shields.io/badge/dplyr-blue.svg)](https://dplyr.tidyverse.org/)
[![](https://img.shields.io/badge/readr-green.svg)](https://readr.tidyverse.org/)
[![](https://img.shields.io/badge/ggvis-black.svg)](https://ggvis.tidyverse.org/)
[![](https://img.shields.io/badge/Shiny-red.svg)](https://shiny.tidyverse.org/)
[![](https://img.shields.io/badge/plotly-green.svg)](https://plotly.com/)
[![](https://img.shields.io/badge/XGBoost-red.svg)](https://xgboost.readthedocs.io/en/stable/#)
[![](https://img.shields.io/badge/Tensorflow-orange.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Keras-red.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/CUDA-gree.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Caret-orange.svg)](https://caret.tidyverse.org/)
[![](https://img.shields.io/badge/Pandas-blue.svg)](https://pandas.pydata.org/) 
[![](https://img.shields.io/badge/Matplotlib-blue.svg)](https://matplotlib.org/)
[![](https://img.shields.io/badge/Seaborn-green.svg)](https://seaborn.pydata.org/)
[![](https://img.shields.io/badge/Matplotlib-orange.svg)](https://scikit-learn.org/stable/) 
[![](https://img.shields.io/badge/Scikit_Learn-green.svg)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/Numpy-white.svg)](https://numpy.org/)
[![](https://img.shields.io/badge/PowerBI-red.svg)](https://powerbi.microsoft.com/pt-br/)

![Logo](https://as2.ftcdn.net/v2/jpg/03/36/74/61/1000_F_336746173_cmSJMBmWlVPYuYnG2f8e7OwqzebUP8qs.jpg)


## Autores

- [@RafaelGallo](https://www.github.com/rafaelgallo)


## Objetivo do projeto de machine learning

Aplicando machine learning com modelos estatísticos. Realizei alguns projetos com estatística ARIMA, SARIMA projeto de previsão de novos clientes e series temporais novos preços de ação. Projeto foi feito em python ou R. 

**Projeto**
- Modelo ARIMA, SARIMA - Previsão de novos usuários no Spotify
- Modelo ARIMA - Previsão do clima durante o ano
- Modelo ARIMA, SARIMA, AR - Previsão novos preços ações

## Stack utilizada

**Programação** Python

**Leitura CSV**: Pandas

**Análise de dados**: Seaborn, Matplotlib

**Machine learning**: Scikit-learn





## Variáveis de Ambiente

Para rodar esse projeto, você vai precisar adicionar as seguintes variáveis de ambiente no seu .env

Instalando a virtualenv

`pip install virtualenv`

Nova virtualenv

`virtualenv nome_virtualenv`

Ativando a virtualenv

`source nome_virtualenv/bin/activate` (Linux ou macOS)

`nome_virtualenv/Scripts/Activate` (Windows)

Retorno da env

`projeto_py source venv/bin/activate` 

Desativando a virtualenv

`(venv) deactivate` 

Instalando pacotes

`(venv) projeto_py pip install flask`

Instalando as bibliotecas

`pip freeze`


## Instalação

Instalação das bibliotecas para esse projeto no python.

```bash
  conda install -c conda-forge pandas 
  conda install -c conda-forge scikitlearn
  conda install -c conda-forge numpy
  conda install -c conda-forge scipy

  python==3.6.4
  numpy==1.13.3
  scipy==1.0.0
  matplotlib==2.1.2
```
Instalação do Python É altamente recomendável usar o anaconda para instalar o python. Clique aqui para ir para a página de download do Anaconda https://www.anaconda.com/download. Certifique-se de baixar a versão Python 3.6. Se você estiver em uma máquina Windows: Abra o executável após a conclusão do download e siga as instruções. 

Assim que a instalação for concluída, abra o prompt do Anaconda no menu iniciar. Isso abrirá um terminal com o python ativado. Se você estiver em uma máquina Linux: Abra um terminal e navegue até o diretório onde o Anaconda foi baixado. 
Altere a permissão para o arquivo baixado para que ele possa ser executado. Portanto, se o nome do arquivo baixado for Anaconda3-5.1.0-Linux-x86_64.sh, use o seguinte comando: chmod a x Anaconda3-5.1.0-Linux-x86_64.sh.

Agora execute o script de instalação usando.


Depois de instalar o python, crie um novo ambiente python com todos os requisitos usando o seguinte comando

```bash
conda env create -f environment.yml
```
Após a configuração do novo ambiente, ative-o usando (windows)
```bash
activate "Nome do projeto"
```
ou se você estiver em uma máquina Linux
```bash
source "Nome do projeto" 
```
Agora que temos nosso ambiente Python todo configurado, podemos começar a trabalhar nas atribuições. Para fazer isso, navegue até o diretório onde as atribuições foram instaladas e inicie o notebook jupyter a partir do terminal usando o comando
```bash
jupyter notebook
```
    
## Demo modelo machine learning - Série temporal

```bash
  # Carregando as bibliotecas 
  import pandas as pd
  import numpy as np
  import seaborn as sns
  
  get_ipython().run_line_magic('matplotlib', 'inline')
  import matplotlib.pyplot as plt
  
  # Carregando base de dados
  df = pd.read_csv("data.csv")
  df.head()
  
  # Renomeando colunas
  df.columns = ["Data", "Pessoas"]
  df

  # Transformar a coluna date para datetime
  df['Date'] = pd.to_datetime(df['Date'])

  # Gráfico de linha
  plt.figure(figsize=(20, 10))
  plt.style.use('seaborn-darkgrid')
  x1 = sns.lineplot(x="Data", y="Pessoas", data=df)
  plt.xticks(rotation=70);
  plt.show()
  
  # Modelo acf e pacf
  from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
  plot_acf(df["Pessoas"]);
  
  # Gráfico pacf
  plot_pacf(df["Pessoas"]);
  
  # Definindo variavel para o modelo
  df_x1 = df[:26][:]
  df_x2 = df[26:][:]
  
  # Modelo ARIMA
  from pmdarima.arima import auto_arima
  modelo_arima = auto_arima(df_x1["Pessoas"].values, start_p = 0, start_q = 0,
                         max_p = 8, max_q = 8, d = 2, seasonal = False, trace = True,
                         error_action = "ignore", suppress_warnings = True, stepwise = False)
  modelo_arima.aic()
  
  # Previsão do modelo ARIMA
  modelo_arima.fit(df_x1["Pessoas"]. values)
  model_predict = modelo_arima.predict(n_periods = 10)
  model_predict
  
  # Gráfico Previsão
  plt.figure(figsize=(30,25))
  plt.plot(df["Data"], df["Pessoas"])
  plt.plot(model_predict)

```


## Melhorias

Que melhorias você fez no seu código? 
- Ex: refatorações, melhorias de performance, acessibilidade, etc


## Suporte

Para suporte, mande um email para rafaelhenriquegallo@gmail.com

