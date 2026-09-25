# Detecção de Fraude em Cartões de Crédito

## 📌 Sobre o projeto

Este projeto foi desenvolvido como parte de um desafio de **Machine Learning** com o objetivo de identificar transações fraudulentas em cartões de crédito.

O principal desafio é o forte desbalanceamento da base de dados: a grande maioria das transações é normal, enquanto uma pequena parcela corresponde a fraudes.

Nesse cenário, utilizar apenas a **acurácia** pode gerar uma interpretação equivocada do desempenho do modelo. Por isso, este projeto dá maior atenção às métricas **Precision, Recall e F1-score**, principalmente para a classe de fraude.

---

## 🎯 Objetivo

Desenvolver um pipeline de **Machine Learning** capaz de identificar transações fraudulentas e comparar diferentes modelos e estratégias de tratamento do desbalanceamento.

Foram realizadas as seguintes etapas:

- Análise exploratória dos dados;
- Identificação do desbalanceamento;
- Criação de novas variáveis;
- Padronização dos dados;
- Divisão entre treino e teste;
- Treinamento de diferentes modelos;
- Undersampling;
- Oversampling com SMOTE;
- Ajuste do threshold;
- Avaliação utilizando diferentes métricas;
- Comparação entre modelos;
- Ajuste de hiperparâmetros com GridSearchCV;
- Análise de importância das variáveis;
- Explicabilidade utilizando SHAP.

---

## 📊 Dataset

O dataset utilizado contém transações de cartões de crédito.

As principais colunas são:

- `Time` — tempo decorrido desde a primeira transação;
- `Amount` — valor da transação;
- `V1` a `V28` — variáveis transformadas por PCA para proteção da privacidade;
- `Class` — variável alvo.

A coluna `Class` possui dois valores:

- `0` — transação normal;
- `1` — transação fraudulenta.

O dataset é carregado diretamente pela URL:
https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

O arquivo de dados não é armazenado neste repositório.

## ⚠️ Desbalanceamento das classes

O dataset possui um forte desbalanceamento entre as classes.

Aproximadamente **99,8% das transações são normais**, enquanto apenas cerca de **0,17% são fraudulentas**.

Esse desbalanceamento faz com que a **acurácia isoladamente não seja uma métrica suficiente** para avaliar o modelo.

Um modelo poderia classificar praticamente todas as transações como normais e ainda obter uma acurácia muito alta, mesmo deixando de identificar grande parte das fraudes.

Por isso, neste projeto foram analisadas principalmente as métricas da classe **Fraude**.

### Precision

Indica, entre as transações classificadas pelo modelo como fraude, quantas realmente eram fraudulentas.

### Recall

Indica, entre todas as fraudes existentes, quantas foram identificadas pelo modelo.

### F1-score

É uma métrica que combina **Precision** e **Recall**, permitindo analisar o equilíbrio entre as duas.

Neste projeto, o **Recall da classe de fraude** recebeu atenção especial, pois deixar uma transação fraudulenta passar é um dos principais problemas que o modelo precisa evitar.

---

## 🔧 Preparação dos dados

### Feature Engineering

Foi criada uma nova variável chamada `Amount_log`:

python
df["Amount_log"] = np.log1p(df["Amount"])

A transformação logarítmica reduz a influência de valores muito elevados da variável `Amount`.

### Padronização

Foi utilizado o `StandardScaler` para padronizar os dados utilizados pelo modelo.

A padronização transforma as variáveis para uma escala comparável, utilizando média próxima de zero e desvio padrão próximo de um.

### Separação entre treino e teste

Os dados foram divididos em:

- **70% para treinamento**
- **30% para teste**

Foi utilizado `stratify=y` para preservar a proporção entre as classes durante a divisão:

python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

## 🤖 Modelos utilizados

### Regressão Logística

A Regressão Logística foi utilizada como modelo **baseline**.

O objetivo foi estabelecer uma referência inicial para comparar o desempenho dos outros modelos.

### Resultado

O modelo apresentou:

- **Precision da fraude:** 6,55%
- **Recall da fraude:** 87,84%
- **F1-score da fraude:** 12,19%

Apesar do recall elevado, a precision foi baixa. Isso significa que o modelo conseguiu identificar uma grande parte das fraudes, mas também classificou muitas transações normais como fraudulentas.

---

### Random Forest

O **Random Forest** utiliza diversas árvores de decisão e combina suas previsões.

Foi utilizado `class_weight="balanced"` para considerar o desbalanceamento entre as classes.

### Resultado

- **Precision da fraude:** 83,09%
- **Recall da fraude:** 76,35%
- **F1-score da fraude:** 79,58%
- **Accuracy:** 99,93%

## XGBoost

O **XGBoost** utiliza uma estratégia de boosting, na qual árvores são construídas sequencialmente para melhorar as previsões.

Foi utilizado `scale_pos_weight` para aumentar a importância da classe minoritária durante o treinamento.

### Resultado

- **Precision da fraude:** 86,03%
- **Recall da fraude:** 79,05%
- **F1-score da fraude:** 82,39%
- **Accuracy:** 99,94%

---

# ⚖️ Técnicas de balanceamento

## Undersampling

No **undersampling**, parte dos exemplos da classe majoritária é removida para reduzir o desequilíbrio entre as classes.

Neste projeto, foram selecionadas transações normais em quantidade semelhante à quantidade de fraudes.

### Resultado

**Random Forest com undersampling:**

- **Precision:** 6,84%
- **Recall:** 86,49%
- **F1-score:** 12,67%

O método aumentou a capacidade de encontrar fraudes, mas apresentou uma **precision baixa**.

---

## SMOTE

Também foi testado o **SMOTE (Synthetic Minority Over-sampling Technique)**.

Essa técnica cria exemplos sintéticos da classe minoritária com base nos exemplos existentes.

### Resultado

**Random Forest com SMOTE:**

- **Precision:** 57,94%
- **Recall:** 83,78%
- **F1-score:** 68,51%

Em comparação com o undersampling, o **SMOTE apresentou uma precision significativamente maior**, mantendo um recall elevado.

## 🎚️ Ajuste do Threshold

Além de comparar os modelos, também foi testada a alteração do limiar de decisão.

O threshold padrão de muitos classificadores é `0.5`. Neste projeto, foi testado um threshold de:
0.30

###Resultado

Com esse ajuste, o resultado para a classe de fraude foi:

- **Precision:** 25,57%
- **Recall:** 83,11%
- **F1-score:** 39,11%

A alteração do threshold modificou o equilíbrio entre **Precision** e **Recall**.

A utilização de um threshold menor pode fazer com que mais transações sejam classificadas como suspeitas, aumentando a capacidade de encontrar fraudes, mas também aumentando a quantidade de **falsos positivos**.

## 📈 Avaliação dos modelos

Foram utilizadas diferentes métricas para avaliar os modelos:

- **Precision**
- **Recall**
- **F1-score**
- **Accuracy**
- **ROC-AUC**
- **Average Precision**
- **Curvas ROC**
- **Curva Precision-Recall**

O **Recall da classe de fraude** recebeu atenção especial devido ao forte desbalanceamento do dataset.

---

## 📊 Comparação dos modelos

A tabela abaixo apresenta os resultados obtidos para a classe de fraude.

| **Modelo** | **Precision** | **Recall** | **F1-score** |
|---|---:|---:|---:|
| Regressão Logística | 6,55% | 87,84% | 12,19% |
| Regressão Logística + Threshold | 25,57% | 83,11% | 39,11% |
| Random Forest | 83,09% | 76,35% | 79,58% |
| Random Forest + Undersampling | 6,84% | 86,49% | 12,67% |
| Random Forest + SMOTE | 57,94% | 83,78% | 68,51% |
| XGBoost | 86,03% | 79,05% | 82,39% |
| XGBoost + GridSearchCV | 62,05% | 81,76% | 70,55% |

Os resultados mostram que diferentes estratégias produzem diferentes equilíbrios entre **Precision** e **Recall**.

A **Regressão Logística** apresentou recall elevado, mas precision baixa.

O **Random Forest** apresentou uma relação mais equilibrada entre Precision e Recall.

O **XGBoost** apresentou:

- **Precision:** 86,03%
- **Recall:** 79,05%
- **F1-score:** 82,39%

As estratégias de balanceamento também apresentaram comportamentos diferentes.

O **undersampling** apresentou recall elevado, mas precision baixa, enquanto o **SMOTE** conseguiu manter um recall elevado com uma precision maior.

## 📉 Curva ROC

A **Regressão Logística** apresentou:
ROC-AUC = 0.9678

A curva ROC foi utilizada para analisar a capacidade do modelo de diferenciar as classes considerando diferentes thresholds.

Apesar do ROC-AUC elevado, a análise não foi feita utilizando essa métrica isoladamente, devido ao forte desbalanceamento existente no dataset.

## 📊 Curva Precision-Recall

A **Regressão Logística** apresentou:
Average Precision = 0.7024

A curva Precision-Recall foi utilizada como uma análise complementar, especialmente por se tratar de um problema em que a classe positiva representa uma pequena parcela das observações.

Essa curva permite observar como Precision e Recall variam de acordo com o threshold utilizado.

# 🔎 Importância das variáveis

A importância das variáveis foi analisada utilizando o modelo **XGBoost**.

Essa análise permite observar quais características foram mais utilizadas pelo modelo durante suas decisões.

Como as variáveis `V1` a `V28` passaram por transformação PCA para proteção da privacidade, seus nomes não representam diretamente características originais das transações.

Os resultados da importância das variáveis foram analisados no notebook.

---

# 🔬 Explicabilidade com SHAP

O **SHAP** foi utilizado para analisar como as variáveis contribuíram para as decisões do modelo.

Essa etapa permite complementar as métricas de desempenho com uma análise da explicabilidade das previsões.

## Resultado da análise SHAP

> **PREENCHER APÓS A ANÁLISE DO GRÁFICO SHAP.**

Será incluído aqui:

- **Gráfico SHAP;**
- **Principais variáveis;**
- **Impacto das variáveis;**
- **Análise de uma previsão individual.**

## ⚙️ GridSearchCV

Foi utilizado **GridSearchCV** para testar diferentes combinações de hiperparâmetros do **XGBoost**.

Foram avaliados os seguintes parâmetros:

- `learning_rate`
- `max_depth`
- `n_estimators`

Os melhores parâmetros encontrados foram:

python
{
    "learning_rate": 0.05,
    "max_depth": 3,
    "n_estimators": 100
}


O melhor recall médio obtido durante a validação cruzada foi:
0.8488176964149504

Ou aproximadamente: 
84,88%


Após o treinamento do modelo utilizando os parâmetros encontrados pelo **GridSearchCV**, o resultado no conjunto de teste para a classe de fraude foi:

- **Precision:** 62,05%
- **Recall:** 81,76%
- **F1-score:** 70,55%

Esse resultado mostra que o melhor desempenho durante a validação cruzada não necessariamente se traduz no maior valor de cada métrica no conjunto de teste.

Por isso, a avaliação final foi realizada separadamente no conjunto de teste.

## 🧠 Principais aprendizados

O principal aprendizado deste projeto foi compreender que a **acurácia não deve ser analisada isoladamente** em problemas altamente desbalanceados.

No conjunto de teste, os modelos apresentaram acurácias muito altas. Entretanto, quando analisamos especificamente a classe de fraude, aparecem diferenças importantes entre **Precision, Recall e F1-score**.

A comparação também mostrou que técnicas diferentes de balanceamento podem alterar significativamente o comportamento do modelo.

O **undersampling** apresentou recall elevado, mas precision baixa.

O **SMOTE** apresentou um equilíbrio diferente entre as duas métricas.

O ajuste do **threshold** também mostrou que o ponto de decisão do modelo influencia diretamente a relação entre **Precision** e **Recall**.

---

## 🗂️ Estrutura do projeto

deteccao-fraude-cartao/
│
├── README.md
│
└── deteccao_fraude_cartao.ipynb

O dataset não está armazenado no repositório.

Ele é carregado diretamente pela URL utilizada no notebook.

## ▶️ Como executar

O projeto pode ser executado utilizando o **Google Colab**.

1. Abra o arquivo `deteccao_fraude_cartao.ipynb`.
2. Execute as células na ordem.
3. O dataset será carregado automaticamente pela URL.
4. Execute as etapas de preparação dos dados.
5. Execute o treinamento dos modelos.
6. Analise as métricas e gráficos apresentados.
7. Execute as etapas de explicabilidade.

## 🛠️ Tecnologias utilizadas

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Imbalanced-learn**
- **XGBoost**
- **SHAP**
- **Matplotlib**
- **Seaborn**
- **Google Colab**

---

# 📌 Conclusão

O projeto apresentou um **pipeline de Machine Learning** aplicado à detecção de fraude em transações de cartão de crédito.

A principal dificuldade encontrada foi o forte **desbalanceamento** entre transações normais e fraudulentas.

Por esse motivo, a análise foi concentrada principalmente nas métricas da classe de fraude, especialmente **Precision, Recall e F1-score**.

Foram comparados diferentes modelos e estratégias, incluindo:

- **Regressão Logística**
- **Random Forest**
- **XGBoost**
- **Undersampling**
- **SMOTE**
- **Alteração do threshold**
- **GridSearchCV**

O projeto também incorporou técnicas de **explicabilidade** para compreender melhor as decisões realizadas pelo modelo.

Como resultado, foi possível analisar não apenas o desempenho dos modelos, mas também os efeitos das diferentes estratégias de preparação, treinamento e avaliação em um problema real de **classificação desbalanceada**.
