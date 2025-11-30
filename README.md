# Predição da Evolução de Pacientes Portadores de SRAG

**Disciplina:** Introdução à Inteligência Artificial  
**Semestre:** 2025.2  
**Professor:** Andre Luis Fonseca Faustino
**Turma:** T03

## Integrantes do Grupo
* Gabriel Araújo Ribeiro (20240022019)
* Kauê Rebouças de Araujo (20240032928)
* Marcelo Augusto Gomes Bastos de Araújo (20250033007)

## Descrição do Projeto
O projeto tem como principal objetivo desenvolver um sistema inteligente de apoio à decisão médica, focado na predição de risco de óbito em pacientes com Síndrome Respiratória Aguda Grave (SRAG).

O modelo foi criado usando o KNN, algoritmo de aprendizado de máquina supervisionado, e um banco de dados disponibilizado no site OpenDataSUS para agir como a base do conhecimento. Além disso, também foram utilizadas algumas bibliotecas da linguagem Python, como Pandas e Scikit-Learn.

## Guia de Instalação e Execução

### 1. Instalação das Dependências
Certifique-se de ter o **Python 3.x** instalado. Clone o repositório e instale as bibliotecas listadas no `requirements.txt`:

```bash
# Clone o repositório
git clone [https://github.com/Gabriel-Araujo12/projeto-introducaoia.git](https://github.com/Gabriel-Araujo12/projeto-introducaoia.git)

# Entre na pasta do projeto
cd projeto-introducaoia

# Instale as dependências
pip install -r requirements.txt
````

### 2. Como Executar

Execute o comando abaixo no terminal para inicializar o projeto:

```bash
# Execute o código principal
python main.py
```

## Estrutura dos Arquivos

  * `src/`: Código-fonte da aplicação.
  * `data/`: Datasets utilizados.

## Resultados e Demonstração

<img width="581" height="461" alt="image" src="https://github.com/user-attachments/assets/dd102031-afca-4331-94d3-4cff2139298c" />

## Referências

  * https://opendatasus.saude.gov.br/dataset/srag-2021-a-2024
  * https://www.ibm.com/br-pt/think/topics/knn
  * https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall
  * https://www.kaggle.com/code/alirezahasannejad/knn-tutorial-breast-cancer
