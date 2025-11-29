import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import metrics
from tabulate import tabulate

# Leitura do banco de dados
caminho = (r"sinresp.csv")
df = pd.read_csv(caminho, on_bad_lines='skip', sep=';', low_memory=False)

# Remove pacientes com EVOLUCAO igual a 9 (ignorados) ou 3 (óbitos não relacionados)
df = df[df['EVOLUCAO'].isin([1, 2])]

# Transforma EVOLUCAO em binário, sendo assim, vivo é indicado por 0 e morto por 1
df['EVOLUCAO'] = df['EVOLUCAO'].replace({1: 0, 2: 1})

# Remove colunas com muitos NaN
limite = len(df) * 0.5
df = df.dropna(thresh = limite, axis = 1)

# Remove linhas sem a coluna alvo (EVOLUCAO)
df = df.dropna(subset = ['EVOLUCAO'])

# Colunas desnecessárias que serão apagadas do banco de dados
colunas_apagadas = ['SG_UF_NOT','ID_REGIONA','CO_REGIONA','ID_MUNICIP','CO_MUN_NOT','ID_PAIS','CO_PAIS','SG_UF','ID_RG_RESI','CO_RG_RESI','ID_MN_RESI','CO_MUN_RES','SG_UF_INTE','ID_RG_INTE','CO_RG_INTE','ID_MN_INTE','CO_MU_INTE','NM_UN_INTE','DT_NOTIFIC','DT_SIN_PRI','DT_NASC','DT_INTERNA','DT_ENTUTI','DT_SAIDUTI','DT_RAIOX','DT_COLETA','DT_PCR','DT_EVOLUCA','DT_ENCERRA','DT_DIGITA','DT_VGM','DT_RT_VGM','DT_TOMO','DT_RES_AN','DT_CO_SOR','DT_RES','DT_TRT_COV','OUTRO_DES','OUT_MORBI','MORB_DESC','OUT_ANTIV','RAIOX_OUT','OUT_AMOST','FLUASU_OUT','FLUBLI_OUT','DS_PCR_OUT','CLASSI_OUT','PAC_DSCBO','OUT_ANIM','TOMO_OUT','DS_AN_OUT','SOR_OUT','OUT_SOR','OUT_TRAT','VG_OMSOUT','VG_METOUT','FAB_COV_1','FAB_COV_2','FAB_COVRF','FAB_COVRF2','FAB_ADIC','FAB_RE_BI','LOTE_1_COV','LOTE_2_COV','LOTE_REF','LOTE_REF2','LOTE_ADIC','LOT_RE_BI','FNT_IN_COV']

# Define quais delas existem no banco de dados
colunas_apagadas = [c for c in colunas_apagadas if c in df.columns]

# Conta quantos casos existem na classe de óbitos
contador_obitos = len(df[df['EVOLUCAO'] == 1])

# Separa as classes em dois bancos diferentes
df_vivos = df[df['EVOLUCAO'] == 0]
df_obitos = df[df['EVOLUCAO'] == 1]

# Realiza o sorteio aleatório dos vivos para ficar com o mesmo tamanho dos óbitos
df_vivos_reduzido = df_vivos.sample(n=contador_obitos, random_state=42)

# Junta os dois grupos
df_balanceado = pd.concat([df_vivos_reduzido, df_obitos])

# Embaralha as linhas para não ficarem ordenadas
df = df_balanceado.sample(frac=1, random_state=42).reset_index(drop=True)

# Remove as colunas desnecessárias definidas anteriormente
X = df.drop(columns = colunas_apagadas + ['EVOLUCAO'])

#Seleciona colunas não numéricas para as converter em numéricas
colunas_categ = ['CS_SEXO','CS_RACA','CS_ETINIA','CS_ESCOL_N','CS_GESTANT','ID_UNIDADE','TP_ANTIVIR','AMOSTRA','RAIOX_RES','TOMO_RES','TP_FLU_PCR','TP_FLU_AN']

# Define quais delas existem no banco de dados
colunas_categ = [c for c in colunas_categ if c in X.columns]

# Aplicação do LabelEncoder
le = LabelEncoder()
for colunas in colunas_categ:
    X[colunas] = le.fit_transform(X[colunas].astype(str))

# Converte todo o restante para numérico
X = X.apply(pd.to_numeric, errors='coerce')

#Preenchendo os valores ausentes das colunas com sua média
X = X.fillna(X.mean())

# Normalização dos dados no banco de dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define a coluna EVOLUCAO como alvo
Y = df.EVOLUCAO

# Divisão em treino e teste
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 10)

# Cria o modelo com 11 vizinhos
clf = RandomForestClassifier(n_estimators=100, random_state=10)

# Realiza o treinamento do modelo
clf.fit(x_train, y_train)

# Retorna as probabilidades previstas para cada classe
pred_scores = clf.predict_proba(x_test)

# Realiza a predição final
y_pred = clf.predict(x_test)

# Calcula a acurácia e realiza a validação cruzada
te_acc= (sklearn.metrics.accuracy_score(y_test, y_pred))
scores = cross_val_score(clf, X, Y, cv=5)

# Gera o relatório
report = classification_report(y_test, y_pred, output_dict=True)

# Converte para DataFrame
df_report = pd.DataFrame(report).transpose()

# Mantém 3 casas decimais
df_report = df_report.round(3)

# Reorganiza colunas para garantir ordem correta
df_report = df_report[["precision", "recall", "f1-score", "support"]]

# Exibe o relatório
print("\n=================== RELATÓRIO DE CLASSIFICAÇÃO ===================\n")
print(tabulate(
    df_report,
    headers="keys",
    tablefmt="github",
    colalign=("left", "right", "right", "right", "right")
), "\n")