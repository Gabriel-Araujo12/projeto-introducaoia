import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tabulate import tabulate
import os

# ------------------------ LEITURA DA BASE DE DADOS ------------------------

caminho = "sinresp.csv"
df = pd.read_csv(caminho, on_bad_lines='skip', sep=';', low_memory=False)

# ------------------------- BALANCEAMENTO DA BASE --------------------------

# Remove pacientes com EVOLUCAO igual a 9 (ignorados) ou 3 (óbitos não relacionados)
df = df[df['EVOLUCAO'].isin([1, 2])]

# Transforma EVOLUCAO em binário, sendo assim, vivo é indicado por 0 e morto por 1
df['EVOLUCAO'] = df['EVOLUCAO'].replace({1: 0, 2: 1})

# Remove colunas com muitos NaN
limite = len(df) * 0.5
df = df.dropna(thresh = limite, axis = 1)

# Remove linhas sem a coluna alvo (EVOLUCAO)
df = df.dropna(subset = ['EVOLUCAO'])

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

# --------------------------- TRATAMENTO DA BASE ---------------------------

# Colunas desnecessárias que serão apagadas do banco de dados
colunas_apagadas = ['SG_UF_NOT','ID_REGIONA','CO_REGIONA','ID_MUNICIP','CO_MUN_NOT','ID_PAIS','CO_PAIS','SG_UF','ID_RG_RESI','CO_RG_RESI','ID_MN_RESI','CO_MUN_RES','SG_UF_INTE','ID_RG_INTE','CO_RG_INTE','ID_MN_INTE','CO_MU_INTE','NM_UN_INTE','DT_NOTIFIC','DT_SIN_PRI','DT_NASC','DT_INTERNA','DT_ENTUTI','DT_SAIDUTI','DT_RAIOX','DT_COLETA','DT_PCR','DT_EVOLUCA','DT_ENCERRA','DT_DIGITA','DT_VGM','DT_RT_VGM','DT_TOMO','DT_RES_AN','DT_CO_SOR','DT_RES','DT_TRT_COV','OUTRO_DES','OUT_MORBI','MORB_DESC','OUT_ANTIV','RAIOX_OUT','OUT_AMOST','FLUASU_OUT','FLUBLI_OUT','DS_PCR_OUT','CLASSI_OUT','PAC_DSCBO','OUT_ANIM','TOMO_OUT','DS_AN_OUT','SOR_OUT','OUT_SOR','OUT_TRAT','VG_OMSOUT','VG_METOUT','FAB_COV_1','FAB_COV_2','FAB_COVRF','FAB_COVRF2','FAB_ADIC','FAB_RE_BI','LOTE_1_COV','LOTE_2_COV','LOTE_REF','LOTE_REF2','LOTE_ADIC','LOT_RE_BI','FNT_IN_COV']

# Define quais delas existem no banco de dados
colunas_apagadas = [c for c in colunas_apagadas if c in df.columns]

# Define o X e Y
X = df.drop(columns = colunas_apagadas + ['EVOLUCAO'])
Y = df.EVOLUCAO

# Colunas identificadas como categóricas
colunas_categ = ['CS_SEXO','CS_RACA','CS_ETINIA','CS_ESCOL_N','CS_GESTANT','ID_UNIDADE','TP_ANTIVIR','AMOSTRA','RAIOX_RES','TOMO_RES','TP_FLU_PCR','TP_FLU_AN']

# Define quais delas existem no banco de dados
colunas_categ = [c for c in colunas_categ if c in X.columns]

# ---------------------------- PREPARAÇÃO FINAL ----------------------------

# Dicionário onde os encoders serão salvos
encoders = {}

# Aplicação do LabelEncoder
for coluna in colunas_categ:
    le = LabelEncoder()
    X[coluna] = le.fit_transform(X[coluna].astype(str))
    encoders[coluna] = le

# Converte todo o restante para numérico
X = X.apply(pd.to_numeric, errors='coerce')

# Preenche os valores ausentes das colunas com a média
medias = X.mean()
X = X.fillna(medias)

# Salva as colunas para garantir a ordem
colunas_modelo = X.columns

# ------------------------ TREINAMENTO E RELATÓRIO -------------------------

# Divisão em treino e teste
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 10)

# Normalização dos dados
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Cria o modelo com 11 vizinhos
clf = KNeighborsClassifier(n_neighbors = 11)

# Realiza o treinamento do modelo
clf.fit(x_train, y_train)

# Realiza a predição final
y_pred = clf.predict(x_test)

# Gera o relatório
relatorio = classification_report(y_test, y_pred, output_dict=True)

# Converte para dataframe e mantém 3 casas decimais
df_relatorio = pd.DataFrame(relatorio).transpose().round(3)

# Reorganiza colunas para garantir ordem correta
df_relatorio = df_relatorio[["precision", "recall", "f1-score", "support"]]

# Exibe o relatório
print("\n=================== RELATÓRIO DE CLASSIFICAÇÃO ===================\n")
print(tabulate(df_relatorio, headers="keys", tablefmt="github", colalign=("left", "right", "right", "right", "right")))

# ------------------------- INTERFACE DE USUÁRIO ---------------------------

while True:
    arquivo = input("\nDigite o nome do arquivo CSV (ou '0' para encerrar o programa): ")

    if arquivo == '0':
        print(f"\nEncerrando o programa...\n")
        break
    
    if not os.path.exists(arquivo):
        print(f"\nERRO: O arquivo não foi encontrado.")
        continue

    df_novo = pd.read_csv(arquivo, on_bad_lines='skip', sep=';', low_memory=False)
    ids_pacientes = df_novo['NU_NOTIFIC'] if 'NU_NOTIFIC' in df_novo.columns else df_novo.index
    df_processado = df_novo.reindex(columns=colunas_modelo)

    for coluna in colunas_categ:
        if coluna in df_processado.columns:
            le = encoders[coluna]
            df_processado[coluna] = df_processado[coluna].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )

    df_processado = df_processado.apply(pd.to_numeric, errors='coerce')
    df_processado = df_processado.fillna(medias)

    X_novo = scaler.transform(df_processado)

    predicoes = clf.predict(X_novo)
    probabilidades = clf.predict_proba(X_novo)

    df_resultado = pd.DataFrame({
        'ID DO PACIENTE': ids_pacientes,
        'PREVISÃO': predicoes,
        'STATUS': ['ÓBITO' if p == 1 else 'PROVÁVEL ALTA' for p in predicoes],
        'PROBABILIDADE DE ÓBITO (%)': (probabilidades[:, 1] * 100).round(2)
    })

    print("\n===================== RELATÓRIO DA NOVA BASE =====================\n")
    print(tabulate(df_resultado.head(10), headers='keys', tablefmt='github', showindex=False, colalign=("left", "right", "right", "right")))