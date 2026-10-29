"""
Desafio: Tratamento e Limpeza de Dados - DataClean Co.
Pipeline completo de pré-processamento de dados e análise exploratória.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. CARREGAMENTO DOS DADOS
# ==========================================================
caminho = "dados_clientes.csv" if os.path.exists("dados_clientes.csv") else "dados/dados_clientes.csv"
df = pd.read_csv(caminho)

# ==========================================================
# 2. EXIBIR INFORMAÇÕES BÁSICAS
# ==========================================================
print("=" * 60)
print("1. INFORMAÇÕES BÁSICAS DO DATASET")
print("=" * 60)
print("\n--- Primeiras 5 linhas ---")
print(df.head())

print(f"\nTotal de linhas (registos): {df.shape[0]}")
print(f"Total de colunas: {df.shape[1]}")

print("\n--- Tipos de Dados por Coluna ---")
print(df.dtypes)

# ==========================================================
# 3. TRATAR DADOS AUSENTES
# ==========================================================
print("\n" + "=" * 60)
print("2. TRATAMENTO DE VALORES AUSENTES")
print("=" * 60)
print("\nValores nulos por coluna antes:")
print(df.isnull().sum())

# Tratamento numérico: preenchimento pela mediana para evitar distorção de outliers
if "idade" in df.columns:
    df["idade"] = df["idade"].fillna(df["idade"].median())

if "renda" in df.columns:
    df["renda"] = df["renda"].fillna(df["renda"].median())

if "valor_gasto" in df.columns:
    df["valor_gasto"] = df["valor_gasto"].fillna(df["valor_gasto"].median())

# Tratamento categórico: preenchimento pela moda ou valor genérico
for col_cat in ["genero", "categoria_produto", "regiao", "produto", "nome"]:
    if col_cat in df.columns:
        df[col_cat] = df[col_cat].fillna("Não Informado")

print("\nValores nulos restantes após tratamento:")
print(df.isnull().sum())

# ==========================================================
# 4. DETECTAR E TRATAR DUPLICADAS
# ==========================================================
print("\n" + "=" * 60)
print("3. TRATAMENTO DE REGISTOS DUPLICADOS")
print("=" * 60)
duplicadas_iniciais = df.duplicated().sum()
print(f"Registos duplicados identificados: {duplicadas_iniciais}")

# Remover duplicadas mantendo a primeira ocorrência
df = df.drop_duplicates(keep="first")
print(f"Linhas restantes após remover duplicadas: {len(df)}")

# ==========================================================
# 5. TRATAMENTO DE OUTLIERS
# ==========================================================
print("\n" + "=" * 60)
print("4. TRATAMENTO DE OUTLIERS (IQR)")
print("=" * 60)
colunas_numericas = [c for c in ["idade", "renda", "valor_gasto"] if c in df.columns]

# Aplicação do método de Intervalo Interquartil (IQR) com capping (limites)
for col in colunas_numericas:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    
    # Aplicação de capping para preservar o volume de dados sem distorções extremas
    df[col] = np.clip(df[col], limite_inferior, limite_superior)
    print(f"Outliers ajustados na coluna '{col}': limites [{limite_inferior:.2f}, {limite_superior:.2f}]")

# ==========================================================
# 6. TRABALHAR COM DATAS
# ==========================================================
print("\n" + "=" * 60)
print("5. ENGENHARIA DE RECURSOS TEMPORAIS")
print("=" * 60)
if "data_compra" in df.columns:
    df["data_compra"] = pd.to_datetime(df["data_compra"], errors="coerce")
    
    # Criar colunas derivadas
    df["ano_compra"] = df["data_compra"].dt.year
    df["mes_compra"] = df["data_compra"].dt.month
    df["dia_semana_compra"] = df["data_compra"].dt.day_name()
    print("Colunas 'ano_compra', 'mes_compra' e 'dia_semana_compra' adicionadas com sucesso.")

# ==========================================================
# 7. APLICAR NORMALIZAÇÃO E ENCODING
# ==========================================================
print("\n" + "=" * 60)
print("6. NORMALIZAÇÃO E ENCODING")
print("=" * 60)

# Normalização Min-Max em colunas numéricas
for col in colunas_numericas:
    min_val = df[col].min()
    max_val = df[col].max()
    if max_val > min_val:
        df[f"{col}_normalizado"] = (df[col] - min_val) / (max_val - min_val)

# One-Hot Encoding em colunas categóricas principais
colunas_encoding = [c for c in ["genero", "regiao"] if c in df.columns]
df_final = pd.get_dummies(df, columns=colunas_encoding, drop_first=False)
print("Normalização Min-Max e One-Hot Encoding aplicados.")

# ==========================================================
# 8. EXPORTAÇÃO DO DATASET LIMPO (EXTRA OPCIONAL)
# ==========================================================
df_final.to_csv("dados_clientes_limpos.csv", index=False)
print("\nDataset limpo exportado para: 'dados_clientes_limpos.csv'")

# ==========================================================
# 9. GERAÇÃO DE VISUALIZAÇÕES (EXTRA OPCIONAL)
# ==========================================================
print("\n" + "=" * 60)
print("7. GERANDO VISUALIZAÇÕES")
print("=" * 60)

# 1. Histograma: Distribuição de Renda
if "renda" in df.columns:
    plt.figure(figsize=(7, 4))
    plt.hist(df["renda"], bins=15, color="#3470a3", edgecolor="black")
    plt.title("Distribuição da Renda dos Clientes")
    plt.xlabel("Renda")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig("distribuicao_renda.png")
    plt.close()

# 2. Boxplot: Gasto por Categoria
if "categoria_produto" in df.columns and "valor_gasto" in df.columns:
    plt.figure(figsize=(8, 4.5))
    df.boxplot(column="valor_gasto", by="categoria_produto", grid=False)
    plt.title("Gasto Médio por Categoria de Produto")
    plt.suptitle("")
    plt.xlabel("Categoria")
    plt.ylabel("Valor Gasto")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("boxplot_categoria_gasto.png")
    plt.close()

# 3. Scatterplot: Idade vs Valor Gasto
if "idade" in df.columns and "valor_gasto" in df.columns:
    plt.figure(figsize=(7, 4))
    plt.scatter(df["idade"], df["valor_gasto"], alpha=0.6, color="#e74c3c")
    plt.title("Relação entre Idade e Valor Gasto")
    plt.xlabel("Idade")
    plt.ylabel("Valor Gasto")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("scatter_idade_gasto.png")
    plt.close()

# 4. Matriz de Correlação
cols_corr = [c for c in ["idade", "renda", "valor_gasto"] if c in df.columns]
if len(cols_corr) > 1:
    plt.figure(figsize=(6, 4.5))
    matriz_corr = df[cols_corr].corr()
    plt.imshow(matriz_corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(cols_corr)), cols_corr)
    plt.yticks(range(len(cols_corr)), cols_corr)
    plt.title("Matriz de Correlação")
    for i in range(len(cols_corr)):
        for j in range(len(cols_corr)):
            plt.text(j, i, f"{matriz_corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig("heatmap_correlacao.png")
    plt.close()

print("Gráficos gerados com sucesso na pasta do projeto.")
