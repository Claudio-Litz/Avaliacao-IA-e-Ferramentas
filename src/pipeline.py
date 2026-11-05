"""Limpeza e criação de atributos para os sensores da VulcanTech."""
from pathlib import Path

import pandas as pd

SENSORES = [
    "temperatura_solda_c",
    "corrente_solda_a",
    "tensao_solda_v",
    "pressao_gas_bar",
    "vibracao_mm_s",
    "velocidade_linha_m_min",
    "area_porosidade_px",
    "temperatura_max_haz_c",
    "tempo_planejado_min",
    "tempo_parada_min",
]

# Limites físicos indicados no enunciado. Valores fora da faixa são tratados como ausentes.
LIMITES = {
    "temperatura_solda_c": (50, 600),
    "corrente_solda_a": (50, 400),
    "tensao_solda_v": (0, 50),
    "pressao_gas_bar": (0, 40),
    "vibracao_mm_s": (0, 20),
    "velocidade_linha_m_min": (0, 50),
    "area_porosidade_px": (0, 100),
    "temperatura_max_haz_c": (50, 400),
    "tempo_planejado_min": (0, 4),
    "tempo_parada_min": (-1, 4),
    "pecas_produzidas": (0, 50),
    "pecas_boas": (0, 50),
    "pecas_defeituosas": (0, 50),
    "disponibilidade_pct": (0, 100),
    "desempenho_pct": (0, 100),
    "qualidade_pct": (0, 100),
    "oee_pct": (0, 100),
    "anomalia_isolamento": (0, 1),
}


def carregar_e_preparar(caminho_csv: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dados = pd.read_csv(caminho_csv, parse_dates=["timestamp"])
    ausencias = (
        dados.groupby(["lote_id"])[SENSORES]
        .apply(lambda grupo: grupo.isna().mean().mul(100).round(2))
        .reset_index()
    )

    print(dados.describe())

    

    dados = dados.sort_values(["lote_id", "timestamp"]).copy()
    for coluna, (minimo, maximo) in LIMITES.items():
        fora_faixa = ~dados[coluna].between(minimo, maximo)
        dados.loc[fora_faixa, coluna] = pd.NA


    '''
    # Z-score calculado dentro de cada liga para não confundir material duro com falha.
    for coluna in SENSORES:
        media = dados.groupby("liga_metalica")[coluna].transform("mean")
        desvio = dados.groupby("liga_metalica")[coluna].transform("std").replace(0, 1)
        z_score = (dados[coluna] - media) / desvio
        dados.loc[z_score.abs() > 3, coluna] = pd.NA
    '''



    # Interpolação temporal dentro de cada célula, limitada a 2 amostras consecutivas.
    dados = dados.set_index("timestamp")
    for coluna in SENSORES:
        dados[coluna] = dados.groupby("lote_id", group_keys=False)[coluna].apply(
            lambda serie: serie.interpolate(method="time", limit=2, limit_area="inside")
        )
    dados = dados.reset_index().sort_values("timestamp").reset_index(drop=True)
    # Para lacunas maiores que 10 min ou nas bordas, usa a mediana da própria célula.
    for coluna in SENSORES:
        mediana_celula = dados.groupby("lote_id")[coluna].transform("median")
        dados[coluna] = dados[coluna].fillna(mediana_celula).fillna(dados[coluna].median())

    for coluna in ["vibracao_mm_s", "corrente_solda_a", "pressao_gas_bar"]:
        nome = coluna.removesuffix("_mm_s").removesuffix("_a").removesuffix("_bar")
        dados[f"{nome}_media_30min"] = dados.groupby("lote_id")[coluna].transform(
            lambda s: s.rolling(6, min_periods=1).mean()
        )
        dados[f"{nome}_desvio_2h"] = dados.groupby("lote_id")[coluna].transform(
            lambda s: s.rolling(24, min_periods=2).std()
        ).fillna(0)

    dados["iem"] = (
        dados["corrente_solda_a"] * dados["vibracao_mm_s"]
        / (dados["pressao_gas_bar"] + 1)
    )


    '''
    # Razão pela média histórica da própria liga (a média é calculada sobre o conjunto disponível).
    for coluna in SENSORES:
        media_liga = dados.groupby("liga_metalica")[coluna].transform("mean").replace(0, 1)
        dados[f"{coluna}_normalizado_liga"] = dados[coluna] / media_liga
    '''
    return dados, ausencias
    

if __name__ == "__main__":
    raiz = Path(__file__).resolve().parents[1]
    dados, ausencias = carregar_e_preparar(raiz / "dataset_industrial_cibersistemas_15k.csv")
    pasta = raiz / "dados_processados"
    pasta.mkdir(exist_ok=True)
    dados.to_csv(pasta / "sensores_preparados.csv", index=False)
    ausencias.to_csv(pasta / "ausencias_por_celula_liga.csv", index=False)
    print(f"Registros processados: {len(dados)}")
    print(f"Arquivos salvos em: {pasta}")
