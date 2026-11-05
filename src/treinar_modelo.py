"""
Treinamento, avaliação comparativa e exportação de modelos preditivos multiclasse.
Atende aos requisitos de divisão temporal 70/30, balanceamento (SMOTE/Class Weights)
e cálculo de impacto no indicador OEE da VulcanTech Smart Metals S.A.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

from pipeline import carregar_e_preparar

RAIZ = Path(__file__).resolve().parents[1]

COLUNAS_BASE = [
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
    "pecas_produzidas",
    "pecas_boas",
    "pecas_defeituosas",
    "disponibilidade_pct",
    "desempenho_pct",
    "qualidade_pct",
    "oee_pct",
    "anomalia_isolamento",
    "vibracao_media_30min",
    "vibracao_desvio_2h",
    "corrente_solda_media_30min",
    "corrente_solda_desvio_2h",
    "pressao_gas_media_30min",
    "pressao_gas_desvio_2h",
    "iem",
]

def main():
    print("=" * 60)
    print("Iniciando Pipeline de Modelagem Preditiva - Avaliacao")
    print("=" * 60)

    arquivo_preparado = RAIZ / "dados_processados" / "sensores_preparados.csv"
    if arquivo_preparado.exists():
        dados = pd.read_csv(arquivo_preparado, parse_dates=["timestamp"])
        print(f"Utilizando base pré-processada: {arquivo_preparado}")
    else:
        csv_bruto = RAIZ / "dataset_industrial_cibersistemas_15k.csv"
        dados, _ = carregar_e_preparar(csv_bruto)
        pasta_proc = RAIZ / "dados_processados"
        pasta_proc.mkdir(exist_ok=True)
        dados.to_csv(arquivo_preparado, index=False)
        print(f"Base processada gerada em: {arquivo_preparado}")

    # Encoders para variáveis categóricas
    encoder_celula = LabelEncoder()
    dados["celula_codigo"] = encoder_celula.fit_transform(dados["lote_id"])

    features = COLUNAS_BASE + ["celula_codigo"]
    dados = dados.sort_values("timestamp").reset_index(drop=True)

    # Divisão temporal estrita (70% treino, 30% teste)
    corte = int(len(dados) * 0.70)
    treino, teste = dados.iloc[:corte], dados.iloc[corte:]

    X_treino = treino[features].copy()
    y_treino = treino["classe_falha"].copy()
    X_teste = teste[features].copy()
    y_teste = teste["classe_falha"].copy()

    print(f"Tamanho do conjunto de Treino: {len(X_treino)} registros")
    print(f"Tamanho do conjunto de Teste:  {len(X_teste)} registros")
    print("Distribuição de classes no treino original:\n", y_treino.value_counts())

    # Aplicação de balanceamento via SMOTE
    if HAS_SMOTE:
        print("\nAplicando SMOTE para balanceamento das classes raras...")
        smote = SMOTE(random_state=42)
        X_treino_bal, y_treino_bal = smote.fit_resample(X_treino, y_treino)
    else:
        print("\nImbalanced-Learn não detectado. Usando balanceamento por pesos de classe.")
        X_treino_bal, y_treino_bal = X_treino, y_treino

    candidatos = {
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=150, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Extra Trees (Balanced)": ExtraTreesClassifier(
            n_estimators=150, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
        )
    }

    melhor_recall_falhas = -1.0
    melhor_nome = None
    melhor_modelo = None
    melhor_predicao = None
    relatorios_comparativos = {}

    # Usamos nome_modelo para evitar colisão com a variável 'nome' interna das classes
    for nome_modelo, modelo in candidatos.items():
        print(f"\nTreinando algoritmo: {nome_modelo}...")
        modelo.fit(X_treino_bal, y_treino_bal)
        y_pred = modelo.predict(X_teste)

        relatorio = classification_report(y_teste, y_pred, output_dict=True, zero_division=0)
        
        classes_todas = ["Sem Falha", "Porosidade na Solda", "Desvio Termico HAZ", "Sobrecarga de Corrente", "Vibracao Excessiva"]
        matriz = confusion_matrix(y_teste, y_pred, labels=classes_todas)

        # Recall médio das classes críticas
        nomes_falhas = ["Porosidade na Solda", "Desvio Termico HAZ", "Sobrecarga de Corrente", "Vibracao Excessiva"]
        recalls_falhas = [relatorio[nome]["recall"] for nome in nomes_falhas if nome in relatorio]

        recall_medio_falhas = float(np.mean(recalls_falhas)) if recalls_falhas else 0.0

        relatorios_comparativos[nome_modelo] = {
            "recall_medio_falhas": recall_medio_falhas,
            "recall_por_classe": {nome: float(relatorio[nome]["recall"]) for nome in classes_todas if nome in relatorio},
            "matriz_confusao": matriz.tolist()
        }

        print("Recalls por clase:")
        for nome in classes_todas:
            if nome in relatorio:
                print(f"  - {nome}: {relatorio[nome]['recall'] * 100:.2f}%")
                
        if recall_medio_falhas > melhor_recall_falhas:
            melhor_recall_falhas = recall_medio_falhas
            melhor_nome = nome_modelo
            melhor_modelo = modelo
            melhor_predicao = y_pred

    print("\n" + "=" * 60)
    print(f"Modelo Campeão Selecionado: {melhor_nome}")
    print(f"Recall Médio Final nas Falhas: {melhor_recall_falhas * 100:.2f}%")
    print("=" * 60)

    # Cálculo preditivo de impacto no OEE usando as strings reais das falhas
    predicoes_serie = pd.Series(melhor_predicao, index=teste.index)
    
    # Penalização de Disponibilidade por risco iminente de quebra/parada (Porosidade na Solda e Sobrecarga de Corrente)
    disponibilidade = 1.0 - float(predicoes_serie.isin(["Porosidade na Solda", "Sobrecarga de Corrente"]).mean())
    # Penalização de Qualidade por divergência de corte/temperatura térmica (Desvio Termico HAZ)
    qualidade = 1.0 - float((predicoes_serie == "Desvio Termico HAZ").mean())
    
    performance = 0.95
    oee_estimado = disponibilidade * qualidade * performance

    print(f"Indicadores de OEE Estimados:")
    print(f"  - Disponibilidade Preditiva: {disponibilidade * 100:.2f}%")
    print(f"  - Qualidade Preditiva:       {qualidade * 100:.2f}%")
    print(f"  - Performance Fixada:       {performance * 100:.2f}%")
    print(f"  - OEE Preditivo Global:      {oee_estimado * 100:.2f}%")

    # Extração de Feature Importances
    if hasattr(melhor_modelo, "feature_importances_"):
        importancias = [
            {"nome": feat, "importancia": float(imp)}
            for feat, imp in pd.Series(melhor_modelo.feature_importances_, index=features)
            .sort_values(ascending=False).head(10).items()
        ]
    else:
        importancias = []

    # Exportação dos artefatos
    pasta_modelos = RAIZ / "modelos"
    pasta_modelos.mkdir(exist_ok=True)
    caminho_joblib = pasta_modelos / "modelo_falhas.joblib"
    caminho_json = pasta_modelos / "relatorio_modelo.json"

    pacote_modelo = {
        "modelo": melhor_modelo,
        "colunas": features,
        "encoder_celula": encoder_celula,
    }
    joblib.dump(pacote_modelo, caminho_joblib)
    print(f"\nModelo salvo com sucesso em: {caminho_joblib}")

    relatorio_json = {
        "modelo": melhor_nome,
        "recall_medio_falhas": melhor_recall_falhas,
        "recall_por_classe": relatorios_comparativos[melhor_nome]["recall_por_classe"],
        "matriz_confusao": relatorios_comparativos[melhor_nome]["matriz_confusao"],
        "indicadores_oee": {
            "disponibilidade_preditiva": disponibilidade,
            "qualidade_preditiva": qualidade,
            "performance_fixada": performance,
            "oee_preditivo_global": oee_estimado
        },
        "feature_importances": importancias
    }

    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(relatorio_json, f, indent=4, ensure_ascii=False)
    print(f"Relatório de métricas exportado em: {caminho_json}")

if __name__ == "__main__":
    main()
