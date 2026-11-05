"""
API RESTful assíncrona para inferência preditiva, controle em malha fechada
e inspeção óptica industrial (VulcanTech Smart Metals S.A.).
Adaptada para o processo de soldagem e indicadores OEE (Cibersistemas).
"""
from pathlib import Path
import json
import secrets
from typing import List, Literal, Optional

import cv2
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline import SENSORES, carregar_e_preparar
from visao import analisar_imagem_solda, analisar_termica_haz

RAIZ = Path(__file__).resolve().parents[1]
ARQUIVO_CSV_BRUTO = RAIZ / "dataset_industrial_cibersistemas_15k.csv"
ARQUIVO_CSV_PREPARADO = RAIZ / "dados_processados" / "sensores_preparados.csv"
ARQUIVO_MODELO = RAIZ / "modelos" / "modelo_falhas.joblib"
ARQUIVO_RELATORIO = RAIZ / "modelos" / "relatorio_modelo.json"
PASTA_WEB = RAIZ / "web"
PASTA_ASSETS = RAIZ / "assets"

app = FastAPI(
    title="VulcanTech Smart Metals - MLOps & Edge Computing API",
    description="Serviço RESTful para monitoramento preditivo de soldagem, atuação em malha fechada, OEE e inspeção visual (Porosidade e HAZ).",
    version="2.0.0"
)

SESSOES_DEMO: set[str] = set()

# Mapeamento das classes reais do dataset para códigos, descrições e ações de malha fechada
MAPA_CODIGOS_FALHA = {
    "Sem Falha": 0,
    "Porosidade na Solda": 1,
    "Desvio Termico HAZ": 2,
    "Sobrecarga de Corrente": 3,
    "Vibracao Excessiva": 4,
}

ACOES_INDUSTRIAIS = {
    "Sem Falha": "MANTER_OPERACAO",
    "Porosidade na Solda": "AJUSTAR_VAZAO_GAS_E_REDUZIR_VELOCIDADE",
    "Desvio Termico HAZ": "REDUZIR_VELOCIDADE_50%",
    "Sobrecarga de Corrente": "INTERROMPER_EMERGENCIA",
    "Vibracao Excessiva": "INTERROMPER_EMERGENCIA",
}

DETALHES_SEGURANCA = {
    "Sem Falha": "Operação de soldagem estável em regime contínuo dentro dos parâmetros nominais.",
    "Porosidade na Solda": "Risco de cavidades no cordão de solda. Verificar pressão do gás de proteção e reduzir velocidade da linha.",
    "Desvio Termico HAZ": "Sobreaquecimento na Zona Afetada pelo Calor (HAZ). Reduzir aporte térmico e velocidade para evitar trincas metalúrgicas.",
    "Sobrecarga de Corrente": "Corrente de solda crítica detectada. Interrupção imediata exigida para proteção da fonte e tocha.",
    "Vibracao Excessiva": "Instabilidade mecânica severa na célula. Interromper imediatamente para inspeção de fixação e servomotores.",
}

AcaoRecomendadaLiteral = Literal[
    "MANTER_OPERACAO",
    "AJUSTAR_VAZAO_GAS_E_REDUZIR_VELOCIDADE",
    "REDUZIR_VELOCIDADE_50%",
    "INTERROMPER_EMERGENCIA",
]


# Schemas Pydantic alinhados aos limites de pipeline.py e colunas de treinar_modelo.py
class LeituraSensor(BaseModel):
    lote_id: str = Field(
        default="LOTE_0001",
        description="Identificador do lote/célula produtiva (ex: LOTE_0001 a LOTE_0150)",
    )
    temperatura_solda_c: float = Field(..., ge=50, le=600, description="Temperatura de solda em °C")
    corrente_solda_a: float = Field(..., ge=50, le=400, description="Corrente de solda em Amperes (A)")
    tensao_solda_v: float = Field(..., ge=0, le=50, description="Tensão do arco de solda em Volts (V)")
    pressao_gas_bar: float = Field(..., ge=0, le=40, description="Pressão do gás de proteção em Bar")
    vibracao_mm_s: float = Field(..., ge=0, le=20, description="Vibração mecânica da linha em mm/s")
    velocidade_linha_m_min: float = Field(..., ge=0, le=50, description="Velocidade da linha em m/min")
    area_porosidade_px: float = Field(default=0.0, ge=0, le=100, description="Área de porosidade detectada em pixels")
    temperatura_max_haz_c: float = Field(..., ge=50, le=400, description="Temperatura máxima na zona HAZ em °C")
    tempo_planejado_min: float = Field(default=4.0, ge=0, le=4, description="Tempo planejado do ciclo em minutos")
    tempo_parada_min: float = Field(default=0.0, ge=-1, le=4, description="Tempo de parada registrado em minutos")
    pecas_produzidas: int = Field(default=20, ge=0, le=50, description="Total de peças produzidas no janela")
    pecas_boas: int = Field(default=20, ge=0, le=50, description="Total de peças aprovadas na janela")
    pecas_defeituosas: int = Field(default=0, ge=0, le=50, description="Total de peças defeituosas na janela")
    disponibilidade_pct: float = Field(default=95.0, ge=0, le=100, description="Disponibilidade operacional (%)")
    desempenho_pct: float = Field(default=95.0, ge=0, le=100, description="Desempenho operacional (%)")
    qualidade_pct: float = Field(default=98.0, ge=0, le=100, description="Qualidade operacional (%)")
    oee_pct: float = Field(default=88.4, ge=0, le=100, description="Indicador OEE global (%)")
    anomalia_isolamento: int = Field(default=0, ge=0, le=1, description="Flag de anomalia por Isolation Forest (0 ou 1)")


class DiagnosticoPredicao(BaseModel):
    lote_id: str
    tipo_falha: int
    classe_falha: str
    descricao: str
    probabilidade: float
    iem_calculado: float
    acao_recomendada: AcaoRecomendadaLiteral
    detalhes_seguranca: str


class RespostaHealth(BaseModel):
    status: str
    modelo_carregado: bool
    versao_api: str
    ambiente: str


# Montagem de diretórios estáticos
if PASTA_WEB.exists():
    app.mount("/static", StaticFiles(directory=PASTA_WEB), name="static")
if PASTA_ASSETS.exists():
    app.mount("/assets", StaticFiles(directory=PASTA_ASSETS), name="assets")


@app.middleware("http")
async def autenticacao_dashboard(request, call_next):
    caminho = request.url.path
    token = request.cookies.get("vulcantech_demo")
    autenticado = token in SESSOES_DEMO

    rotas_publicas = (
        "/demo-login",
        "/health",
        "/predict",
        "/predict-batch",
        "/docs",
        "/openapi.json",
        "/inspect",
    )
    if any(caminho.startswith(r) for r in rotas_publicas):
        return await call_next(request)

    if caminho.startswith(("/static/", "/assets/")):
        resp = await call_next(request)
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    if caminho == "/login":
        if autenticado:
            return RedirectResponse("/dashboard", status_code=303)
        return await call_next(request)

    if not autenticado:
        if caminho.startswith("/api/"):
            return JSONResponse({"detail": "Autenticação requerida."}, status_code=401)
        return RedirectResponse("/login", status_code=303)

    return await call_next(request)


@app.post("/demo-login")
async def demo_login(email: str = Form(...), senha: str = Form(...)):
    if not email.strip() or not senha.strip():
        raise HTTPException(status_code=400, detail="Credenciais incompletas.")
    token = secrets.token_urlsafe(32)
    SESSOES_DEMO.add(token)
    resposta = JSONResponse({"status": "autenticado"})
    resposta.set_cookie(
        "vulcantech_demo",
        token,
        httponly=True,
        samesite="lax",
        max_age=28800,
        path="/",
    )
    return resposta


# Rotas de Interface Web
@app.get("/")
async def rota_raiz():
    return FileResponse(PASTA_WEB / "index.html")


@app.get("/login")
async def rota_login():
    return FileResponse(PASTA_WEB / "login.html")


@app.get("/dashboard")
async def rota_dashboard():
    return FileResponse(PASTA_WEB / "dashboard.html")


@app.get("/sensors")
async def rota_sensors():
    return FileResponse(PASTA_WEB / "sensors.html")


@app.get("/inspection")
async def rota_inspection():
    return FileResponse(PASTA_WEB / "inspection.html")


# Rotas de Dados para o Dashboard
@app.get("/api/dashboard-data")
async def api_dashboard_data():
    if ARQUIVO_CSV_PREPARADO.exists():
        df = pd.read_csv(ARQUIVO_CSV_PREPARADO, parse_dates=["timestamp"])
    else:
        df, _ = carregar_e_preparar(ARQUIVO_CSV_BRUTO)

    orig = pd.read_csv(ARQUIVO_CSV_BRUTO)
    taxa_ausencia = float(
        100 * orig[SENSORES].isna().sum().sum() / (len(orig) * len(SENSORES))
    )

    colunas_disponiveis = [
        "timestamp",
        "lote_id",
        *SENSORES,
        "pecas_produzidas",
        "pecas_boas",
        "pecas_defeituosas",
        "disponibilidade_pct",
        "desempenho_pct",
        "qualidade_pct",
        "oee_pct",
        "anomalia_isolamento",
        "iem",
        "classe_falha",
    ]
    colunas = [c for c in colunas_disponiveis if c in df.columns]
    df_exp = df[colunas].copy()
    df_exp["timestamp"] = df_exp["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    return {
        "rows": df_exp.to_dict(orient="records"),
        "missing_rate": round(taxa_ausencia, 2),
    }


@app.get("/api/model-report")
async def api_model_report():
    if not ARQUIVO_RELATORIO.exists():
        return {
            "disponivel": False,
            "mensagem": "Execute 'python treinar_modelo.py' para compilar o modelo.",
        }
    return {
        "disponivel": True,
        **json.loads(ARQUIVO_RELATORIO.read_text(encoding="utf-8")),
    }


# Endpoints MLOps e Malha Fechada
@app.get("/health", response_model=RespostaHealth)
async def health():
    return {
        "status": "online",
        "modelo_carregado": ARQUIVO_MODELO.exists(),
        "versao_api": "2.0.0",
        "ambiente": "Edge Industrial Node",
    }


def _processar_vetor_predicao(
    leitura: LeituraSensor, colunas_esperadas: list, encoder_celula=None
) -> tuple[pd.DataFrame, float]:
    dados = leitura.model_dump()

    # Índice de Estresse Mecânico/Solda (mesma equação do pipeline.py)
    iem = (dados["corrente_solda_a"] * dados["vibracao_mm_s"]) / (
        dados["pressao_gas_bar"] + 1.0
    )

    # Codificação segura do lote_id usando o LabelEncoder treinado
    celula_codigo = 0
    if encoder_celula is not None and hasattr(encoder_celula, "classes_"):
        if dados["lote_id"] in encoder_celula.classes_:
            celula_codigo = int(encoder_celula.transform([dados["lote_id"]])[0])

    linha = {
        **dados,
        "vibracao_media_30min": dados["vibracao_mm_s"],
        "vibracao_desvio_2h": 0.0,
        "corrente_solda_media_30min": dados["corrente_solda_a"],
        "corrente_solda_desvio_2h": 0.0,
        "pressao_gas_media_30min": dados["pressao_gas_bar"],
        "pressao_gas_desvio_2h": 0.0,
        "iem": iem,
        "celula_codigo": celula_codigo,
    }

    vetor_alinhado = {c: linha.get(c, 0.0) for c in colunas_esperadas}
    return pd.DataFrame([vetor_alinhado]), iem


def _executar_inferencia_unica(leitura: LeituraSensor, pacote: dict) -> dict:
    modelo = pacote["modelo"]
    colunas = pacote["colunas"]
    encoder_celula = pacote.get("encoder_celula")

    df_entrada, iem = _processar_vetor_predicao(leitura, colunas, encoder_celula)
    predicao_bruta = modelo.predict(df_entrada)[0]
    classe_str = str(predicao_bruta)

    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(df_entrada)[0]
        probabilidade_max = float(np.max(probabilidades))
    else:
        probabilidade_max = 1.0

    codigo_falha = MAPA_CODIGOS_FALHA.get(classe_str, -1)
    acao = ACOES_INDUSTRIAIS.get(classe_str, "MANTER_OPERACAO")
    detalhe = DETALHES_SEGURANCA.get(classe_str, "Sem intervenção necessária.")

    return {
        "lote_id": leitura.lote_id,
        "tipo_falha": codigo_falha,
        "classe_falha": classe_str,
        "descricao": classe_str,
        "probabilidade": round(probabilidade_max, 4),
        "iem_calculado": round(float(iem), 4),
        "acao_recomendada": acao,
        "detalhes_seguranca": detalhe,
    }


@app.post("/predict", response_model=DiagnosticoPredicao)
async def predict(leitura: LeituraSensor):
    if not ARQUIVO_MODELO.exists():
        raise HTTPException(
            status_code=503,
            detail="Modelo preditivo não treinado. Execute treinar_modelo.py",
        )

    pacote = joblib.load(ARQUIVO_MODELO)
    return _executar_inferencia_unica(leitura, pacote)


@app.post("/predict-batch", response_model=List[DiagnosticoPredicao])
async def predict_batch(leituras: List[LeituraSensor]):
    if not ARQUIVO_MODELO.exists():
        raise HTTPException(
            status_code=503, detail="Modelo preditivo não disponível."
        )

    pacote = joblib.load(ARQUIVO_MODELO)
    return [_executar_inferencia_unica(leitura, pacote) for leitura in leituras]


# Rotas de Visão Computacional (integração direta com visao.py)
async def _extrair_imagem(upload: UploadFile, roi_x, roi_y, roi_w, roi_h):
    conteudo = await upload.read()
    matriz = np.frombuffer(conteudo, dtype=np.uint8)
    imagem = cv2.imdecode(matriz, cv2.IMREAD_COLOR)
    if imagem is None:
        raise HTTPException(
            status_code=400, detail="Formato de imagem inválido (use JPEG ou PNG)."
        )

    roi_coords = (roi_x, roi_y, roi_w, roi_h)
    if any(v is not None for v in roi_coords):
        if not all(v is not None for v in roi_coords):
            raise HTTPException(
                status_code=422, detail="Parâmetros de ROI incompletos."
            )
        roi = tuple(roi_coords)
    else:
        roi = None
    return imagem, roi


@app.post("/inspect/weld")
async def inspect_weld(
    imagem: UploadFile = File(...),
    roi_x: Optional[int] = Form(None),
    roi_y: Optional[int] = Form(None),
    roi_w: Optional[int] = Form(None),
    roi_h: Optional[int] = Form(None),
):
    img, roi = await _extrair_imagem(imagem, roi_x, roi_y, roi_w, roi_h)
    try:
        return analisar_imagem_solda(img, roi)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err


@app.post("/inspect/thermal")
async def inspect_thermal(
    imagem: UploadFile = File(...),
    roi_x: Optional[int] = Form(None),
    roi_y: Optional[int] = Form(None),
    roi_w: Optional[int] = Form(None),
    roi_h: Optional[int] = Form(None),
):
    img, roi = await _extrair_imagem(imagem, roi_x, roi_y, roi_w, roi_h)
    try:
        return analisar_termica_haz(img, roi)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err