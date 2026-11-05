"""Triagem visual experimental de cordões de solda usando OpenCV."""
import base64

import cv2
import numpy as np


def _preparar_recorte(imagem: np.ndarray, roi: tuple[int, int, int, int] | None) -> tuple[np.ndarray, int, int]:
    """Isola a região selecionada, limita o tamanho e devolve a matriz BGR recortada."""
    altura, largura = imagem.shape[:2]
    if roi:
        x, y, w, h = roi
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(largura, x + w), min(altura, y + h)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("A área selecionada está fora da imagem.")
        recorte = imagem[y1:y2, x1:x2].copy()
    else:
        recorte = imagem.copy()

    h, w = recorte.shape[:2]
    if max(h, w) > 1100:
        escala = 1100 / max(h, w)
        recorte = cv2.resize(recorte, (int(w * escala), int(h * escala)))
        h, w = recorte.shape[:2]
    return recorte, h, w


def analisar_imagem_solda(imagem: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> dict:
    """Gera um mapa visual de contraste e uma triagem heurística da região selecionada.

    O mapa mostra variações locais da imagem; não representa temperatura real.
    """
    recorte, h, w = _preparar_recorte(imagem, roi)

    cinza = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(cinza, (5, 5), 0)
    bordas = cv2.Canny(suavizada, 50, 140)
    contornos, _ = cv2.findContours(bordas, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Estimativa simples de cavidades pequenas e arredondadas. Bordas próximas
    # do mesmo ponto são agrupadas para reduzir a contagem duplicada.
    poros = []
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        if not 20 <= area <= 400 or perimetro == 0:
            continue
        circularidade = 4 * np.pi * area / (perimetro * perimetro)
        momentos = cv2.moments(contorno)
        if circularidade < 0.55 or momentos["m00"] == 0:
            continue
        centro = (momentos["m10"] / momentos["m00"], momentos["m01"] / momentos["m00"])
        if not any(np.hypot(centro[0] - px, centro[1] - py) < 10 for px, py, _ in poros):
            poros.append((*centro, area))

    total = h * w
    area_porosa = 100 * sum(item[2] for item in poros) / total
    densidade_bordas = 100 * cv2.countNonZero(bordas) / total
    nitidez = float(cv2.Laplacian(cinza, cv2.CV_64F).var())

    # Mapa falso-cor feito a partir de diferenças locais, suavizado para reduzir
    # o ruído da superfície. Só os pontos de maior contraste recebem a cor.
    fundo_local = cv2.GaussianBlur(cinza, (0, 0), 9)
    detalhe = cv2.absdiff(cinza, fundo_local)
    detalhe = cv2.medianBlur(detalhe, 7)
    detalhe = cv2.GaussianBlur(detalhe, (0, 0), 4)
    corte = float(np.percentile(detalhe, 88))
    limite = max(float(np.percentile(detalhe, 99.5)), corte + 1)
    intensidade = np.clip((detalhe.astype(np.float32) - corte) * 255 / (limite - corte), 0, 255).astype(np.uint8)
    mapa = cv2.applyColorMap(intensidade, cv2.COLORMAP_INFERNO)
    mascara = intensidade > 24
    sobreposicao = recorte.copy()
    misturada = cv2.addWeighted(recorte, 0.58, mapa, 0.42, 0)
    sobreposicao[mascara] = misturada[mascara]

    motivos = []
    if min(h, w) < 80 or nitidez < 5:
        avaliacao = "Imagem inconclusiva"
        motivos.append("A região está pequena ou sem nitidez suficiente para a triagem.")
    else:
        if len(poros) >= 5:
            motivos.append("Foram encontrados vários contornos pequenos e arredondados que podem indicar poros.")
        if area_porosa >= 0.5:
            motivos.append("A área estimada desses contornos passou do limite inicial do filtro.")
        if motivos:
            avaliacao = "Revisar a solda"
        else:
            avaliacao = "Sem alerta visual pelo filtro"
            motivos.append("O filtro não encontrou um número elevado de pequenos contornos arredondados.")

    sucesso, buffer = cv2.imencode(".png", sobreposicao)
    if not sucesso:
        raise ValueError("Não foi possível gerar a imagem de resultado.")

    return {
        "tipo": "solda",
        "avaliacao": avaliacao,
        "motivos": motivos,
        "possiveis_poros": len(poros),
        "area_porosa_estimada_percentual": round(area_porosa, 3),
        "densidade_bordas_percentual": round(densidade_bordas, 2),
        "nitidez": round(nitidez, 1),
        "mapa_visual_base64": base64.b64encode(buffer).decode("ascii"),
        "observacao": "Triagem heurística de imagem RGB; as cores mostram contraste visual, não temperatura. Não substitui inspeção ou critério técnico de aceitação.",
    }


# Limiares iniciais para análise térmica HSV (H em 0-179 no OpenCV).
# Matiz quente = vermelho/laranja/amarelo; zona crítica = saturada e bem clara.
MATIZ_QUENTE_MAX = 30
SATURACAO_MIN = 60
VALOR_MIN = 60
VALOR_CRITICO = 220
AREA_MINIMA_CONTORNO = 0.003  # fração da região (0,3%)
LIMIAR_CRITICA_PERCENTUAL = 8.0
LIMIAR_QUENTE_PERCENTUAL = 2.0


def analisar_termica_haz(imagem: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> dict:
    """Analisa mapa térmico da zona afetada pelo calor (HAZ) em espaço HSV.

    Converte a imagem para HSV, isola os matizes quentes (limiarização) e destaca
    as zonas críticas com maior valor/brilho dentro da região quente.
    Não mede temperatura absoluta: os limites são parâmetros iniciais não calibrados.
    """
    recorte, h, w = _preparar_recorte(imagem, roi)
    if min(h, w) < 40:
        raise ValueError("A área selecionada é pequena demais para análise.")

    hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)
    matiz, saturacao, valor = cv2.split(hsv)

    quente = (matiz <= MATIZ_QUENTE_MAX) & (saturacao >= SATURACAO_MIN) & (valor >= VALOR_MIN)
    total = quente.size
    pct_quente = 100 * int(quente.sum()) / total

    critica = quente & (valor >= VALOR_CRITICO)
    mascara = critica.astype(np.uint8) * 255
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    pct_critica = 100 * int((mascara > 0).sum()) / total

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zonas = [c for c in contornos if cv2.contourArea(c) >= AREA_MINIMA_CONTORNO * total]
    zonas_ordenadas = sorted(zonas, key=cv2.contourArea, reverse=True)
    maior_zona_pct = (100 * cv2.contourArea(zonas_ordenadas[0]) / total) if zonas_ordenadas else 0.0

    motivos = []
    if pct_quente < LIMIAR_QUENTE_PERCENTUAL:
        avaliacao = "Imagem inconclusiva"
        motivos.append("Pouca variação de matiz quente na região; a imagem não parece um mapa térmico (ou está sem cor suficiente).")
    elif pct_critica >= LIMIAR_CRITICA_PERCENTUAL:
        avaliacao = "Revisar a solda"
        motivos.append(f"Zona crítica de sobreaquecimento cobre {pct_critica:.1f}% da região (limiar inicial: {LIMIAR_CRITICA_PERCENTUAL:.0f}%).")
        if zonas_ordenadas:
            motivos.append(f"Foram isoladas {len(zonas_ordenadas)} zona(s) crítica(s) por limiarização HSV; a maior cobre {maior_zona_pct:.1f}% da região.")
    else:
        avaliacao = "Sem ponto quente crítico pelo filtro"
        motivos.append(f"Região quente presente ({pct_quente:.1f}%), mas nenhuma zona crítica acima do limiar inicial ({LIMIAR_CRITICA_PERCENTUAL:.0f}%).")

    # Sobreposição: quente translúcida em laranja; crítica em vermelho; contornos em magenta.
    sobreposicao = recorte.copy()
    veu = recorte.copy()
    veu[quente] = (0, 140, 255)
    sobreposicao = cv2.addWeighted(sobreposicao, 0.6, veu, 0.4, 0)
    crit_bool = mascara > 0
    veu_critico = sobreposicao.copy()
    veu_critico[crit_bool] = (0, 0, 255)
    sobreposicao = cv2.addWeighted(sobreposicao, 0.45, veu_critico, 0.55, 0)
    cv2.drawContours(sobreposicao, zonas_ordenadas, -1, (255, 0, 255), 2)

    sucesso, buffer = cv2.imencode(".png", sobreposicao)
    if not sucesso:
        raise ValueError("Não foi possível gerar a imagem de resultado.")

    return {
        "tipo": "termica",
        "avaliacao": avaliacao,
        "motivos": motivos,
        "regiao_quente_percentual": round(pct_quente, 2),
        "zona_critica_percentual": round(pct_critica, 2),
        "zonas_criticas": len(zonas_ordenadas),
        "maior_zona_percentual": round(maior_zona_pct, 2),
        "mapa_visual_base64": base64.b64encode(buffer).decode("ascii"),
        "observacao": "Análise HSV de mapa térmico (matiz/saturação/valor) com limiarização; não mede temperatura absoluta e não substitui pirômetro ou inspeção por especialista.",
    }
