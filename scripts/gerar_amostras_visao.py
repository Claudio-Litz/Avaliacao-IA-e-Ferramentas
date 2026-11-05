"""Gera duas imagens sintéticas para experimentar a página de inspeção."""
from pathlib import Path

import cv2
import numpy as np


DESTINO = Path(__file__).resolve().parents[1] / "assets" / "amostras_visao"
DESTINO.mkdir(parents=True, exist_ok=True)
ALTURA, LARGURA = 540, 960
ALEATORIO = np.random.default_rng(17)


def criar_cordao():
    # Placa metálica cinza com textura leve.
    imagem = np.zeros((ALTURA, LARGURA, 3), dtype=np.uint8)
    for y in range(ALTURA):
        tom = int(78 + 20 * y / ALTURA)
        imagem[y, :] = (tom, tom + 4, tom + 7)
    ruido = ALEATORIO.normal(0, 1, imagem.shape).astype(np.int16)
    imagem = np.clip(imagem.astype(np.int16) + ruido, 0, 255).astype(np.uint8)

    # Bordas da chapa e sombra do cordão.
    cv2.line(imagem, (0, 260), (LARGURA, 260), (45, 53, 58), 9)
    cv2.line(imagem, (0, 272), (LARGURA, 272), (165, 171, 170), 3)
    eixo = np.array([(x, 300 + int(10 * np.sin(x / 85))) for x in range(0, LARGURA, 8)], dtype=np.int32)
    cv2.polylines(imagem, [eixo], False, (40, 47, 50), 37, cv2.LINE_AA)
    cv2.polylines(imagem, [eixo], False, (150, 160, 158), 26, cv2.LINE_AA)
    cv2.polylines(imagem, [eixo], False, (194, 199, 192), 8, cv2.LINE_AA)
    cv2.polylines(imagem, [eixo], False, (222, 178, 121), 2, cv2.LINE_AA)

    # Poros desenhados como pequenas cavidades escuras com borda clara.
    for x, y, raio in [(290, 294, 8), (468, 307, 7), (645, 296, 9), (760, 303, 6)]:
        cv2.circle(imagem, (x, y), raio + 2, (205, 203, 190), -1, cv2.LINE_AA)
        cv2.circle(imagem, (x, y), raio, (32, 37, 40), -1, cv2.LINE_AA)
        cv2.ellipse(imagem, (x - 2, y - 3), (max(2, raio // 3), 2), 0, 0, 180, (227, 222, 204), 1, cv2.LINE_AA)

    # Pequenos reflexos quentes dão contraste ao ensaio visual.
    for x, y in [(155, 228), (378, 226), (850, 231)]:
        cv2.line(imagem, (x, y), (x + 18, y - 32), (54, 142, 224), 3, cv2.LINE_AA)
        cv2.circle(imagem, (x + 18, y - 32), 4, (55, 181, 250), -1, cv2.LINE_AA)
    cv2.putText(imagem, "AMOSTRA SINTETICA - CORDÃO COM DESCONTINUIDADES", (28, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 229, 224), 2, cv2.LINE_AA)
    cv2.imwrite(str(DESTINO / "cordao_solda_com_porosos.png"), imagem)


def criar_termica():
    yy, xx = np.mgrid[0:ALTURA, 0:LARGURA]
    hsv = np.zeros((ALTURA, LARGURA, 3), dtype=np.uint8)
    hsv[:, :] = (108, 190, 115)  # Fundo frio em tons azulados.
    centro_y = 275 + 14 * np.sin(xx / 100)
    distancia = np.abs(yy - centro_y)
    intensidade = np.exp(-((distancia / 85) ** 2))
    zona_quente = intensidade > 0.18
    hsv[:, :, 0][zona_quente] = (24 - 12 * intensidade[zona_quente]).astype(np.uint8)
    hsv[:, :, 1][zona_quente] = 235
    hsv[:, :, 2][zona_quente] = (120 + 130 * intensidade[zona_quente]).astype(np.uint8)

    # Uma área concentrada representa um ponto de sobreaquecimento.
    mancha = np.exp(-(((xx - 675) / 95) ** 2 + ((yy - 273) / 55) ** 2))
    critico = mancha > 0.12
    hsv[:, :, 0][critico] = 14
    hsv[:, :, 1][critico] = 250
    hsv[:, :, 2][critico] = np.maximum(hsv[:, :, 2][critico], (145 + 110 * mancha[critico]).astype(np.uint8))

    imagem = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.line(imagem, (0, 350), (LARGURA, 350), (65, 70, 76), 5, cv2.LINE_AA)
    cv2.putText(imagem, "AMOSTRA TERMICA SINTETICA - REGIAO QUENTE", (28, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 235), 2, cv2.LINE_AA)
    cv2.imwrite(str(DESTINO / "termica_com_sobreaquecimento.png"), imagem)


if __name__ == "__main__":
    criar_cordao()
    criar_termica()
    print(f"Imagens salvas em: {DESTINO}")
