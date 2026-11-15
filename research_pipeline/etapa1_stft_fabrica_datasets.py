"""
Pesquisa Original (PRH-ANP): Fábrica de Datasets HDF5
Objetivo: Converter séries temporais 1D em tensores 3D (Tempo x Banda x Canal) via STFT.
Decisão Metodológica: A normalização min-max não é calculada nesta etapa para prevenir 
vazamento de dados (data leakage) entre treino e teste. Os valores RMS são salvos brutos.
"""

import gc
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import h5py
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal.windows import get_window

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s', datefmt='%H:%M:%S')

CONFIG = {
    "FREQUENCIA_AMOSTRAGEM": 25600,
    "TIPO_JANELA": 'hann',
    "FREQUENCIA_MINIMA_ANALISE": 2.0,
    "FREQUENCIA_MAXIMA_ANALISE": 5000,
    "CAMINHO_DADOS": Path(r"C:\Projeto_compressores\Dados_amaciamento"),
    "CAMINHO_RESULTADOS": Path(r"C:\Projeto_compressores\resultado_stft_linear"),
}

SINAIS = ['current', 'vibrationLateral', 'vibrationLongitudinal']

ENSAIOS_SELECIONADOS = [
    'A2_N_2019_07_09', 'A2_A_2019_08_08', 'A2_A_2019_08_28',
    'A3_N_2019_12_04', 'A3_A_2019_12_09', 'A3_A_2019_12_11',
    'A4_N_2019_12_16', 'A4_A_2019_12_19', 'A4_A_2020_01_06',
    'A5_N_2020_01_22', 'A5_A_2020_01_27', 'A5_A_2020_01_28',
    'B5_N_2020_10_16', 'B5_A_2020_10_22', 'B5_A_2020_10_27',
    'B7_N_2021_02_05', 'B7_A_2021_02_08', 'B7_A_2021_02_15',
    'B10_N_2021_03_22', 'B10_A_2021_03_25', 'B10_A_2021_03_30',
    'B11_N_2021_04_05', 'B11_A_2021_04_08', 'B11_A_2021_04_22',
    'B12_N_2021_04_27', 'B12_A_2021_04_30', 'B12_A_2021_05_04',
    'B15_N_2021_05_31', 'B15_A_2021_06_09', 'B15_A_2021_06_15',
]

def normalizar_nome_pasta(nome: str) -> str:
    return nome.replace('_N_', '_').replace('_A_', '_')

MAPA_PASTA_ENSAIO = {normalizar_nome_pasta(e): e for e in ENSAIOS_SELECIONADOS}

class CarregadorSinalCsv:
    @staticmethod
    def carregar(caminho: Path) -> Dict[str, np.ndarray]:
        df = pd.read_csv(caminho, sep=',', skipinitialspace=True, encoding='utf-8-sig', header=0)
        df.columns = df.columns.str.strip()
        colunas = ['current', 'vibrationLateral', 'vibrationLongitudinal', 'time']
        df = df[colunas].apply(pd.to_numeric, errors='coerce').dropna()
        return {col: df[col].to_numpy(dtype=np.float32) for col in colunas}

class AnalisadorEspectral:
    def __init__(self, config: dict, num_bandas: int):
        self.fs = config['FREQUENCIA_AMOSTRAGEM']
        self.janela_tipo = config['TIPO_JANELA']
        self.f_min = config['FREQUENCIA_MINIMA_ANALISE']
        self.f_max = config['FREQUENCIA_MAXIMA_ANALISE']
        self.num_bandas = num_bandas
        self.bandas_bordas = [(self.f_min + i * ((self.f_max - self.f_min) / self.num_bandas), 
                               self.f_min + (i + 1) * ((self.f_max - self.f_min) / self.num_bandas)) 
                              for i in range(self.num_bandas)]

    def calcular_bandas_rms(self, sinal: np.ndarray) -> np.ndarray:
        n = sinal.size
        if n == 0: return np.zeros(self.num_bandas, dtype=np.float32)
        janela = get_window(self.janela_tipo, n)
        n_fft = 1 << (n - 1).bit_length()
        fft_result = fft(sinal * janela, n=n_fft)
        freqs = fftfreq(n_fft, 1 / self.fs)[:n_fft // 2]
        amps = (2.0 / n) * np.abs(fft_result[:n_fft // 2])
        
        bandas_rms = np.zeros(self.num_bandas, dtype=np.float32)
        for i, (f_inf, f_sup) in enumerate(self.bandas_bordas):
            mask = (freqs >= f_inf) & (freqs < f_sup)
            if mask.any(): bandas_rms[i] = np.sqrt(np.mean(amps[mask] ** 2))
        return bandas_rms

def criar_janelas_deslizantes(matriz: np.ndarray, tamanho_janela: int, passo: int = 1) -> np.ndarray:
    n, nb = matriz.shape
    if n < tamanho_janela: return np.empty((0, tamanho_janela, nb), dtype=np.float32)
    n_janelas = (n - tamanho_janela) // passo + 1
    strides = (matriz.strides[0] * passo, matriz.strides[0], matriz.strides[1])
    return np.lib.stride_tricks.as_strided(matriz, shape=(n_janelas, tamanho_janela, nb), strides=strides).copy().astype(np.float32)

def arquivo_hdf5_valido(caminho: Path) -> bool:
    if not caminho.exists(): return False
    try:
        with h5py.File(caminho, 'r') as f: return len(f.keys()) > 0
    except OSError: return False

def main():
    CONFIG['CAMINHO_RESULTADOS'].mkdir(parents=True, exist_ok=True)
    pastas = sorted([d for d in CONFIG['CAMINHO_DADOS'].iterdir() if d.is_dir()])
    lista_num_bandas = [16, 32, 64, 128, 256]
    lista_tam_janela = [16, 32, 64, 128]

    for nb in lista_num_bandas:
        arquivos_deste_nb = [CONFIG['CAMINHO_RESULTADOS'] / f"dataset_bandas{nb}_janela{tj}.h5" for tj in lista_tam_janela]
        if all(arquivo_hdf5_valido(p) for p in arquivos_deste_nb):
            continue

        analisador = AnalisadorEspectral(CONFIG, nb)
        dados_ensaios = {}

        for pasta in pastas:
            if normalizar_nome_pasta(pasta.name) not in MAPA_PASTA_ENSAIO: continue
            arquivos = sorted([f for f in pasta.iterdir() if f.is_file() and f.stem.isdigit()], key=lambda p: int(p.stem))
            acumuladores = {s: [] for s in SINAIS}
            for arq in arquivos:
                try:
                    dados = CarregadorSinalCsv.carregar(arq)
                    for s in SINAIS: acumuladores[s].append(analisador.calcular_bandas_rms(dados[s]))
                except Exception: continue
            if len(set([len(acumuladores[s]) for s in SINAIS])) > 1 or len(acumuladores[SINAIS[0]]) == 0: continue
            
            dados_ensaios[pasta.name] = np.stack([np.array(acumuladores[s], dtype=np.float32) for s in SINAIS], axis=-1)

        for tj in lista_tam_janela:
            caminho_hdf5 = CONFIG['CAMINHO_RESULTADOS'] / f"dataset_bandas{nb}_janela{tj}.h5"
            if arquivo_hdf5_valido(caminho_hdf5): continue
            
            with h5py.File(caminho_hdf5, 'w') as f:
                for nome_ensaio, tensor in dados_ensaios.items():
                    if tensor.shape[0] < tj: continue
                    canais_janelados = [criar_janelas_deslizantes(tensor[:, :, ch], tj, passo=1) for ch in range(3)]
                    f.create_dataset(nome_ensaio, data=np.stack(canais_janelados, axis=-1), dtype='f4', compression='gzip', compression_opts=4)

if __name__ == "__main__":
    main()
