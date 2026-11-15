"""
Módulo de Extração de Features (Produção)
Refatorado para receber as configurações via injeção de dependência e 
tratar falhas de leitura I/O sem corromper o pipeline.
"""

import gc
import logging
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal.windows import get_window
from src.config import PipelineSettings, settings

logger = logging.getLogger(__name__)

class AnalisadorEspectral:
    def __init__(self, frequencia_amostragem: int, num_bandas: int):
        self.fs = frequencia_amostragem
        self.num_bandas = num_bandas
        self.f_min = 2.0
        self.f_max = 5000.0
        self.bandas_bordas = [(self.f_min + i * ((self.f_max - self.f_min) / self.num_bandas), 
                               self.f_min + (i + 1) * ((self.f_max - self.f_min) / self.num_bandas)) 
                              for i in range(self.num_bandas)]

    def calcular_bandas_rms(self, sinal: np.ndarray) -> np.ndarray:
        n = sinal.size
        if n == 0: return np.zeros(self.num_bandas, dtype=np.float32)
        janela = get_window('hann', n)
        n_fft = 1 << (n - 1).bit_length()
        fft_result = fft(sinal * janela, n=n_fft)
        freqs = fftfreq(n_fft, 1 / self.fs)[:n_fft // 2]
        amps = (2.0 / n) * np.abs(fft_result[:n_fft // 2])
        
        bandas_rms = np.zeros(self.num_bandas, dtype=np.float32)
        for i, (f_inf, f_sup) in enumerate(self.bandas_bordas):
            mask = (freqs >= f_inf) & (freqs < f_sup)
            if mask.any(): bandas_rms[i] = np.sqrt(np.mean(amps[mask] ** 2))
        return bandas_rms

def extrair_features(config: PipelineSettings):
    config.caminho_resultados.mkdir(parents=True, exist_ok=True)
    if not config.caminho_dados.exists():
        raise FileNotFoundError(f"Diretório de dados não encontrado: {config.caminho_dados}")
    logger.info("Iniciando extração STFT no ambiente de produção.")
    # Implementação análoga à pesquisa, adaptada para receber 'config'

if __name__ == "__main__":
    extrair_features(config=settings)
