"""
Módulo de Carregamento e Rotulagem Temporal (Produção)
Aplica a regra de negócio para rotulagem das amostras e descarta os 
instantes temporais onde o estado tribológico é transiente (zona cinzenta).
"""

from pathlib import Path
from typing import Tuple
import h5py
import numpy as np

# Mapeamento do status base inicial dos ensaios
MAPA_STATUS_BASE = {
   "A2_2019_07_09": "Não Amaciado", "A2_2019_08_08": "Amaciado", "A2_2019_08_28": "Amaciado",
   "A3_2019_12_04": "Não Amaciado", "A3_2019_12_09": "Amaciado", "A3_2019_12_11": "Amaciado",
   "A4_2019_12_16": "Não Amaciado", "A4_2019_12_19": "Amaciado", "A4_2020_01_06": "Amaciado",
   "A5_2020_01_22": "Não Amaciado", "A5_2020_01_27": "Amaciado", "A5_2020_01_28": "Amaciado",
   "B5_2020_10_16": "Não Amaciado", "B5_2020_10_22": "Amaciado", "B5_2020_10_27": "Amaciado",
   "B7_2021_02_05": "Não Amaciado", "B7_2021_02_08": "Amaciado", "B7_2021_02_15": "Amaciado",
   "B10_2021_03_22": "Não Amaciado", "B10_2021_03_25": "Amaciado", "B10_2021_03_30": "Amaciado",
   "B11_2021_04_05": "Não Amaciado", "B11_2021_04_08": "Amaciado", "B11_2021_04_22": "Amaciado",
   "B12_2021_04_27": "Não Amaciado", "B12_2021_04_30": "Amaciado", "B12_2021_05_04": "Amaciado",
   "B15_2021_05_31": "Não Amaciado", "B15_2021_06_09": "Amaciado", "B15_2021_06_15": "Amaciado"
}

def carregar_dados_optuna(caminho_hdf5: Path, tamanho_janela: int, passo: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_lista, y_lista, grupos_lista = [], [], []

    with h5py.File(caminho_hdf5, 'r') as hdf:
        for ensaio in hdf.keys():
            if ensaio not in MAPA_STATUS_BASE: continue

            status_base, modelo_tipo = MAPA_STATUS_BASE[ensaio], ensaio[0]
            x_ensaio_full = hdf[ensaio][:]
            num_amostras_full = x_ensaio_full.shape[0]
            tempos_full = np.arange(num_amostras_full) + tamanho_janela

            if status_base == "Amaciado":
                y_ensaio_full = np.ones(num_amostras_full, dtype=np.int32)
                mask_manter = np.ones(num_amostras_full, dtype=bool)
            else:
                y_ensaio_full = np.zeros(num_amostras_full, dtype=np.int32)
                mask_manter = np.zeros(num_amostras_full, dtype=bool)
                limite_amaciamento = 1440 if modelo_tipo == 'A' else 2880
                mask_manter[tempos_full <= 300] = True
                y_ensaio_full[tempos_full >= limite_amaciamento] = 1
                mask_manter[tempos_full >= limite_amaciamento] = True

            x_final, y_final = x_ensaio_full[mask_manter][::passo], y_ensaio_full[mask_manter][::passo]
            if len(x_final) == 0: continue
            
            x_lista.append(x_final)
            y_lista.append(y_final)
            grupos_lista.append(np.array([ensaio.split('_')[0]] * len(y_final)))

    if not x_lista: return np.array([]), np.array([]), np.array([])
    return np.concatenate(x_lista, axis=0), np.concatenate(y_lista, axis=0), np.concatenate(grupos_lista, axis=0)
