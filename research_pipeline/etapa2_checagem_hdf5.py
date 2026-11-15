"""
Pesquisa Original (PRH-ANP): Carregamento e Rotulagem Temporal
Objetivo: Extrair os tensores e aplicar os critérios temporais de assentamento tribológico.
Inclui a definição da zona cinzenta (transiente) onde os rótulos são descartados.
"""

import h5py
import numpy as np
from pathlib import Path

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

def carregar_dados_optuna(caminho_hdf5: Path, tamanho_janela: int, passo: int = 1) -> tuple:
    X_lista, Y_lista, grupos_lista = [], [], []

    with h5py.File(caminho_hdf5, 'r') as hdf:
        for ensaio in hdf.keys():
            if ensaio not in MAPA_STATUS_BASE: continue
            
            status_base = MAPA_STATUS_BASE[ensaio]
            X_ensaio_full = hdf[ensaio][:]
            num_amostras_full = X_ensaio_full.shape[0]
            tempos_full = np.arange(num_amostras_full) + tamanho_janela

            if status_base == "Amaciado":
                Y_ensaio_full, mask_manter = np.ones(num_amostras_full, dtype=np.int32), np.ones(num_amostras_full, dtype=bool)
            else:
                Y_ensaio_full, mask_manter = np.zeros(num_amostras_full, dtype=np.int32), np.zeros(num_amostras_full, dtype=bool)
                limite_amaciamento = 1440 if ensaio[0] == 'A' else 2880
                mask_manter[tempos_full <= 300] = True
                Y_ensaio_full[tempos_full >= limite_amaciamento] = 1
                mask_manter[tempos_full >= limite_amaciamento] = True

            X_final, Y_final = X_ensaio_full[mask_manter][::passo], Y_ensaio_full[mask_manter][::passo]
            if len(X_final) == 0: continue
            
            X_lista.append(X_final)
            Y_lista.append(Y_final)
            grupos_lista.append(np.array([ensaio.split('_')[0]] * len(Y_final)))

    if not X_lista: return np.array([]), np.array([]), np.array([])
    return np.concatenate(X_lista, axis=0), np.concatenate(Y_lista, axis=0), np.concatenate(grupos_lista, axis=0)
