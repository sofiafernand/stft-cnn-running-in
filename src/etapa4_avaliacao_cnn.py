"""
Avaliação do Modelo Final e Teste Cego
Constrói a arquitetura final com a melhor configuração validada pelo Optuna 
e avalia os tensores frente a uma máquina em holdout (invisível no treinamento).
"""

import os, gc, logging
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import gmean

from src.config import settings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(4)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

SEED = 10
tf.random.set_seed(SEED)
np.random.seed(SEED)

class Config:
    CAMINHO_HDF5 = settings.caminho_resultados / "dataset_bandas64_janela16.h5"
    DIR_SAIDA = settings.caminho_resultados / "resultados_finais_trial195"
    MODELO_ALVO = 'B'
    PASSO = 1
    LR = 0.0015897321077381554
    BATCH_SIZE = 8
    FILTROS_1 = 32
    FILTROS_2 = 16
    DENSE_UNITS = 64
    DROPOUT = 0.33355633186604894
    L2 = 0.0018407194543918653
    USAR_FOCAL_LOSS = True
    FOCAL_GAMMA = 1.1703443800045081
    FOCAL_ALPHA = 0.27884919422210147
    EPOCHS = 50
    PATIENCE = 8
    NOMES_CLASSES = ['Não Amaciado', 'Amaciado']

MAPA_STATUS_BASE = {
    "A2_2019_07_09": "Não Amaciado", "A2_2019_08_08": "Amaciado",
    "B15_2021_05_31": "Não Amaciado", "B15_2021_06_15": "Amaciado" # Resumido para visualização
}

def normalizar_ensaio(x_ensaio: np.ndarray) -> np.ndarray:
    x = x_ensaio.astype(np.float32, copy=False)
    mn, mx = x.min(axis=(0, 1), keepdims=True), x.max(axis=(0, 1), keepdims=True)
    dif = mx - mn
    dif[dif < 1e-6] = 1.0
    return ((x - mn) / dif).astype(np.float32)

def carregar_dados_finais(config: Config) -> tuple:
    x_lista, y_lista, g_lista = [], [], []
    with h5py.File(config.CAMINHO_HDF5, 'r') as hdf:
        for ensaio in hdf.keys():
            if ensaio not in MAPA_STATUS_BASE: continue
            status_base, modelo_tipo = MAPA_STATUS_BASE[ensaio], ensaio[0]
            x_ensaio = normalizar_ensaio(hdf[ensaio][:])
            num_amostras, tamanho_janela = x_ensaio.shape[0], x_ensaio.shape[1]
            tempos = np.arange(num_amostras) + tamanho_janela
            
            if status_base == "Amaciado":
                y_ensaio, mask_manter = np.ones(num_amostras, dtype=np.int32), np.ones(num_amostras, dtype=bool)
            else:
                y_ensaio, mask_manter = np.zeros(num_amostras, dtype=np.int32), np.zeros(num_amostras, dtype=bool)
                limite = 1440 if modelo_tipo == 'A' else 2880
                mask_manter[tempos <= 300] = True
                y_ensaio[tempos >= limite] = 1
                mask_manter[tempos >= limite] = True

            x_f, y_f = x_ensaio[mask_manter][::config.PASSO], y_ensaio[mask_manter][::config.PASSO]
            x_lista.append(x_f); y_lista.append(y_f); g_lista.append(np.array([ensaio.split('_')[0]] * len(y_f)))
            
    x, y, g = np.concatenate(x_lista, axis=0), np.concatenate(y_lista, axis=0), np.concatenate(g_lista, axis=0)
    mask_modelo = np.array([grp.startswith(config.MODELO_ALVO) for grp in g])
    return x[mask_modelo], y[mask_modelo], g[mask_modelo]

def construir_modelo_final(input_shape: tuple, config: Config):
    reg = tf.keras.regularizers.l2(config.L2)
    modelo = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(config.FILTROS_1, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation('relu'), tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(config.FILTROS_2, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation('relu'), tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(config.DENSE_UNITS, kernel_initializer='he_normal', kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation('relu'), tf.keras.layers.Dropout(config.DROPOUT),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=config.FOCAL_GAMMA, alpha=config.FOCAL_ALPHA) if config.USAR_FOCAL_LOSS else 'binary_crossentropy'
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.LR), loss=loss, metrics=['accuracy'])
    return modelo

def calcular_class_weight(y_train: np.ndarray, usar_focal: bool):
    if usar_focal: return None
    return dict(zip(np.unique(y_train).tolist(), compute_class_weight('balanced', classes=np.unique(y_train), y=y_train).tolist()))

def encontrar_threshold_otimo(probs, y_true):
    melhor_f1, melhor_thr = 0.0, 0.5
    for thr in np.arange(0.10, 0.91, 0.01):
        f1 = f1_score(y_true, (probs >= thr).astype(int), average='binary', pos_label=0, zero_division=0)
        if f1 > melhor_f1: melhor_f1, melhor_thr = f1, float(thr)
    return melhor_thr, melhor_f1

if __name__ == "__main__":
    conf = Config()
    conf.DIR_SAIDA.mkdir(parents=True, exist_ok=True)
    x, y, groups = carregar_dados_finais(conf)

    mask_test = np.array([g.startswith('B15') for g in groups])
    x_cv, y_cv, groups_cv = x[~mask_test], y[~mask_test], groups[~mask_test]
    x_test, y_test, groups_test = x[mask_test], y[mask_test], groups[mask_test]

    logger.info("Iniciando Treinamento Final...")
    tf.keras.backend.clear_session()
    pesos_keras_final = calcular_class_weight(y_cv, conf.USAR_FOCAL_LOSS)
    modelo_final = construir_modelo_final(x_cv.shape[1:], conf)

    x_cv_train, x_cv_val, y_cv_train, y_cv_val = train_test_split(x_cv, y_cv, test_size=0.10, stratify=y_cv, random_state=SEED)
    
    modelo_final.fit(
        x_cv_train, y_cv_train, epochs=conf.EPOCHS, batch_size=conf.BATCH_SIZE,
        validation_data=(x_cv_val, y_cv_val), class_weight=pesos_keras_final,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=conf.PATIENCE, restore_best_weights=True)], verbose=1
    )

    thr_final, _ = encontrar_threshold_otimo(modelo_final.predict(x_cv_val, verbose=0).ravel(), y_cv_val)
    logger.info("Avaliando Máquina B15...")
    y_pred_test = (modelo_final.predict(x_test, verbose=0).ravel() >= thr_final).astype(int)

    f1_test_binario = f1_score(y_test, y_pred_test, average='binary', pos_label=0, zero_division=0)
    cm_test = confusion_matrix(y_test, y_pred_test)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm_test, display_labels=conf.NOMES_CLASSES).plot(cmap='Blues', ax=ax, values_format='d')
    ax.set_title(f'Teste B15 | F1_C0={f1_test_binario:.4f} | thr={thr_final:.2f}')
    plt.tight_layout()
    plt.savefig(conf.DIR_SAIDA / "matriz_teste_B15.png", dpi=300)
    plt.close(fig)
    modelo_final.save(conf.DIR_SAIDA / "modelo_final_B15.keras")
    logger.info(f"Salvo em: {conf.DIR_SAIDA}")
