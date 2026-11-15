"""
Módulo de Otimização de Hiperparâmetros (Optuna)
Refatorado para importar configurações dinâmicas via Pydantic Settings,
eliminando caminhos locais fixos.
"""

import os, gc, logging, threading, warnings, math
import numpy as np
from pathlib import Path
from functools import lru_cache

# Importação corrigida e padronizada
from src.etapa2_pre_processamento import carregar_dados_optuna
from src.config import settings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(4)

import optuna
from optuna_dashboard import run_server
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from scipy.stats import gmean

warnings.filterwarnings('ignore', category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

SEED = 10
tf.random.set_seed(SEED)
np.random.seed(SEED)

MODELO_ALVO      = 'B'
NOME_ESTUDO      = f"Otimizacao_Modelo_{MODELO_ALVO}"
NOME_BANCO       = f"sqlite:///optuna_modelo_{MODELO_ALVO}_v2.db"
DASHBOARD_PORT   = 8080
MAQUINA_HOLDOUT  = 'B15'
N_SPLITS = 5
EPOCHS   = 20
PATIENCE = 5

def iniciar_dashboard(storage: str, port: int) -> None:
    try:
        run_server(storage, host="localhost", port=port)
    except Exception as e:
        logger.warning(f"Dashboard encerrado: {e}")

def calcular_scaler_treino(X_train: np.ndarray) -> tuple:
    mn, mx = X_train.min(axis=(0, 1, 2), keepdims=True), X_train.max(axis=(0, 1, 2), keepdims=True)
    dif = mx - mn
    dif[dif < 1e-6] = 1.0
    return mn, dif

def aplicar_scaler(X: np.ndarray, mn: np.ndarray, dif: np.ndarray) -> np.ndarray:
    return ((X - mn) / dif).astype(np.float32)

def construir_cnn(input_shape, f1, f2, dense, dropout, l2, lr, usar_focal, focal_gamma, focal_alpha):
    reg = tf.keras.regularizers.l2(l2)
    modelo = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(f1, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation('relu'), tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(f2, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation('relu'), tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(dense, kernel_initializer='he_normal', kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation('relu'), tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=focal_gamma, alpha=focal_alpha) if usar_focal else 'binary_crossentropy'
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=['accuracy'])
    return modelo

def encontrar_threshold_otimo(probs, y_true):
    melhor_f1, melhor_thr = 0.0, 0.5
    for thr in np.arange(0.10, 0.91, 0.01):
        f1 = f1_score(y_true, (probs >= thr).astype(int), average='binary', pos_label=0, zero_division=0)
        if f1 > melhor_f1: melhor_f1, melhor_thr = f1, float(thr)
    return melhor_thr, melhor_f1

@lru_cache(maxsize=None)
def carregar_dados_cache(nb: int, tj: int, p: int):
    # Uso do settings (Injeção de dependência via .env)
    caminho_hdf5 = settings.caminho_resultados / f"dataset_bandas{nb}_janela{tj}.h5"
    if not caminho_hdf5.exists(): return None
    X_all, Y_all, grupos_all = carregar_dados_optuna(caminho_hdf5, tj, passo=p)
    mask = np.array([g.startswith(MODELO_ALVO) for g in grupos_all])
    X, Y, grupos = X_all[mask], Y_all[mask], grupos_all[mask]
    mask_holdout = np.array([not g.startswith(MAQUINA_HOLDOUT) for g in grupos])
    return X[mask_holdout], Y[mask_holdout], grupos[mask_holdout]

def objective(trial: optuna.Trial) -> float:
    nb = trial.suggest_categorical('num_bandas',     [16, 32, 64, 128, 256])
    tj = trial.suggest_categorical('tamanho_janela', [16, 32, 64, 128])
    p  = trial.suggest_categorical('passo',          [1, 4, 8, 16])
    
    resultado = carregar_dados_cache(nb, tj, p)
    if resultado is None: raise optuna.exceptions.TrialPruned()
    X, Y, grupos = resultado
    if len(X) == 0 or len(np.unique(Y)) < 2: raise optuna.exceptions.TrialPruned()

    lr         = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size',    [8, 16, 32])
    f1_conv    = trial.suggest_categorical('filtros_conv1', [16, 32, 64])
    f2_conv    = trial.suggest_categorical('filtros_conv2', [16, 32, 64])
    dense      = trial.suggest_categorical('dense_units',   [32, 64, 128])
    dropout    = trial.suggest_float('dropout_rate', 0.1, 0.5)
    l2         = trial.suggest_float('l2_reg', 1e-4, 1e-2, log=True)
    usar_focal = trial.suggest_categorical('usar_focal_loss', [True, False])
    
    focal_gamma, focal_alpha = (trial.suggest_float('focal_gamma', 0.5, 5.0), trial.suggest_float('focal_alpha', 0.1, 0.9)) if usar_focal else (2.0, 0.25)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS)
    f1_folds = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, Y, groups=grupos), start=1):
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val,   Y_val   = X[val_idx],   Y[val_idx]
        if len(np.unique(Y_train)) < 2: continue

        mn_fold, dif_fold = calcular_scaler_treino(X_train)
        X_train = aplicar_scaler(X_train, mn_fold, dif_fold)
        X_val   = aplicar_scaler(X_val,   mn_fold, dif_fold)

        pesos_keras = None if usar_focal else dict(zip(np.unique(Y_train).tolist(), compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train).tolist()))
        tf.keras.backend.clear_session()
        model = construir_cnn(X_train.shape[1:], f1_conv, f2_conv, dense, dropout, l2, lr, usar_focal, focal_gamma, focal_alpha)

        X_train_train, X_train_val, Y_train_train, Y_train_val = train_test_split(X_train, Y_train, test_size=0.10, stratify=Y_train, random_state=SEED)
        
        model.fit(X_train_train, Y_train_train, validation_data=(X_train_val, Y_train_val), epochs=EPOCHS, batch_size=batch_size, class_weight=pesos_keras, callbacks=[EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)], verbose=0)
        thr_val, _ = encontrar_threshold_otimo(model.predict(X_train_val, verbose=0).ravel(), Y_train_val)
        preds_oof = (model.predict(X_val, verbose=0).ravel() >= thr_val).astype(int)
        
        f1_fold = f1_score(Y_val, preds_oof, average='binary', pos_label=0, zero_division=0)
        f1_folds.append(f1_fold)

        if fold >= 2:
            valor_relatorio = float(gmean(f1_folds)) if f1_folds else 0.0
            if math.isfinite(valor_relatorio):
                trial.report(valor_relatorio, step=fold)
                if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        del model, X_train, Y_train
    gc.collect()
    return float(gmean(f1_folds)) if f1_folds else 0.0

def log_progresso(study: optuna.Study, trial: optuna.Trial) -> None:
    if trial.number % 5 == 0:
        concluidos = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        valor_str = f'{trial.value:.4f}' if trial.value is not None else 'PODADO'
        logger.info(f"Trial {trial.number} | Valor: {valor_str} | Melhor: {study.best_value:.4f} | Concluídos: {concluidos}")

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED), pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2), study_name=NOME_ESTUDO, storage=optuna.storages.RDBStorage(url=NOME_BANCO, engine_kwargs={"connect_args": {"timeout": 30}}), load_if_exists=True)
    threading.Thread(target=iniciar_dashboard, args=(NOME_BANCO, DASHBOARD_PORT), daemon=True).start()
    try: study.optimize(objective, n_trials=None, gc_after_trial=True, callbacks=[log_progresso])
    except KeyboardInterrupt: logger.info("Otimização pausada.")
