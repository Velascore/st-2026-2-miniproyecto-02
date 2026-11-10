"""Miniproyecto 2: aprendizaje automático y profundo contra el mejor modelo clásico.

Se corre de principio a fin con

    python miniproyecto.py

y deja las figuras en figuras/ y las tablas en resultados/. Las partes
marcadas con TODO las completa el grupo; mientras falten, el guion las
reporta como pendientes y sigue. El informe, docs/informe.md, cita las cifras
y las figuras que produce este guion.
"""
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: E402
from torch import nn  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from comun import (RAIZ, backtest, cargar_serie, estacional_ingenuo,  # noqa: E402
                   fechas_siguientes, rmse)

# Después de importar statsmodels, que al cargarse fija sus propios avisos.
warnings.filterwarnings("ignore")
torch.set_num_threads(1)
FIGURAS = RAIZ / "figuras"
RESULTADOS = RAIZ / "resultados"
FIGURAS.mkdir(exist_ok=True)
RESULTADOS.mkdir(exist_ok=True)

SEMILLA = 2026
N_PRUEBA = 24          # los meses del backtest, los mismos de comun.backtest

# El último modelo que ajustó cada función modelo(entrenamiento, h). Sirve
# para contar sus parámetros después del backtest.
ULTIMO = {}

# ============================================================
# 1. La serie y el modelo a superar
# ============================================================
# Los mismos datos del miniproyecto 1: índice de producción industrial de
# fabricación de cajas de cartón corrugado (EE. UU., FRED IPN32221N), mensual.
ARCHIVO = "datos/IPN32221N_cajas_carton_corrugado.csv"
FECHA = "observation_date"
VALOR = "IPN32221N"
PERIODO = 12

# El mejor modelo clásico del miniproyecto 1 y sus órdenes, fijos aquí.
ORDEN = (0, 1, 2)
ESTACIONAL = (0, 1, 2, PERIODO)


def preparar(serie):
    """La misma preparación del miniproyecto 1: winsoriza los atípicos.

    La serie IPN32221N no tiene meses vacíos (0 NaN), así que no hay huecos que
    rellenar. Los atípicos no son errores de medición sino dos episodios
    económicos reales —el desplome de la COVID (abr–jun 2020) y el rebote
    post-pandemia (feb–jun 2022)—, que inflan la varianza del residuo y pueden
    dominar el ajuste del SARIMA. La decisión es *winsorizarlos*: recortar al
    límite, sin borrar meses, solo los puntos cuyo residuo de una STL robusta
    (periodo 12) se sale de la banda Q1−3·IQR .. Q3+3·IQR. Devuelve una serie
    sin NaN.
    """
    serie = serie.astype(float)
    if serie.isna().any():  # por robustez; la serie base no tiene huecos
        serie = serie.interpolate(method="time")
    resid = STL(serie, period=PERIODO, robust=True).fit().resid
    q1, q3 = resid.quantile(0.25), resid.quantile(0.75)
    iqr = q3 - q1
    bajo, alto = q1 - 3 * iqr, q3 + 3 * iqr
    exceso = resid - resid.clip(bajo, alto)   # 0 donde no hay atípico
    limpia = serie - exceso                    # recorta solo los puntos fuera de banda
    limpia.name = serie.name
    print(f"Preparar: 0 huecos; {int((exceso != 0).sum())} atípicos winsorizados "
          f"(banda residuo STL [{bajo:.2f}, {alto:.2f}])")
    return limpia


# ---------- Festivos federales de EE. UU. (sin la librería `holidays`) ----------
# El miniproyecto 2 solo admite numpy, pandas, matplotlib, statsmodels,
# scikit-learn, xgboost y torch, así que la exógena de días hábiles del
# miniproyecto 1 recalcula los festivos federales por sus reglas. Reproduce
# exactamente el conteo mensual de días hábiles de `holidays.US()`.

def _enesimo_lunes(anio, mes, weekday, n):
    """n-ésimo `weekday` (0=lunes) del mes; n negativo cuenta desde el final."""
    dias = pd.date_range(f"{anio}-{mes:02d}-01", periods=31, freq="D")
    dias = dias[dias.month == mes]
    coinc = dias[dias.weekday == weekday]
    return coinc[n - 1] if n > 0 else coinc[n]


def _observado(fecha):
    """Regla federal: sábado -> viernes anterior; domingo -> lunes siguiente."""
    if fecha.weekday() == 5:
        return fecha - pd.Timedelta(days=1)
    if fecha.weekday() == 6:
        return fecha + pd.Timedelta(days=1)
    return fecha


def _festivos_eeuu(anios):
    """Fechas de festivos federales de EE. UU., con el día observado si cae en fin de semana."""
    fechas = set()
    for anio in anios:
        fijos = [pd.Timestamp(anio, 1, 1), pd.Timestamp(anio, 7, 4),
                 pd.Timestamp(anio, 11, 11), pd.Timestamp(anio, 12, 25)]
        if anio >= 2021:
            fijos.append(pd.Timestamp(anio, 6, 19))   # Juneteenth, federal desde 2021
        moviles = [_enesimo_lunes(anio, 1, 0, 3),      # MLK
                   _enesimo_lunes(anio, 2, 0, 3),      # Washington
                   _enesimo_lunes(anio, 5, 0, -1),     # Memorial
                   _enesimo_lunes(anio, 9, 0, 1),      # Trabajo
                   _enesimo_lunes(anio, 10, 0, 2),     # Colón
                   _enesimo_lunes(anio, 11, 3, 4)]     # Acción de Gracias
        for f in fijos:
            fechas.add(f.date())
            fechas.add(_observado(f).date())
        for f in moviles:
            fechas.add(f.date())
    return fechas


def exogenas(serie):
    """Días hábiles de cada mes (lunes a viernes, sin festivos federales de EE. UU.).

    Es una variable de calendario: depende solo de la fecha, se conoce de
    antemano para cualquier mes futuro y por eso nunca introduce fuga. Cubre el
    índice de la serie y los 12 meses siguientes, para poder pronosticar.
    """
    idx = pd.date_range(serie.index[0],
                        serie.index[-1] + pd.offsets.MonthBegin(12), freq="MS")
    festivos = _festivos_eeuu(range(idx[0].year, idx[-1].year + 1))
    conteo = []
    for mes in idx:
        dias = pd.date_range(mes.replace(day=1),
                             mes + pd.offsets.MonthEnd(0), freq="D")
        conteo.append(sum(1 for dia in dias
                          if dia.weekday() < 5 and dia.date() not in festivos))
    return pd.DataFrame({"dias_habiles": conteo}, index=idx, dtype=float)


def modelo_clasico(serie):
    """La función modelo(entrenamiento, h) del mejor modelo del miniproyecto 1:
    SARIMAX(0,1,2)(0,1,2)₁₂ con los días hábiles del mes como exógena.

    La exógena de calendario se precalcula sobre toda la serie y sus 12 meses
    siguientes; dentro del backtest cada origen usa solo x.loc[entrenamiento] y
    los días hábiles de las fechas siguientes (calendario, no datos futuros de
    la serie), así que no hay fuga. El ajuste se guarda en ULTIMO["modelo"]
    para contar sus parámetros.
    """
    x = exogenas(serie)

    def modelo(entrenamiento, h):
        x_tr = x.loc[entrenamiento.index]
        x_fut = x.loc[fechas_siguientes(entrenamiento, h)]
        ajuste = SARIMAX(entrenamiento, order=ORDEN, seasonal_order=ESTACIONAL,
                         exog=x_tr, enforce_stationarity=False,
                         enforce_invertibility=False).fit(disp=False)
        ULTIMO["modelo"] = ajuste
        return ajuste.forecast(h, exog=x_fut)
    return modelo


NOMBRE_CLASICO = "SARIMAX(0,1,2)(0,1,2)₁₂ + días hábiles"


def graficar_serie(serie):
    figura, eje = plt.subplots(figsize=(9, 3.4))
    eje.plot(serie.index, serie.to_numpy())
    eje.axvline(serie.index[-N_PRUEBA], color="gray", linestyle="--")
    eje.set_title(f"{VALOR}: serie mensual; a la derecha de la línea, los meses del backtest")
    eje.set_ylabel(VALOR)
    figura.tight_layout()
    figura.savefig(FIGURAS / "serie.png", dpi=120)
    plt.close(figura)


# ============================================================
# 2. La tabla supervisada
# ============================================================
# Lo que aprende cada modelo: "nivel" es la serie tal cual; "dif12" es la
# diferencia con el mismo mes del año anterior.
# Los modelos aprenden la diferencia con el mismo mes del año anterior
# (dif12), no el nivel. La serie tiene tendencia y una estacionalidad fuerte
# (fuerza estacional Fs = 0,87 en el miniproyecto 1); al restar y(t-12) se
# quita casi toda la tendencia y la estacionalidad y el objetivo queda casi
# estacionario, que es lo que KNN, SVR y los árboles extrapolan bien. Sobre el
# nivel, en cambio, esos modelos no saben extrapolar una tendencia creciente.
OBJETIVO = "dif12"
REZAGOS = 12


def objetivo(y):
    return y.diff(PERIODO) if OBJETIVO == "dif12" else y


def a_nivel(y, z):
    """Del pronóstico del objetivo al nivel: y(t) = y(t-12) + z(t) si z es la diferencia."""
    return y.iloc[-PERIODO] + z if OBJETIVO == "dif12" else z


def tabla_supervisada(z):
    """Una fila por mes con los rezagos 1..REZAGOS del objetivo, el mes del año
    (codificado en seno y coseno) y dos estadísticas de ventana móvil.

    Ninguna columna de la fila t usa z(t): los rezagos son z.shift(k) con
    k>=1 y las ventanas se calculan sobre z.shift(1), es decir hasta el mes
    anterior. Los rezagos se llaman rezago_1..rezago_12 porque las redes de
    secuencia (LSTM, Conv1D) los leen en ese orden.
    """
    columnas = {f"rezago_{k}": z.shift(k) for k in range(1, REZAGOS + 1)}
    mes = z.index.month
    # El mes del año, cíclico: diciembre (12) queda junto a enero (1).
    columnas["mes_sin"] = pd.Series(np.sin(2 * np.pi * mes / 12), index=z.index)
    columnas["mes_cos"] = pd.Series(np.cos(2 * np.pi * mes / 12), index=z.index)
    # Ventanas móviles sobre el objetivo corrido un mes (nunca ven z(t)).
    movil = z.shift(1)
    columnas["media_movil_3"] = movil.rolling(3).mean()
    columnas["desv_movil_6"] = movil.rolling(6).std()
    return pd.DataFrame(columnas, index=z.index)


def comprobar_sin_fuga(serie):
    """Prueba en el código que ninguna fila ve el mes que predice.

    Altera el valor de cada mes con un cambio grande y verifica que la fila de
    la tabla de ESE mes no cambie: si cambiara, la fila estaría usando y(t)
    —directamente o a través del objetivo z(t)— para predecir y(t). Si alguna
    fila cambia, lanza AssertionError.
    """
    z0 = objetivo(serie)
    X0 = tabla_supervisada(z0)
    revisados = 0
    for t in X0.index:
        if X0.loc[t].isna().any():          # filas sin historia completa
            continue
        alterada = serie.copy()
        alterada.loc[t] = alterada.loc[t] + 100.0   # cambio grande en y(t)
        fila_alterada = tabla_supervisada(objetivo(alterada)).loc[t]
        if not np.allclose(X0.loc[t].to_numpy(), fila_alterada.to_numpy(),
                           equal_nan=True):
            raise AssertionError(
                f"Fuga: la fila {t:%Y-%m} cambia al alterar su propio mes")
        revisados += 1
    print(f"Sin fuga: {revisados} filas revisadas, ninguna cambia al alterar "
          f"el mes que predice")


def filas_completas(y):
    z = objetivo(y)
    X = tabla_supervisada(z)
    completas = X.notna().all(axis=1) & z.notna()
    return X[completas], z[completas]


def recursivo(ajustado, y, h):
    """Pronóstico recursivo a h meses.

    En cada paso agrega el mes siguiente como NaN, arma su fila de la tabla,
    pronostica el objetivo con ajustado.predict, lo lleva al nivel con a_nivel
    (usando la historia hasta el mes anterior, cuyo iloc[-12] es y(t-12)) y lo
    escribe como si fuera dato antes de pasar al mes siguiente. Así cada
    pronóstico entra como rezago del mes que sigue.
    """
    historia = y.copy()
    pronosticos = []
    for _ in range(h):
        siguiente = historia.index[-1] + pd.offsets.MonthBegin(1)
        futuro = pd.concat(
            [historia, pd.Series([np.nan], index=[siguiente], name=historia.name)])
        fila = tabla_supervisada(objetivo(futuro)).loc[[siguiente]]
        z_hat = float(ajustado.predict(fila)[0])
        y_hat = a_nivel(historia, z_hat)    # historia termina en t-1 -> iloc[-12]=y(t-12)
        historia = pd.concat(
            [historia, pd.Series([y_hat], index=[siguiente], name=historia.name)])
        pronosticos.append(y_hat)
    return np.array(pronosticos)


def con_tabla(fabrica):
    """Modelo para el backtest: arma la tabla con el entrenamiento del origen,
    ajusta un modelo nuevo y pronostica h meses en modo recursivo. El
    escalador vive dentro del modelo, así que se ajusta en cada origen."""
    def modelo(entrenamiento, h):
        X, z = filas_completas(entrenamiento)
        ajustado = fabrica()
        ajustado.fit(X, z)
        ULTIMO["modelo"] = ajustado
        return recursivo(ajustado, entrenamiento, h)
    return modelo


# ============================================================
# 3. Aprendizaje automático
# ============================================================
# Las cuatro fábricas devuelven un modelo nuevo sin entrenar; con_tabla
# ajusta uno por origen. KNN y SVR llevan el escalador dentro del pipeline.

def knn(n_neighbors=5):
    return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=n_neighbors))


def svr(C=1.0, epsilon=0.1):
    return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=C, epsilon=epsilon))


def bosque(max_depth=None, min_samples_leaf=1):
    return RandomForestRegressor(n_estimators=100, max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf, random_state=SEMILLA)


def xgboost(max_depth=3, learning_rate=0.1):
    return XGBRegressor(n_estimators=200, max_depth=max_depth, learning_rate=learning_rate,
                        random_state=SEMILLA)


# Cada rejilla lleva de dos a cuatro valores por hiperparámetro.
REJILLAS = {
    "KNN": (knn, {"n_neighbors": [3, 5, 10, 15]}),
    "SVR": (svr, {"C": [1.0, 10.0, 100.0], "epsilon": [0.05, 0.1, 0.2]}),
    "Random Forest": (bosque, {"max_depth": [None, 3, 5, 10], "min_samples_leaf": [1, 2, 4]}),
    "XGBoost": (xgboost, {"max_depth": [2, 3, 4], "learning_rate": [0.05, 0.1, 0.2]}),
}


def sin_prefijo(parametros):
    return {nombre.split("__")[-1]: valor for nombre, valor in parametros.items()}


def ajustar_hiperparametros(serie):
    """Para cada modelo de REJILLAS, la combinación con menor RMSE en validación
    temporal.

    La tabla se arma con serie.iloc[:-N_PRUEBA], de modo que ninguno de los 24
    meses del backtest participa en la elección. La búsqueda es GridSearchCV con
    TimeSeriesSplit (5 cortes): cada corte entrena con el pasado y valida con el
    tramo siguiente, nunca al revés. KNN y SVR son make_pipeline, así que sus
    claves llevan el prefijo del paso (p. ej. kneighborsregressor__n_neighbors);
    sin_prefijo lo quita para devolver {hiperparámetro: valor}.
    """
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    corte_temporal = TimeSeriesSplit(n_splits=5)
    elegidos = {}
    print("  hiperparámetros (TimeSeriesSplit, solo meses previos al backtest):")
    for nombre, (fabrica, rejilla) in REJILLAS.items():
        estimador = fabrica()
        prefijo = estimador.steps[-1][0] + "__" if hasattr(estimador, "steps") else ""
        malla = {prefijo + clave: valores for clave, valores in rejilla.items()}
        busqueda = GridSearchCV(estimador, malla, cv=corte_temporal,
                                scoring="neg_root_mean_squared_error")
        busqueda.fit(X, z)
        elegidos[nombre] = sin_prefijo(busqueda.best_params_)
        print(f"    {nombre}: {elegidos[nombre]}  "
              f"RMSE_val={-busqueda.best_score_:.2f}")
    return elegidos


def graficar_importancia(serie, elegidos):
    """Importancia de las variables de Random Forest y XGBoost, ajustados con lo
    elegido sobre los meses previos al backtest; guarda figuras/importancia.png."""
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    figura, ejes = plt.subplots(1, 2, figsize=(10, 5))
    for eje, nombre, fabrica in [(ejes[0], "Random Forest", bosque),
                                 (ejes[1], "XGBoost", xgboost)]:
        modelo = fabrica(**elegidos[nombre]).fit(X, z)
        imp = pd.Series(modelo.feature_importances_, index=X.columns).sort_values()
        eje.barh(imp.index, imp.to_numpy(), color="steelblue")
        eje.set_title(f"{nombre}", fontsize=11)
        eje.tick_params(axis="y", labelsize=8)
    figura.suptitle("Importancia de las variables")
    figura.tight_layout()
    figura.savefig(FIGURAS / "importancia.png", dpi=120)
    plt.close(figura)
    print("Importancia de variables en figuras/importancia.png")


# ============================================================
# 4. Aprendizaje profundo en PyTorch
# ============================================================
# Redes pequeñas y pocas épocas: el guion completo tiene que correr en CPU
# en pocos minutos.
EPOCAS_MAXIMAS = 300
# Tasa de aprendizaje: con 0.01 el entrenamiento de pleno lote del MLP divergía
# en algunos orígenes del backtest (RMSE ~1e4); con 0.005 las tres redes
# entrenan estables en todo el rango de épocas.
TASA = 0.005


class Red:
    """Una red de PyTorch con fit y predict, para entrar a la tabla como un
    modelo de scikit-learn. Escala X y z con lo que ve en fit, que es el
    entrenamiento del origen, y devuelve el pronóstico en la escala original.
    Con secuencia=True la red recibe los rezagos en orden, con forma
    (lote, 12, 1), del más viejo al más reciente."""

    def __init__(self, construir, secuencia=False, epocas=EPOCAS_MAXIMAS, semilla=SEMILLA):
        self.construir = construir
        self.secuencia = secuencia
        self.epocas = epocas
        self.semilla = semilla

    def entradas(self, X):
        if self.secuencia:
            columnas = [f"rezago_{k}" for k in range(REZAGOS, 0, -1)]
            x = (X[columnas].to_numpy() - self.media) / self.desv
            return torch.tensor(x[:, :, None], dtype=torch.float32)
        return torch.tensor(self.escala_x.transform(X), dtype=torch.float32)

    def fit(self, X, z, X_val=None, z_val=None):
        torch.manual_seed(self.semilla)
        self.media, self.desv = z.mean(), z.std()
        self.escala_x = StandardScaler().fit(X)
        x = self.entradas(X)
        objetivo_t = torch.tensor(((z - self.media) / self.desv).to_numpy()[:, None],
                                  dtype=torch.float32)
        self.red = self.construir(x.shape[-1])
        optimizador = torch.optim.Adam(self.red.parameters(), lr=TASA)
        perdida = nn.MSELoss()
        self.perdidas = []
        for _ in range(self.epocas):
            self.red.train()
            optimizador.zero_grad()
            error = perdida(self.red(x), objetivo_t)
            error.backward()
            # Recorte de la norma del gradiente, como segundo seguro contra la
            # divergencia del entrenamiento de pleno lote.
            nn.utils.clip_grad_norm_(self.red.parameters(), max_norm=1.0)
            optimizador.step()
            fila = [error.item()]
            if X_val is not None:
                fila.append(float(np.mean(((self.predict(X_val) - z_val) / self.desv) ** 2)))
            self.perdidas.append(fila)
        return self

    def predict(self, X):
        self.red.eval()
        with torch.no_grad():
            salida = self.red(self.entradas(X)).numpy()[:, 0]
        return salida * self.desv + self.media


class MLP(nn.Module):
    def __init__(self, entradas, ocultas=16):
        super().__init__()
        self.capas = nn.Sequential(nn.Linear(entradas, ocultas), nn.ReLU(),
                                   nn.Linear(ocultas, 1))

    def forward(self, x):
        return self.capas(x)


class LSTM(nn.Module):
    """nn.LSTM de una capa sobre la secuencia (lote, 12, 1) y una capa lineal
    que lee el último estado oculto."""

    def __init__(self, entradas=1, ocultas=16):
        super().__init__()
        self.lstm = nn.LSTM(entradas, ocultas, batch_first=True)
        self.salida = nn.Linear(ocultas, 1)

    def forward(self, x):
        _, (estado, _) = self.lstm(x)      # estado: (1, lote, ocultas)
        return self.salida(estado[-1])     # lee el último estado oculto


class Conv1D(nn.Module):
    """La tercera red: una convolución 1D sobre la secuencia (lote, 12, 1).

    Una capa Conv1d (un canal de entrada -> `canales`, núcleo 3) con ReLU lee
    ventanas de tres meses de los rezagos; un promedio global sobre el tiempo
    resume la secuencia y una capa lineal da el pronóstico. Se eligió frente a
    un Transformer o un híbrido porque, con solo ~170 filas por origen, una red
    convolucional pequeña es la que mejor aprovecha los datos sin sobreajustar
    y entra limpia en el mismo mecanismo de secuencia que la LSTM.
    """

    def __init__(self, entradas=1, canales=16, nucleo=3):
        super().__init__()
        self.conv = nn.Conv1d(entradas, canales, kernel_size=nucleo, padding=nucleo // 2)
        self.salida = nn.Linear(canales, 1)

    def forward(self, x):
        z = x.transpose(1, 2)              # (lote, 12, 1) -> (lote, 1, 12)
        z = torch.relu(self.conv(z))       # (lote, canales, 12)
        z = z.mean(dim=2)                  # promedio global en el tiempo
        return self.salida(z)


# Las candidatas: nombre, cómo se construye y si recibe la secuencia. Los dos
# tamaños de MLP reciben las columnas sueltas; la LSTM y la Conv1D reciben la
# secuencia de rezagos (lote, 12, 1).
CANDIDATAS = {
    "MLP (8 ocultas)": (lambda n: MLP(n, 8), False),
    "MLP (32 ocultas)": (lambda n: MLP(n, 32), False),
    "LSTM (16 ocultas)": (lambda n: LSTM(n, 16), True),
    "Conv1D (16 canales)": (lambda n: Conv1D(n, 16), True),
}


def validar_redes(serie):
    """Elige las épocas de cada red con validación temporal.

    La tabla se arma con serie.iloc[:-N_PRUEBA] (ningún mes del backtest entra)
    y sus últimos 12 meses son la validación. Cada candidata se entrena hasta
    EPOCAS_MAXIMAS guardando por época la pérdida de entrenamiento y la de
    validación; las curvas van a figuras/perdida.png. De cada red se toman las
    épocas de menor pérdida de validación, y entre los dos MLP se conserva solo
    el de menor pérdida. Devuelve {nombre: épocas}.
    """
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    X_ent, z_ent = X.iloc[:-12], z.iloc[:-12]
    X_val, z_val = X.iloc[-12:], z.iloc[-12:]

    figura, ejes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    epocas, val_min = {}, {}
    for (nombre, (construir, secuencia)), eje in zip(CANDIDATAS.items(), ejes.ravel()):
        modelo = Red(construir, secuencia, EPOCAS_MAXIMAS).fit(X_ent, z_ent, X_val, z_val)
        perdidas = np.array(modelo.perdidas)        # columnas: entrenamiento, validación
        e_opt = int(np.argmin(perdidas[:, 1])) + 1
        epocas[nombre] = e_opt
        val_min[nombre] = perdidas[:, 1].min()
        eje.plot(range(1, len(perdidas) + 1), perdidas[:, 0], color="steelblue",
                 label="entrenamiento")
        eje.plot(range(1, len(perdidas) + 1), perdidas[:, 1], color="crimson",
                 label="validación")
        eje.axvline(e_opt, ls="--", lw=.8, color="gray")
        eje.set_title(f"{nombre}: mín. validación @ {e_opt} épocas", fontsize=9)
        eje.set_ylabel("MSE (escala del objetivo)", fontsize=8)
    ejes.ravel()[0].legend(fontsize=8)
    for eje in ejes[-1]:
        eje.set_xlabel("época")
    figura.suptitle("Validación temporal de las redes: pérdida por época")
    figura.tight_layout()
    figura.savefig(FIGURAS / "perdida.png", dpi=120)
    plt.close(figura)

    # Entre los dos tamaños de MLP se queda el de menor pérdida de validación.
    mlps = {n: v for n, v in val_min.items() if n.startswith("MLP")}
    mejor_mlp = min(mlps, key=mlps.get)
    for n in [n for n in list(epocas) if n.startswith("MLP") and n != mejor_mlp]:
        del epocas[n]
    print(f"  MLP elegido: {mejor_mlp} (val {val_min[mejor_mlp]:.3f}); épocas {epocas}")
    return epocas


def red(nombre, epocas, semilla=SEMILLA):
    """La fábrica de una red para con_tabla."""
    construir, secuencia = CANDIDATAS[nombre]
    return lambda: Red(construir, secuencia, epocas, semilla)


def repetir_con_semillas(serie, nombre, epocas):
    """La mejor red con cinco semillas, cada una por el backtest.

    Mide la variación del RMSE de la red al cambiar solo la semilla de la
    inicialización. Guarda resultados/semillas.csv con una fila por semilla e
    imprime la media y la desviación del RMSE.
    """
    filas = []
    for semilla in range(SEMILLA, SEMILLA + 5):
        _, promedio = backtest(serie, con_tabla(red(nombre, epocas, semilla)))
        filas.append({"semilla": semilla, "RMSE": round(promedio["RMSE"], 3)})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(RESULTADOS / "semillas.csv", index=False)
    media, desv = tabla["RMSE"].mean(), tabla["RMSE"].std()
    print(f"  {nombre} con 5 semillas: RMSE media={media:.3f}, desv={desv:.3f} "
          f"(min {tabla['RMSE'].min():.3f}, max {tabla['RMSE'].max():.3f})")
    return tabla


# ============================================================
# 5. La tabla y la elección
# ============================================================

def contar_parametros(ajustado):
    """Cuántos números guarda el modelo ajustado. En los clásicos y las redes
    son los parámetros; en KNN, los valores que memoriza; en SVR, los vectores
    de soporte y sus coeficientes; en los árboles, los nodos."""
    if hasattr(ajustado, "params"):
        return len(ajustado.params)
    if isinstance(ajustado, Red):
        return sum(p.numel() for p in ajustado.red.parameters())
    final = ajustado.steps[-1][1] if hasattr(ajustado, "steps") else ajustado
    if isinstance(final, KNeighborsRegressor):
        return final.n_samples_fit_ * final.n_features_in_
    if isinstance(final, SVR):
        return final.support_vectors_.size + final.dual_coef_.size + 1
    if isinstance(final, RandomForestRegressor):
        return sum(arbol.tree_.node_count for arbol in final.estimators_)
    if isinstance(final, XGBRegressor):
        return len(final.get_booster().trees_to_dataframe())
    return 0


def fila(nombre, serie, modelo):
    ULTIMO.clear()
    inicio = time.perf_counter()
    _, promedio = backtest(serie, modelo)
    segundos = time.perf_counter() - inicio
    parametros = contar_parametros(ULTIMO["modelo"]) if "modelo" in ULTIMO else 0
    print(f"  {nombre}: RMSE {promedio['RMSE']:.2f} en {segundos:.1f} s")
    return {"modelo": nombre, "RMSE": round(promedio["RMSE"], 2),
            "MAE": round(promedio["MAE"], 2), "MAPE": round(promedio["MAPE"], 2),
            "parametros": parametros, "segundos": round(segundos, 1)}


def graficar_pronosticos(serie, modelos):
    """Pronóstico desde el último origen del backtest de cada modelo elegido."""
    entrenamiento, real = serie.iloc[:-12], serie.iloc[-12:]
    figura, eje = plt.subplots(figsize=(9, 3.6))
    eje.plot(serie.index[-36:], serie.iloc[-36:].to_numpy(), color="black", label="real")
    for nombre, modelo in modelos.items():
        pronostico = modelo(entrenamiento, 12)
        eje.plot(real.index, pronostico, marker="o", markersize=3,
                 label=f"{nombre} (RMSE {rmse(real, pronostico):.1f})")
    eje.set_ylabel(VALOR)
    eje.set_title("Pronóstico a 12 meses desde el último origen")
    eje.legend(fontsize=8)
    figura.tight_layout()
    figura.savefig(FIGURAS / "pronosticos.png", dpi=120)
    plt.close(figura)


def pendiente(parte, error):
    print(f"  pendiente: {parte} ({error})")


def main():
    inicio = time.perf_counter()
    cruda = cargar_serie(ARCHIVO, FECHA, VALOR)
    print(f"Serie: {len(cruda)} meses, de {cruda.index[0]:%Y-%m} a "
          f"{cruda.index[-1]:%Y-%m}, {cruda.isna().sum()} vacíos")
    serie = preparar(cruda)
    graficar_serie(serie)

    print("1. Línea base y modelo a superar")
    filas = [fila("estacional ingenuo", serie,
                  lambda t, h: estacional_ingenuo(t, h, PERIODO))]
    elegidos_para_figura = {}
    try:
        clasico = modelo_clasico(serie)
        filas.append(fila(NOMBRE_CLASICO, serie, clasico))
        elegidos_para_figura[NOMBRE_CLASICO] = clasico
    except NotImplementedError as error:
        pendiente("parte 1", error)

    print("2. Tabla supervisada")
    try:
        comprobar_sin_fuga(serie)
        tabla_lista = True
    except NotImplementedError as error:
        pendiente("parte 2", error)
        tabla_lista = False

    if tabla_lista:
        print("3. Aprendizaje automático")
        try:
            elegidos = ajustar_hiperparametros(serie)
            graficar_importancia(serie, elegidos)
            modelos_aa = {nombre: con_tabla(lambda f=fabrica, p=elegidos[nombre]: f(**p))
                          for nombre, (fabrica, _) in REJILLAS.items()}
            filas_aa = [fila(nombre, serie, modelo) for nombre, modelo in modelos_aa.items()]
            filas += filas_aa
            mejor_aa = min(filas_aa, key=lambda f: f["RMSE"])["modelo"]
            elegidos_para_figura[mejor_aa] = modelos_aa[mejor_aa]
        except NotImplementedError as error:
            pendiente("parte 3", error)

        print("4. Aprendizaje profundo")
        try:
            epocas = validar_redes(serie)
            filas_red = [fila(f"{nombre}, {e} épocas", serie, con_tabla(red(nombre, e)))
                         for nombre, e in epocas.items()]
            filas += filas_red
            mejor = min(zip(epocas, filas_red), key=lambda par: par[1]["RMSE"])[0]
            elegidos_para_figura[mejor] = con_tabla(red(mejor, epocas[mejor]))
            repetir_con_semillas(serie, mejor, epocas[mejor])
        except NotImplementedError as error:
            pendiente("parte 4", error)

    print("5. La tabla")
    tabla = pd.DataFrame(filas).sort_values("RMSE")
    tabla.to_csv(RESULTADOS / "metricas.csv", index=False)
    print(tabla.to_string(index=False))
    print("Mejor modelo por RMSE:", tabla.iloc[0]["modelo"])
    if len(elegidos_para_figura) > 1:
        graficar_pronosticos(serie, elegidos_para_figura)
    print(f"Tiempo total: {time.perf_counter() - inicio:.0f} s")


if __name__ == "__main__":
    main()
