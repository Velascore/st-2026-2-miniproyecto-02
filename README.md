# Miniproyecto 2: aprendizaje automático y profundo contra el mejor modelo clásico

Pronóstico de Series Temporales con IA · Universidad Autónoma de Occidente,
2026-II · Carlos Andrés Delgado Saavedra

El grupo vuelve a su serie y a su mejor modelo clásico, el del miniproyecto
1, y lo enfrenta con la serie convertida en tabla supervisada, cuatro
modelos de aprendizaje automático (KNN, SVR, Random Forest y XGBoost) y tres
redes neuronales en PyTorch. Todos pasan por el mismo *backtest* y terminan
en una sola tabla. El enunciado completo, con la rúbrica, está publicado en
el campus. Este repositorio es el punto de partida del grupo.

## El enfoque

Como en el miniproyecto 1, se evalúa el análisis y no el número más bajo. Una
red que pierde con el SARIMA, explicada con un dato de la serie, cumple la
parte. Se evalúa que el grupo use las herramientas del curso sobre sus datos,
tome decisiones y las sostenga con sus cifras.

La serie y los grupos son los del miniproyecto 1. Del miniproyecto 1 se
traen el CSV con su ficha en `datos/README.md`, la función `preparar` y la
función del mejor modelo de su tabla, que vuelve a pasar por el *backtest*
como fila de referencia.

## Integrantes

Los grupos son de hasta 3 integrantes; también se puede trabajar de forma individual.

Esta tabla identifica al grupo: quien no aparezca aquí no cuenta como parte
del grupo. Si falta algún dato, se advierte en la realimentación.

| Nombre completo | Código | Correo institucional |
|---|---|---|
| Camilo Andrés Velasco | 1144179583 | camilo_and.velasco@uao.edu.co |

## Entrega

| Qué | Fecha y hora |
|---|---|
| Cierre de la entrega | viernes 27 de noviembre de 2026, 6:29 p. m. |
| Sustentación | viernes 27 (después del quiz 2) y sábado 28 (8:00 a. m.) de noviembre de 2026, virtual |

## Cómo se entrega

1. **Se trabaja sobre un fork de este repositorio.** El grupo hace el fork,
   lo deja público, agrega a los demás integrantes como colaboradores y
   cada uno trabaja con su propia cuenta, en ramas. Una entrega que no sea
   fork de este repositorio pierde el 30 % de la nota.
2. **Un solo integrante pega el enlace del fork en la tarea *Miniproyecto 2*
   del campus**, y esa entrega vale para todos los que aparecen en la tabla
   de integrantes. Si un grupo entrega más de una vez, se califica la de
   menor nota. Nada más: ni archivos, ni comprimidos, ni código por correo.
   El grupo que no registre el enlace dentro del plazo tiene 0.0.
3. **Se califica el último commit anterior a la hora de cierre.** Lo que se
   suba después no se tiene en cuenta.
4. **El informe es `docs/informe.md`, en Markdown**, con las fórmulas en
   LaTeX y las figuras que genera el guion en `figuras/`. No se aceptan PDF,
   DOCX, presentaciones ni el informe como enlace a otro sitio: lo que no
   está en `docs/informe.md` no se califica.
5. Cada integrante trabaja con su propia cuenta para que el historial de
   commits muestre el trabajo del grupo.

## Cómo está organizado

```
miniproyecto.py      el guion del grupo, en cinco partes con TODO
comun.py             carga, métricas, líneas base y backtest; no se modifica
datos/               la serie del grupo y su ficha (datos/README.md)
docs/informe.md      la plantilla del informe; se llena en el mismo archivo
figuras/             las figuras que genera el guion
resultados/          metricas.csv y semillas.csv, las tablas que genera el guion
```

`comun.py` es el mismo del miniproyecto 1, igual para todos los grupos: el
mismo *backtest* (origen móvil sobre los últimos 24 meses, horizonte de 1 a
12) y las mismas métricas (RMSE, MAE y MAPE). Si se modifica, la parte de la
tabla y la elección vale 0.

`miniproyecto.py` trae la serie de ejemplo y el estacional ingenuo, y corre
desde el primer día. También trae escritos el paso de la tabla supervisada
al *backtest* (`con_tabla`), las fábricas de KNN, SVR, Random Forest y
XGBoost, la clase `Red`, que entrena cualquier red de PyTorch con el ciclo
de entrenamiento del curso, un MLP de ejemplo y el conteo de parámetros. Las
partes marcadas con `TODO` las completa el grupo; mientras falten, el guion
las reporta como pendientes y sigue.

El guion completo tiene que correr en CPU en menos de 10 minutos en la
acción de GitHub: redes pequeñas, de unas decenas de neuronas, y pocas
épocas. Solo se usan `numpy`, `pandas`, `matplotlib`, `statsmodels`,
`scikit-learn`, `xgboost` y `torch`.

## Cómo se corre

Con el entorno `series` del curso activo, PyTorch para CPU se instala
primero, para que `pip` no baje la versión con CUDA, que pesa unos 2,5 GB:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python miniproyecto.py
```

En la sustentación, un integrante elegido en el momento comparte pantalla y
corre el guion. La acción de GitHub del repositorio lo corre en cada *push*:
si queda en rojo, el guion no corre de principio a fin.
