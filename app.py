
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import sympy as sp

# =====================================================
# 1. CONFIGURACION DE LA APLICACION
# =====================================================

st.set_page_config(
    page_title="Ecuaciones Diferenciales",
    page_icon="📈",
    layout="wide"
)

st.title("Ecuaciones Diferenciales - Método Gráfico")
st.write(
    "Calcula el campo de pendientes, la matriz de valores "
    "y las curvas aproximadas mediante el método de Euler."
)

# Variables matematicas
x, y = sp.symbols("x y")

# Funciones matematicas permitidas
funciones = {
    "x": x,
    "y": y,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "pi": sp.pi,
    "E": sp.E
}

nombres_permitidos = set(funciones.keys())


# =====================================================
# 2. INTERPRETAR LA ECUACION
# =====================================================

def interpretar_ecuacion(expresion):
    """Convierte una expresion matematica en una funcion."""

    expresion = expresion.strip().replace("^", "**")

    # Permitir solamente caracteres matematicos basicos
    if not re.fullmatch(
        r"[0-9A-Za-z_+\-*/().,\s]+",
        expresion
    ):
        raise ValueError(
            "La ecuacion contiene caracteres no permitidos."
        )

    # Comprobar los nombres utilizados
    nombres = set(re.findall(r"[A-Za-z_]+", expresion))

    if not nombres.issubset(nombres_permitidos):
        raise ValueError(
            "Usa solamente x, y, sin, cos, tan, exp, "
            "log, sqrt, pi o E."
        )

    # Convertir la expresion a una funcion numerica
    expresion_simbolica = sp.sympify(
        expresion,
        locals=funciones
    )

    if not expresion_simbolica.free_symbols.issubset({x, y}):
        raise ValueError(
            "La ecuacion solo puede depender de x e y."
        )

    funcion = sp.lambdify(
        (x, y),
        expresion_simbolica,
        modules=["numpy"]
    )

    return expresion_simbolica, funcion


# =====================================================
# 3. METODO DE EULER
# =====================================================

def calcular_euler(f, x0, y0, limite, h, direccion):
    """
    Aproxima la solucion desde el punto inicial.
    direccion = 1: hacia la derecha.
    direccion = -1: hacia la izquierda.
    """

    xs = [x0]
    ys = [y0]

    xn = x0
    yn = y0

    for _ in range(20000):
        distancia = limite - xn

        if direccion * distancia <= 1e-10:
            break

        # Ajustar el ultimo paso al limite del rango
        paso = direccion * min(h, abs(distancia))

        pendiente = float(f(xn, yn))

        if not np.isfinite(pendiente):
            break

        y_siguiente = yn + paso * pendiente
        x_siguiente = xn + paso

        if not np.isfinite(y_siguiente):
            break

        if abs(y_siguiente) > 1e8:
            break

        xn = x_siguiente
        yn = y_siguiente

        xs.append(xn)
        ys.append(yn)

    return np.array(xs), np.array(ys)


# =====================================================
# 4. PARAMETROS DE ENTRADA
# =====================================================

st.sidebar.header("Parametros de la ecuacion")

ecuacion = st.sidebar.text_input(
    "Ecuacion dy/dx =",
    value="x - 5*y"
)

st.sidebar.subheader("Condicion inicial")

x0 = st.sidebar.number_input(
    "Valor inicial x₀",
    value=2.0,
    step=0.5
)

y0 = st.sidebar.number_input(
    "Valor inicial y₀",
    value=4.0,
    step=0.5
)

st.sidebar.subheader("Rango de los ejes")

col_a, col_b = st.sidebar.columns(2)

with col_a:
    xmin = st.number_input("x minimo", value=-1.0)
    ymin = st.number_input("y minimo", value=-1.0)

with col_b:
    xmax = st.number_input("x maximo", value=5.0)
    ymax = st.number_input("y maximo", value=8.0)

h = st.sidebar.number_input(
    "Tamano del paso de Euler (h)",
    min_value=0.001,
    max_value=1.0,
    value=0.1,
    step=0.05,
    format="%.3f"
)

separacion = st.sidebar.number_input(
    "Separacion de la matriz",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.5
)

st.sidebar.caption(
    "La separacion controla los puntos del campo. "
    "El paso h controla la precision de Euler."
)


# =====================================================
# 5. VALIDAR Y CALCULAR
# =====================================================

if st.sidebar.button("Graficar y calcular", type="primary"):
    st.session_state["calcular"] = True

if "calcular" not in st.session_state:
    st.info(
        "Configura los parametros y pulsa "
        "'Graficar y calcular' para comenzar."
    )
    st.stop()

try:
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(
            "El valor minimo debe ser menor que el maximo."
        )

    if not (xmin <= x0 <= xmax):
        raise ValueError(
            "El valor x₀ debe estar dentro del rango de x."
        )

    if not (ymin <= y0 <= ymax):
        raise ValueError(
            "El valor y₀ debe estar dentro del rango de y."
        )

    expresion, f = interpretar_ecuacion(ecuacion)

    pendiente_inicial = float(f(x0, y0))

    if not np.isfinite(pendiente_inicial):
        raise ValueError(
            "La pendiente inicial no es un numero finito."
        )

except Exception as error:
    st.error(f"No se pudo calcular: {error}")
    st.stop()


# =====================================================
# 6. CALCULAR LA MATRIZ DEL CAMPO DE PENDIENTES
# =====================================================

def crear_rango(minimo, maximo, paso):
    valores = np.arange(
        minimo,
        maximo + paso * 1e-8,
        paso
    )

    # Incluir el extremo si no coincide con la separacion
    if len(valores) == 0 or valores[-1] < maximo - 1e-8:
        valores = np.append(valores, maximo)

    return valores


x_valores = crear_rango(xmin, xmax, separacion)
y_valores = crear_rango(ymin, ymax, separacion)

# Limitar el tamano de la matriz
if len(x_valores) * len(y_valores) > 10000:
    st.error(
        "La matriz es demasiado grande. "
        "Aumenta la separacion de la matriz."
    )
    st.stop()

X, Y = np.meshgrid(x_valores, y_valores)

with np.errstate(all="ignore"):
    M = np.asarray(f(X, Y), dtype=float)

# Reemplazar valores invalidos para poder graficar
M = np.where(np.isfinite(M), M, np.nan)

# Matriz tabular: filas = y, columnas = x
matriz_df = pd.DataFrame(
    M,
    index=np.round(y_valores, 6),
    columns=np.round(x_valores, 6)
)

matriz_df.index.name = "y / x"


# =====================================================
# 7. CALCULAR LA CURVA CON EULER
# =====================================================

# Desde el punto inicial hacia la derecha
x_derecha, y_derecha = calcular_euler(
    f, x0, y0, xmax, h, 1
)

# Desde el punto inicial hacia la izquierda
x_izquierda, y_izquierda = calcular_euler(
    f, x0, y0, xmin, h, -1
)

# Unir ambas partes sin repetir el punto inicial
x_curva = np.concatenate((
    x_izquierda[::-1],
    x_derecha[1:]
))

y_curva = np.concatenate((
    y_izquierda[::-1],
    y_derecha[1:]
))

# Tabla de iteraciones hacia la derecha
pendientes_euler = np.asarray(
    f(x_derecha, y_derecha),
    dtype=float
)

tabla_euler = pd.DataFrame({
    "Iteracion": np.arange(len(x_derecha)),
    "x": x_derecha,
    "y": y_derecha,
    "f(x,y)": pendientes_euler,
    "y siguiente": np.append(
        y_derecha[1:],
        np.nan
    )
})

tabla_euler = tabla_euler.round(6)


# =====================================================
# 8. MOSTRAR EL RESUMEN
# =====================================================

st.subheader("Resumen del resultado")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Pendiente inicial",
    f"{pendiente_inicial:.4f}"
)

c2.metric(
    "Punto inicial",
    f"({x0:g}, {y0:g})"
)

c3.metric(
    "Paso de Euler",
    f"{h:g}"
)

st.latex(
    rf"\frac{{dy}}{{dx}}={sp.latex(expresion)}"
)

st.caption(
    "La curva se aproxima numericamente mediante Euler; "
    "no representa necesariamente la solucion exacta."
)


# =====================================================
# 9. GRAFICAR EL CAMPO Y LA CURVA
# =====================================================

st.subheader("Campo de pendientes y curva de solucion")

# Normalizar los segmentos para que tengan longitud similar
U = 1 / np.sqrt(1 + np.nan_to_num(M, nan=0.0) ** 2)
V = np.nan_to_num(M, nan=0.0) / np.sqrt(
    1 + np.nan_to_num(M, nan=0.0) ** 2
)

# Evitar segmentos en posiciones con pendientes invalidas
U = np.where(np.isfinite(M), U, np.nan)
V = np.where(np.isfinite(M), V, np.nan)

fig, ax = plt.subplots(figsize=(11, 6))

ax.quiver(
    X, Y, U, V,
    M,
    cmap="coolwarm",
    pivot="mid",
    angles="xy",
    scale_units="xy",
    scale=2.5,
    alpha=0.8
)

ax.plot(
    x_curva,
    y_curva,
    color="blue",
    linewidth=2.5,
    label="Curva aproximada (Euler)"
)

ax.scatter(
    [x0], [y0],
    color="red",
    s=70,
    zorder=5,
    label=f"Condicion inicial ({x0:g}, {y0:g})"
)

ax.annotate(
    f"({x0:g}, {y0:g})",
    (x0, y0),
    xytext=(8, 8),
    textcoords="offset points"
)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Campo de pendientes y solucion aproximada")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)


# =====================================================
# 10. MOSTRAR LAS MATRICES Y LA TABLA DE EULER
# =====================================================

st.subheader("Matriz de pendientes")

st.write(
    "Cada celda contiene el valor de f(x,y) "
    "en la coordenada correspondiente."
)

st.dataframe(
    matriz_df.style.format("{:.3f}"),
    use_container_width=True,
    height=350
)

st.subheader("Matriz de valores: metodo de Euler")

st.write(
    "La columna 'y siguiente' contiene el valor "
    "calculado para el siguiente paso."
)

st.dataframe(
    tabla_euler,
    use_container_width=True,
    height=350
)

# Descargar los resultados
col1, col2 = st.columns(2)

with col1:
    st.download_button(
        "Descargar matriz de pendientes (CSV)",
        data=matriz_df.to_csv().encode("utf-8"),
        file_name="matriz_pendientes.csv",
        mime="text/csv"
    )

with col2:
    st.download_button(
        "Descargar tabla de Euler (CSV)",
        data=tabla_euler.to_csv(index=False).encode("utf-8"),
        file_name="tabla_euler.csv",
        mime="text/csv"
    )

st.success("Calculo completado.")
