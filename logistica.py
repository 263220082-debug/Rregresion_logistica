"""Genera un dataset sintético, el análisis de regresión logística y el informe DOCX.

Dependencias: numpy, pandas, matplotlib, statsmodels, scikit-learn y python-docx.
Ejecucion: python generar_entregable.py
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
SEED = 20261120
N = 2400
ALPHA = 0.05
TARGET = "incumplimiento"
FEATURES = [
    "ingreso_mensual",
    "razon_deuda",
    "utilizacion_credito",
    "atrasos_12m",
    "antiguedad_credito",
    "consultas_6m",
    "ahorro_mensual",
    "estabilidad_empleo",
    "pagos_sobre_ingreso",
    "edad",
]

VARIABLES = {
    "ingreso_mensual": ("Ingreso mensual", "Pesos mexicanos", "Ingreso mensual neto simulado."),
    "razon_deuda": ("Razón de deuda", "Proporción (0-1)", "Deuda total respecto del límite de crédito."),
    "utilizacion_credito": ("Utilización de crédito", "Proporción (0-1)", "Porcentaje del crédito disponible que se utiliza."),
    "atrasos_12m": ("Atrasos en 12 meses", "Número de atrasos", "Pagos atrasados registrados en el último año."),
    "antiguedad_credito": ("Antigüedad crediticia", "Años", "Tiempo desde la apertura de la primera cuenta."),
    "consultas_6m": ("Consultas recientes", "Número de consultas", "Consultas de crédito realizadas en seis meses."),
    "ahorro_mensual": ("Ahorro mensual", "Pesos mexicanos", "Ahorro promedio mensual simulado."),
    "estabilidad_empleo": ("Estabilidad laboral", "Anos", "Antiguedad en el empleo actual."),
    "pagos_sobre_ingreso": ("Pagos / ingreso", "Proporción (0-1)", "Obligaciones mensuales de crédito respecto del ingreso."),
    "edad": ("Edad", "Anos", "Edad de la persona solicitante."),
}

COLORS = {
    "navy": "17365D",
    "blue": "2F75B5",
    "pale": "D9EAF7",
    "light": "F3F6F9",
    "gray": "566573",
    "green": "548235",
}


def build_dataset() -> pd.DataFrame:
    """Crea registros ficticios reproducibles; no corresponden a personas reales."""
    rng = np.random.default_rng(SEED)
    z = rng.normal(size=(N, len(FEATURES)))
    raw = pd.DataFrame(
        {
            "ingreso_mensual": np.clip(72000 + 21000 * z[:, 0], 12000, None),
            "razon_deuda": np.clip(0.35 + 0.11 * z[:, 1], 0.02, 0.90),
            "utilizacion_credito": np.clip(0.45 + 0.19 * z[:, 2], 0.01, 0.99),
            "atrasos_12m": np.clip(np.rint(1.5 + 1.2 * z[:, 3]), 0, 6),
            "antiguedad_credito": np.clip(7 + 2.5 * z[:, 4], 0.5, 20),
            "consultas_6m": np.clip(np.rint(2.2 + 1.1 * z[:, 5]), 0, 7),
            "ahorro_mensual": np.clip(22000 + 13000 * z[:, 6], 0, None),
            "estabilidad_empleo": np.clip(5 + 2.2 * z[:, 7], 0.5, 20),
            "pagos_sobre_ingreso": np.clip(0.28 + 0.095 * z[:, 8], 0.01, 0.80),
            "edad": np.clip(38 + 10 * z[:, 9], 18, 75),
        }
    )
    latent_beta = np.array([-0.20, 0.65, 0.70, 0.85, -0.35, 0.28, -0.45, -0.45, 0.40, -0.10])
    probabilities = expit(-1.8 + z @ latent_beta)
    raw[TARGET] = rng.binomial(1, probabilities)
    raw.insert(0, "registro_id", np.arange(1, N + 1))
    for column in FEATURES:
        if column in {"atrasos_12m", "consultas_6m"}:
            raw[column] = raw[column].astype(int)
    raw[TARGET] = raw[TARGET].astype(int)
    return raw


def fit_logit(frame: pd.DataFrame, features: list[str]):
    x = sm.add_constant(frame[features], has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sm.Logit(frame[TARGET], x).fit(disp=False, maxiter=200)


def assess_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    probabilities = model.predict(sm.add_constant(x_test, has_constant="add"))
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }


def subset_search(train: pd.DataFrame, features: list[str]) -> tuple[dict, dict, list[dict]]:
    rows = []
    for size in range(1, 5):
        for subset in combinations(features, size):
            result = fit_logit(train, list(subset))
            rows.append(
                {
                    "features": list(subset),
                    "n_features": size,
                    "llf": float(result.llf),
                    "llnull": float(result.llnull),
                    "llr": float(result.llr),
                    "llr_pvalue": float(result.llr_pvalue),
                    "prsquared": float(result.prsquared),
                    "aic": float(result.aic),
                }
            )
    by_llr = min(rows, key=lambda row: (row["llr_pvalue"], -row["prsquared"], row["n_features"]))
    by_r2 = max(rows, key=lambda row: (row["prsquared"], -row["llr_pvalue"], -row["n_features"]))
    return by_llr, by_r2, rows


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    tc_pr.append(shading)


def set_cell_text(cell, text: str, *, bold: bool = False, color: str | None = None, size: float = 8.5) -> None:
    cell.text = ""
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run(str(text))
    run.bold = bold
    run.font.name = "Aptos"
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], font_size: float = 8.5):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = True
    for idx, header in enumerate(headers):
        set_cell_text(table.rows[0].cells[idx], header, bold=True, color="FFFFFF", size=font_size)
        shade_cell(table.rows[0].cells[idx], COLORS["navy"])
    for row_index, values in enumerate(rows):
        cells = table.add_row().cells
        for idx, value in enumerate(values):
            set_cell_text(cells[idx], value, size=font_size)
            if row_index % 2 == 1:
                shade_cell(cells[idx], COLORS["light"])
    return table


def add_caption(doc: Document, label: str, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(5)
    run = paragraph.add_run(f"{label}. {text}")
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor.from_string(COLORS["gray"])


def add_code_block(doc: Document, code: str) -> None:
    for line in code.strip("\n").splitlines():
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.left_indent = Inches(0.18)
        paragraph.paragraph_format.space_after = Pt(0)
        run = paragraph.add_run(line or " ")
        run.font.name = "Consolas"
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor.from_string(COLORS["navy"])


def add_hyperlink(paragraph, text: str, target: str) -> None:
    relationship_id = paragraph.part.relate_to(
        target,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relationship_id)
    run = OxmlElement("w:r")
    properties = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), COLORS["blue"])
    properties.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    properties.append(underline)
    run.append(properties)
    text_node = OxmlElement("w:t")
    text_node.text = text
    run.append(text_node)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("Pagina ")
    run.font.size = Pt(8)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    paragraph._p.append(field)


def make_figures(data: pd.DataFrame, test: pd.DataFrame, selected_features: list[str], selected_model) -> dict[str, Path]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    files = {
        "prevalence": ROOT / "figura_01_frecuencia_incumplimiento.png",
        "logistic": ROOT / "figura_02_curva_logistica.png",
        "confusion": ROOT / "figura_03_matriz_confusion.png",
    }

    counts = data[TARGET].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(7.1, 4.1))
    bars = ax.bar(["Sin incumplimiento (0)", "Incumplimiento (1)"], counts.values, color=["#2F75B5", "#C0504D"])
    ax.set_title("Frecuencia de la variable dependiente")
    ax.set_ylabel("Número de registros")
    ax.grid(axis="y", alpha=0.25)
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(files["prevalence"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    coef = selected_model.params.drop("const")
    best_feature = str(coef.abs().idxmax())
    observed = test[best_feature].to_numpy()
    labels = test[TARGET].to_numpy()
    jitter_rng = np.random.default_rng(SEED + 1)
    jittered = labels + jitter_rng.uniform(-0.035, 0.035, size=len(labels))
    sample_idx = np.arange(len(observed))
    if len(sample_idx) > 500:
        sample_idx = jitter_rng.choice(sample_idx, size=500, replace=False)
    grid = np.linspace(
        max(float(test[best_feature].min()), float(np.quantile(observed, 0.01))),
        min(float(test[best_feature].max()), float(np.quantile(observed, 0.99))),
        250,
    )
    curve = pd.DataFrame(0.0, index=np.arange(len(grid)), columns=selected_features)
    curve[best_feature] = grid
    curve_probability = selected_model.predict(sm.add_constant(curve, has_constant="add"))
    fig, ax = plt.subplots(figsize=(7.1, 4.4))
    ax.scatter(observed[sample_idx], jittered[sample_idx], s=18, alpha=0.27, color="#7F8C8D", label="Observaciones de prueba")
    ax.plot(grid, curve_probability, color="#C0504D", linewidth=2.4, label="Probabilidad estimada")
    ax.set_title("Curva logística y observaciones (con otras variables en su media)")
    ax.set_xlabel(f"{VARIABLES[best_feature][0]} estandarizada (z)")
    ax.set_ylabel("Probabilidad / resultado observado")
    ax.set_ylim(-0.08, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(alpha=0.22)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(files["logistic"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    evaluation = assess_model(selected_model, test[selected_features], test[TARGET])
    matrix = evaluation["matrix"]
    fig, ax = plt.subplots(figsize=(5.6, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Matriz de confusion: modelo de seleccion LLR")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Valor real")
    ax.set_xticks([0, 1], ["No incumple", "Incumple"])
    ax.set_yticks([0, 1], ["No incumple", "Incumple"])
    threshold = matrix.max() / 2
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:,}", ha="center", va="center", color="white" if value > threshold else "#17365D", fontsize=12)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Registros")
    fig.tight_layout()
    fig.savefig(files["confusion"], dpi=180, bbox_inches="tight")
    plt.close(fig)
    return files


def add_picture(doc: Document, path: Path, width: float = 6.35) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(path), width=Inches(width))


def create_report(
    data: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    full_model,
    llr_model,
    r2_model,
    llr_choice: dict,
    r2_choice: dict,
    full_eval: dict,
    llr_eval: dict,
    figures: dict[str, Path],
) -> Path:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.72)
    section.left_margin = Inches(0.82)
    section.right_margin = Inches(0.82)
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    styles = doc.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"].font.size = Pt(10)
    styles["Normal"].paragraph_format.space_after = Pt(6)
    for style_name, size in (("Title", 25), ("Heading 1", 17), ("Heading 2", 13), ("Heading 3", 11)):
        styles[style_name].font.name = "Aptos Display"
        styles[style_name].font.size = Pt(size)
        styles[style_name].font.color.rgb = RGBColor.from_string(COLORS["navy"])
    footer = section.footer.paragraphs[0]
    add_page_number(footer)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(48)
    run = p.add_run("MAESTRIA EN INTELIGENCIA ARTIFICIAL")
    run.bold = True
    run.font.size = Pt(15)
    run.font.color.rgb = RGBColor.from_string(COLORS["blue"])
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(32)
    run = p.add_run("ACTIVIDAD 5")
    run.bold = True
    run.font.size = Pt(30)
    run.font.color.rgb = RGBColor.from_string(COLORS["navy"])
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("ESTADISTICA APLICADA")
    run.bold = True
    run.font.size = Pt(19)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(14)
    run = p.add_run("Modelo de regresión logística para predecir incumplimiento crediticio")
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor.from_string(COLORS["gray"])
    doc.add_paragraph("")
    for text in [
        "Integrantes:",
        "1. [PENDIENTE: nombre completo del alumno 1]",
        "2. [PENDIENTE: nombre completo del alumno 2]",
        "Docente: [PENDIENTE]",
        "Grupo: [PENDIENTE]",
        "Periodo: septiembre-diciembre de 2026",
        "Fecha de entrega: [PENDIENTE]",
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(text).font.size = Pt(11)
    doc.add_page_break()

    doc.add_heading("Resumen ejecutivo", level=1)
    event_rate = data[TARGET].mean()
    best_full = str(full_model.params.drop("const").abs().idxmax())
    summary = (
        f"Se construyó y analizó un conjunto sintético de {len(data):,} registros, con diez predictores numéricos "
        f"y una respuesta binaria (incumplimiento). No se descargaron datos de Kaggle ni se utilizaron datos "
        f"personales. El modelo completo se estimo con {len(train):,} observaciones de entrenamiento y se evaluo "
        f"en {len(test):,} observaciones reservadas. La proporción de incumplimiento simulada fue {event_rate:.1%}. "
        f"En el modelo completo, la variable con mayor |z| fue {VARIABLES[best_full][0].lower()}; el modelo de "
        f"hasta cuatro variables elegido por p-value LLR obtuvo exactitud de {llr_eval['accuracy']:.1%} en prueba."
    )
    doc.add_paragraph(summary)
    doc.add_paragraph(
        "Importante: los resultados son demostrativos y reproducibles, no estimaciones sobre la población real. "
        "La semilla fija permite regenerar exactamente los mismos registros y resultados."
    )
    add_caption(doc, "Figura 1", "Distribución de la variable dependiente en el dataset sintético.")
    add_picture(doc, figures["prevalence"], 5.7)

    doc.add_heading("Objetivo y plan de trabajo", level=1)
    doc.add_paragraph(
        "Aplicar regresión logística binaria con Python para modelar la probabilidad de incumplimiento, "
        "interpretar los estadisticos del ajuste, comparar especificaciones con un maximo de cuatro predictores, "
        "y evaluar la clasificacion en datos que no participaron en el ajuste."
    )
    doc.add_paragraph(
        "Se reserva el 25% para prueba mediante particion estratificada. El 75% restante se usa para estimar la "
        "escala de los predictores, ajustar el modelo completo y realizar la busqueda de subconjuntos. El umbral "
        "de clasificación es 0.50 y el nivel de significancia fijado para las pruebas individuales es alpha=0.05."
    )

    doc.add_heading("Parte I. Plataforma y datos", level=1)
    doc.add_heading("I.1 Analisis de Kaggle", level=2)
    doc.add_paragraph(
        "Kaggle es una plataforma en linea de ciencia de datos e inteligencia artificial que presenta competencias, "
        "benchmarks, comunidades y recursos para que sus usuarios compartan y evalúen soluciones. Su pagina principal "
        "la describe como un espacio para evaluar modelos y tecnologia mediante competencias y actividades colaborativas "
        "[1]. Para este ejercicio se tomo Kaggle como referencia de la plataforma, pero se decidio no descargar un "
        "dataset: los registros de este informe se generaron mediante un proceso sintético documentado y reproducible."
    )
    doc.add_heading("I.2 Dataset de interes", level=2)
    doc.add_paragraph(
        f"El dataset se denomina “Perfil crediticio simulado”. Contiene {len(data):,} registros, diez variables "
        f"independientes numericas y una dependiente binaria: {TARGET}. La clase 1 representa un incumplimiento "
        f"simulado y la clase 0 su ausencia. La semilla aleatoria es {SEED}; no hay sujetos reales ni datos observados "
        "de una entidad financiera. El archivo CSV asociado es datos_credito_sinteticos.csv."
    )
    doc.add_paragraph(
        "La variable respuesta se generó con una probabilidad logística a partir de factores latentes. Por ello, "
        "el dataset es apropiado para practicar el flujo estadistico, aunque no permite inferir relaciones causales "
        "ni estimar riesgo crediticio real."
    )
    doc.add_heading("I.3 Descripcion de variables", level=2)
    variable_rows = []
    for name in FEATURES:
        label, units, description = VARIABLES[name]
        variable_rows.append([name, label, units, description])
    add_table(doc, ["Nombre", "Descripcion", "Unidad", "Que mide"], variable_rows, 8)
    add_caption(doc, "Tabla 1", "Variables numericas predictoras; registro_id identifica filas y no se utiliza en el modelo.")
    doc.add_paragraph(
        "La estandarizacion se ajusta solo con la particion de entrenamiento: z=(x-media_entrenamiento)/"
        "desviación_estandar_entrenamiento. Esto evita que el conjunto de prueba influya en la estimación. "
        "Por consiguiente, cada coeficiente representa el cambio en log-odds asociado a un aumento de una desviación "
        "estandar del predictor, manteniendo constantes los demas."
    )

    doc.add_heading("Implementacion y evidencia de codigo", level=2)
    doc.add_paragraph(
        "El archivo generar_entregable.py acompana este informe. Crea el CSV, divide los datos, estandariza los "
        "predictores, ajusta los modelos, produce las graficas y genera este documento. Para reproducirlo, instalar "
        "numpy, pandas, scipy, matplotlib, statsmodels, scikit-learn y python-docx; despues ejecutar "
        "`python generar_entregable.py` desde la carpeta que contiene el script."
    )
    code_link = doc.add_paragraph("Enlace al codigo fuente: ")
    add_hyperlink(code_link, "generar_entregable.py", "generar_entregable.py")
    code_link.add_run(" (archivo adjunto en la misma carpeta).")
    data_link = doc.add_paragraph("Enlace al dataset: ")
    add_hyperlink(data_link, "datos_credito_sinteticos.csv", "datos_credito_sinteticos.csv")
    data_link.add_run(" (datos sintéticos generados por el script).")
    doc.add_paragraph("Fragmento 1. Generacion de la respuesta binaria y ajuste del modelo (la semilla esta fijada):")
    add_code_block(
        doc,
        """probabilidades = expit(-1.8 + z @ latent_beta)
raw["incumplimiento"] = rng.binomial(1, probabilidades)

X_train_z = scaler.fit_transform(train[FEATURES])
train_z = pd.DataFrame(X_train_z, columns=FEATURES, index=train.index)
resultado = sm.Logit(train["incumplimiento"],
                     sm.add_constant(train_z)).fit(disp=False)""",
    )
    doc.add_paragraph(
        "La primera expresion transforma un puntaje lineal en una probabilidad entre 0 y 1; la segunda sortea "
        "una respuesta Bernoulli por registro. El ajuste estima los coeficientes por máxima verosimilitud. "
        "El script completo contiene el generador, el preprocesamiento, la busqueda exhaustiva de subconjuntos "
        "y la exportacion de resultados."
    )

    doc.add_heading("Parte II. Regresión logística con todas las variables", level=1)
    doc.add_heading("II.1 Ecuacion estimada e interpretacion", level=2)
    doc.add_paragraph(
        "Sea p=P(Y=1|X), donde Y=1 indica incumplimiento. El modelo logit es "
        "log[p/(1-p)]=beta0+beta1*x1+...+beta10*x10. La probabilidad estimada se obtiene con "
        "p=1/(1+exp(-eta)), donde eta es el lado derecho de la ecuación."
    )
    equation = "log[p/(1-p)] = " + fmt(full_model.params["const"], 4)
    for feature in FEATURES:
        beta = full_model.params[feature]
        equation += f" {'+' if beta >= 0 else '-'} {abs(beta):.4f}·z({feature})"
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.18)
    run = p.add_run(equation)
    run.bold = True
    run.font.name = "Cambria Math"
    run.font.size = Pt(9)
    doc.add_paragraph(
        "La pendiente beta_j es el cambio en el logaritmo de las odds de incumplimiento por un aumento de una "
        "desviación estándar de x_j, con los otros predictores fijos. El signo positivo eleva las odds y el negativo "
        "las reduce. Al exponentiar beta_j se obtiene la razon de momios (odds ratio), exp(beta_j): un valor mayor "
        "que 1 indica incremento multiplicativo de las odds; menor que 1, disminucion. No es un cambio directo en "
        "puntos porcentuales de probabilidad."
    )
    doc.add_paragraph(
        f"En este ajuste, beta0={full_model.params['const']:.4f} representa el log-odds estimado cuando todos los "
        "predictores estandarizados valen cero (sus medias de entrenamiento). Los coeficientes numericos se "
        "presentan en la Tabla 3."
    )

    doc.add_heading("Parte III. Estadisticos de ajuste", level=1)
    doc.add_paragraph(
        "El modelo nulo contiene solo intercepto y el modelo completo incorpora los diez predictores. La "
        "log-verosimilitud se define como LL=sum[y_i ln(p_i)+(1-y_i)ln(1-p_i)]. Un valor menos negativo indica "
        "mayor verosimilitud de los datos bajo el modelo."
    )
    doc.add_paragraph(
        "El pseudo R-cuadrado de McFadden se calcula como 1-LL_modelo/LL_nulo; compara el ajuste relativo con el "
        "modelo de solo intercepto. No es la fraccion de varianza explicada ni tiene la misma escala interpretativa "
        "que el R-cuadrado de regresión lineal."
    )
    doc.add_paragraph(
        "El estadístico LLR es 2(LL_modelo-LL_nulo) y, bajo la hipótesis nula de que todos los coeficientes de "
        "pendiente son cero, se compara con una distribución chi-cuadrada con grados de libertad iguales al número "
        "de predictores. LL-Null es la log-verosimilitud del modelo nulo."
    )
    fit_rows = [
        ["Observaciones de entrenamiento", f"{len(train):,}"],
        ["Log-Likelihood (LL)", fmt(full_model.llf, 4)],
        ["LL-Null", fmt(full_model.llnull, 4)],
        ["LLR", fmt(full_model.llr, 4)],
        ["Grados de libertad del LLR", str(int(full_model.df_model))],
        ["LLR p-value", f"{full_model.llr_pvalue:.4g}"],
        ["Pseudo R-cuadrado de McFadden", fmt(full_model.prsquared, 4)],
        ["AIC (menor es preferible al comparar ajustes)", fmt(full_model.aic, 2)],
    ]
    add_table(doc, ["Estadistico", "Modelo completo"], fit_rows)
    add_caption(doc, "Tabla 2", "Indicadores de ajuste calculados unicamente en la particion de entrenamiento.")
    doc.add_paragraph(
        f"III.1. El pseudo R-cuadrado es {full_model.prsquared:.4f}: el modelo mejora el ajuste frente al nulo "
        f"en esa medida relativa. No se interpreta como {full_model.prsquared:.1%} de la variabilidad explicada."
    )
    doc.add_paragraph(
        f"III.2. LL={full_model.llf:.4f}. Al ser menos negativo que LL-Null, el modelo completo asigna mayor "
        "verosimilitud a las respuestas observadas; por sí solo, LL no es una medida porcentual de precisión."
    )
    doc.add_paragraph(
        f"III.3. LL-Null={full_model.llnull:.4f}; es el valor de referencia de un modelo que solo conoce la "
        "proporción de clase 1 del entrenamiento, sin usar características."
    )
    doc.add_paragraph(
        f"III.4. LLR p-value={full_model.llr_pvalue:.4g}. Como es "
        f"{'menor' if full_model.llr_pvalue < ALPHA else 'mayor'} que alpha={ALPHA:.2f}, se "
        f"{'rechaza' if full_model.llr_pvalue < ALPHA else 'no se rechaza'} la hipótesis de que todas las "
        "pendientes sean simultaneamente cero; el modelo con predictores "
        f"{'aporta evidencia de mejora global' if full_model.llr_pvalue < ALPHA else 'no muestra evidencia suficiente de mejora global'} "
        "frente al nulo."
    )

    doc.add_heading("Parte IV. Interpretacion de la tabla de coeficientes", level=1)
    doc.add_paragraph(
        "Coef es la pendiente estimada; std err es el error estándar de su estimación; z=coef/std err; p>|z| "
        "contrasta H0: beta=0 con una prueba bilateral; IC 95% es el intervalo de Wald para beta. OR=exp(coef) "
        "facilita comunicar el efecto multiplicativo en odds. Las inferencias son asociativas y dependen de la "
        "especificación y del mecanismo sintético."
    )
    coef_rows = []
    ci = full_model.conf_int(alpha=ALPHA)
    for feature in FEATURES:
        row = full_model.params[feature]
        coef_rows.append(
            [
                feature,
                fmt(row, 4),
                fmt(full_model.bse[feature], 4),
                fmt(full_model.tvalues[feature], 3),
                f"{full_model.pvalues[feature]:.4g}",
                fmt(np.exp(row), 3),
                f"[{ci.loc[feature, 0]:.3f}, {ci.loc[feature, 1]:.3f}]",
            ]
        )
    add_table(doc, ["Variable", "Coef", "Std err", "z", "p>|z|", "OR", "IC 95% coef"], coef_rows, 7.7)
    add_caption(doc, "Tabla 3", "Estimaciones del modelo con diez predictores, entrenado sobre datos estandarizados.")
    doc.add_paragraph("IV.1-2. Interpretacion variable por variable:")
    for feature in FEATURES:
        beta = float(full_model.params[feature])
        se = float(full_model.bse[feature])
        pvalue = float(full_model.pvalues[feature])
        odds_ratio = float(np.exp(beta))
        direction = "aumentan" if beta > 0 else "disminuyen"
        change = (odds_ratio - 1) * 100
        relation = f"aumentan {change:.1f}%" if change >= 0 else f"disminuyen {abs(change):.1f}%"
        evidence = "significativa al 5%" if pvalue < ALPHA else "no significativa al 5%"
        doc.add_paragraph(
            f"{VARIABLES[feature][0]} ({feature}): coef={beta:.4f}; un aumento de una desviación estándar, "
            f"manteniendo lo demás constante, se asocia con odds que {relation} (OR={odds_ratio:.3f}). "
            f"Su error estándar es {se:.4f}, que cuantifica la incertidumbre de la estimación en escala logit; "
            f"la evidencia individual es {evidence} (p={pvalue:.4g})."
        )
    significant = [feature for feature in FEATURES if full_model.pvalues[feature] < ALPHA]
    doc.add_paragraph(
        f"IV.3. Adoptando alpha=0.05 (el enunciado no especifica el valor), se consideran "
        f"estadísticamente asociadas las variables con p<0.05: {', '.join(significant) if significant else 'ninguna'}. "
        "La significancia no demuestra causalidad ni importancia práctica."
    )
    ranked_z = sorted(FEATURES, key=lambda feature: abs(full_model.tvalues[feature]), reverse=True)
    doc.add_paragraph(
        f"IV.4. Por magnitud absoluta de z, las dos variables que ofrecen mayor evidencia individual contra "
        f"coeficiente cero son {VARIABLES[ranked_z[0]][0].lower()} (z={full_model.tvalues[ranked_z[0]]:.3f}) y "
        f"{VARIABLES[ranked_z[1]][0].lower()} (z={full_model.tvalues[ranked_z[1]]:.3f}). El signo indica la "
        "dirección de la asociación; |z| no mide por sí solo importancia causal."
    )
    best_z = ranked_z[0]
    lower, upper = ci.loc[best_z]
    doc.add_paragraph(
        f"IV.5. La variable con mayor |z| es {VARIABLES[best_z][0].lower()}. Su coeficiente tiene IC de Wald "
        f"95% [{lower:.4f}, {upper:.4f}] en escala log-odds por desviación estándar. En escala de OR, el intervalo "
        f"es [{np.exp(lower):.3f}, {np.exp(upper):.3f}]. Si el intervalo del coeficiente no contiene cero (equivale "
        "a que el intervalo de OR no contiene uno), existe evidencia individual al 5% bajo los supuestos del modelo."
    )

    doc.add_heading("Parte V. Clasificacion y modelo segun LLR", level=1)
    doc.add_heading("V.1 Matriz de confusion", level=2)
    full_matrix = full_eval["matrix"]
    llr_matrix = llr_eval["matrix"]
    add_table(
        doc,
        ["Modelo completo (prueba)", "Predice 0", "Predice 1"],
        [
            ["Real 0", f"{full_matrix[0, 0]:,}", f"{full_matrix[0, 1]:,}"],
            ["Real 1", f"{full_matrix[1, 0]:,}", f"{full_matrix[1, 1]:,}"],
        ],
    )
    add_caption(doc, "Tabla 4", "Matriz del modelo completo en la muestra de prueba, umbral 0.50.")
    doc.add_paragraph(
        f"TN={full_matrix[0, 0]:,} (negativos correctamente identificados), FP={full_matrix[0, 1]:,}, "
        f"FN={full_matrix[1, 0]:,} y TP={full_matrix[1, 1]:,}. La clase positiva es incumplimiento."
    )
    doc.add_heading("V.2 Mejor modelo con hasta cuatro variables por p-value LLR", level=2)
    doc.add_paragraph(
        "Se ajustaron exhaustivamente las 385 combinaciones no vacias posibles de uno a cuatro predictores, "
        "usando solo entrenamiento. Se selecciono el menor p-value del LLR global; los desempates se resolvieron "
        "por mayor pseudo R-cuadrado y después por menor número de variables. Este criterio identifica evidencia "
        "de mejora contra el nulo, no necesariamente el mejor desempeno predictivo en prueba."
    )
    doc.add_paragraph(
        f"Variables seleccionadas: {', '.join(llr_choice['features'])}. "
        f"LL={llr_choice['llf']:.4f}; LL-Null={llr_choice['llnull']:.4f}; "
        f"LLR={llr_choice['llr']:.4f}; p-value={llr_choice['llr_pvalue']:.4g}; "
        f"pseudo R-cuadrado={llr_choice['prsquared']:.4f}."
    )
    doc.add_heading("V.3 Curva logística y diagrama de dispersión", level=2)
    best_sub_feature = str(llr_model.params.drop("const").abs().idxmax())
    doc.add_paragraph(
        f"La curva representa la probabilidad predicha por el modelo de V.2 al variar "
        f"{VARIABLES[best_sub_feature][0].lower()}, dejando los otros predictores estandarizados en cero "
        "(sus medias de entrenamiento). Los puntos muestran respuestas binarias de la particion de prueba, "
        "con una pequena dispersion vertical solo para facilitar su lectura. Por ser multiples predictores, "
        "la curva es una visualizacion condicional, no un ajuste univariado."
    )
    add_caption(doc, "Figura 2", "Probabilidad condicional predicha y resultados observados para el predictor con mayor |coef| en el modelo LLR.")
    add_picture(doc, figures["logistic"], 6.05)
    add_caption(doc, "Figura 3", "Matriz de confusion del modelo de hasta cuatro predictores seleccionado por LLR.")
    add_picture(doc, figures["confusion"], 4.8)
    add_table(
        doc,
        ["Modelo LLR (prueba)", "Predice 0", "Predice 1"],
        [
            ["Real 0", f"{llr_matrix[0, 0]:,}", f"{llr_matrix[0, 1]:,}"],
            ["Real 1", f"{llr_matrix[1, 0]:,}", f"{llr_matrix[1, 1]:,}"],
        ],
    )
    add_caption(doc, "Tabla 5", "Conteos de clasificacion en prueba para el modelo seleccionado por p-value LLR.")
    doc.add_heading("V.4 Precision del modelo", level=2)
    doc.add_paragraph(
        f"Para el modelo elegido por LLR, la exactitud (accuracy) en prueba es {llr_eval['accuracy']:.1%}; "
        f"la precision positiva es {llr_eval['precision']:.1%}, la sensibilidad/recall es {llr_eval['recall']:.1%} "
        f"y F1 es {llr_eval['f1']:.1%}. Por tanto, si “precisión” se usa como porcentaje global de aciertos, "
        f"la respuesta es {llr_eval['accuracy']:.1%}; la métrica llamada precisión en clasificación es "
        f"{llr_eval['precision']:.1%}. Se reportan ambas porque no son sinónimos. La exactitud del modelo completo "
        f"es {full_eval['accuracy']:.1%}. La línea base que siempre predice la clase mayoritaria alcanza "
        f"{1 - test[TARGET].mean():.1%}; el modelo LLR la supera por "
        f"{llr_eval['accuracy'] - (1 - test[TARGET].mean()):.1%}. Aun así, su recall de "
        f"{llr_eval['recall']:.1%} muestra que omite una parte considerable de los incumplimientos. "
        "La evaluación usa datos reservados y umbral 0.50."
    )

    doc.add_heading("Parte VI. Mejor modelo segun pseudo R-cuadrado", level=1)
    doc.add_paragraph(
        "Se reutilizo la busqueda exhaustiva de modelos con uno a cuatro predictores y se eligio el mayor pseudo "
        "R-cuadrado de McFadden en entrenamiento. En empates se prefirio menor p-value LLR y luego menos variables."
    )
    doc.add_paragraph(
        f"VI.1. Variables seleccionadas: {', '.join(r2_choice['features'])}. "
        f"Pseudo R-cuadrado={r2_choice['prsquared']:.4f}; LL={r2_choice['llf']:.4f}; "
        f"LL-Null={r2_choice['llnull']:.4f}; LLR={r2_choice['llr']:.4f}; "
        f"p-value LLR={r2_choice['llr_pvalue']:.4g}."
    )
    doc.add_paragraph(
        f"El pseudo R-cuadrado de {r2_choice['prsquared']:.4f} expresa una mejora relativa respecto del modelo "
        "nulo para esta muestra de entrenamiento; no se debe leer como porcentaje de variacion explicada. "
        f"El conjunto de variables {'coincide' if set(r2_choice['features']) == set(llr_choice['features']) else 'difiere'} "
        "del seleccionado por p-value LLR porque ambos criterios optimizan aspectos distintos."
    )
    comparison_rows = [
        [
            "Completo (10)",
            fmt(full_model.prsquared, 4),
            fmt(full_model.llr, 3),
            f"{full_model.llr_pvalue:.3g}",
            f"{full_eval['accuracy']:.3f}",
        ],
        [
            "Maximo LLR (<=4)",
            fmt(llr_choice["prsquared"], 4),
            fmt(llr_choice["llr"], 3),
            f"{llr_choice['llr_pvalue']:.3g}",
            f"{llr_eval['accuracy']:.3f}",
        ],
        [
            "Maximo pseudo R2 (<=4)",
            fmt(r2_choice["prsquared"], 4),
            fmt(r2_choice["llr"], 3),
            f"{r2_choice['llr_pvalue']:.3g}",
            f"{assess_model(r2_model, test[r2_choice['features']], test[TARGET])['accuracy']:.3f}",
        ],
    ]
    add_table(doc, ["Especificacion", "Pseudo R2", "LLR", "p-value LLR", "Accuracy prueba"], comparison_rows)
    add_caption(doc, "Tabla 6", "Comparacion de indicadores en entrenamiento y exactitud independiente en prueba.")

    doc.add_heading("Conclusiones", level=1)
    doc.add_paragraph(
        "Se cumplieron los incisos de descripción del dataset, estimación e interpretación del modelo logístico, "
        "evaluacion de estadisticos globales e individuales, matrices de confusion, visualizacion y comparacion de "
        "modelos limitados a cuatro variables. La estandarización hace comparables los efectos por desviación "
        "estandar, mientras que los intervalos y pruebas reflejan incertidumbre bajo el modelo."
    )
    doc.add_paragraph(
        "Los hallazgos solo describen el mecanismo sintético fijado en el script. Un uso aplicado requeriría una "
        "muestra real autorizada, definicion operativa del evento, tratamiento de datos faltantes, validacion temporal "
        "y externa, calibracion, analisis de costos de errores y revision de equidad. La exactitud debe leerse junto "
        "con precisión, sensibilidad y la matriz de confusión, especialmente si las clases están desbalanceadas."
    )
    doc.add_heading("Archivos de entrega", level=1)
    doc.add_paragraph(
        "1) Este informe: ACTIVIDAD-5-ESTADISTICA-APLICADA-SEP-DIC-2026-ENTREGABLE.docx. "
        "2) Script reproducible: generar_entregable.py. 3) Dataset: datos_credito_sinteticos.csv. "
        "4) Evidencias visuales: figura_01_frecuencia_incumplimiento.png, figura_02_curva_logistica.png y "
        "figura_03_matriz_confusion.png. Los campos pendientes de portada deben completarse antes de entregar."
    )
    doc.add_heading("Referencias", level=1)
    doc.add_paragraph("[1] Kaggle. Pagina principal y descripcion de la plataforma. https://www.kaggle.com/ (consulta: 20 de noviembre de 2026).")
    doc.add_paragraph("[2] statsmodels. Logit — Discrete Model API. https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html")
    doc.add_paragraph("[3] scikit-learn. Model evaluation: confusion matrix and classification metrics. https://scikit-learn.org/stable/modules/model_evaluation.html")

    output = ROOT / "ACTIVIDAD-5-ESTADISTICA-APLICADA-SEP-DIC-2026-ENTREGABLE.docx"
    doc.save(output)
    return output


def main() -> None:
    data = build_dataset()
    data.to_csv(ROOT / "datos_credito_sinteticos.csv", index=False, encoding="utf-8-sig")
    train_raw, test_raw = train_test_split(
        data, test_size=0.25, random_state=SEED, stratify=data[TARGET]
    )
    scaler = StandardScaler()
    train = train_raw.copy()
    test = test_raw.copy()
    train[FEATURES] = scaler.fit_transform(train_raw[FEATURES])
    test[FEATURES] = scaler.transform(test_raw[FEATURES])

    full_model = fit_logit(train, FEATURES)
    llr_choice, r2_choice, _ = subset_search(train, FEATURES)
    llr_model = fit_logit(train, llr_choice["features"])
    r2_model = fit_logit(train, r2_choice["features"])
    full_eval = assess_model(full_model, test[FEATURES], test[TARGET])
    llr_eval = assess_model(llr_model, test[llr_choice["features"]], test[TARGET])
    figures = make_figures(data, test, llr_choice["features"], llr_model)
    report = create_report(
        data,
        train,
        test,
        full_model,
        llr_model,
        r2_model,
        llr_choice,
        r2_choice,
        full_eval,
        llr_eval,
        figures,
    )
    print(f"Informe: {report}")
    print(f"Dataset: {ROOT / 'datos_credito_sinteticos.csv'} ({len(data)} filas)")
    print(f"Incumplimiento simulado: {data[TARGET].mean():.3%}")
    print(f"Pseudo R2 completo: {full_model.prsquared:.5f}; LLR p-value: {full_model.llr_pvalue:.5g}")
    print(f"Seleccion LLR: {llr_choice['features']}; pseudo R2: {llr_choice['prsquared']:.5f}")
    print(f"Seleccion pseudo R2: {r2_choice['features']}; pseudo R2: {r2_choice['prsquared']:.5f}")
    print(f"Accuracy de prueba (modelo LLR): {llr_eval['accuracy']:.3%}")
    print(f"Figuras: {', '.join(str(path.name) for path in figures.values())}")


if __name__ == "__main__":
    main()
