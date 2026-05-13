import pandas as pd
import numpy as np
from pathlib import Path
from shiny import App, ui, render, reactive
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

DATA_PATH = Path(__file__).parent / "data" / "datos_procesados.csv"
DATA_ZAFRA_PATH = Path(__file__).parent / "data" / "datos_agrupados_por_zafra.csv"

def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()

def load_data_zafra():
    if DATA_ZAFRA_PATH.exists():
        return pd.read_csv(DATA_ZAFRA_PATH)
    return pd.DataFrame()

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Controles"),
        ui.input_select("x_var", "Variable predictora:",
            {"cana_molida_neta": "Caña Molida Neta (ton)",
             "superficie_cosechada": "Superficie Cosechada (ha)"}),
        ui.input_selectize("ingenio", "Ingenio:", choices=["Todos"]),
        ui.input_slider("zafra_range", "Rango de Zafra:", min=2014, max=2025, value=(2014, 2025), step=1),
        ui.input_action_button("update", "Actualizar", class_="btn-primary w-100"),
    ),
    ui.navset_tab(
        ui.nav_panel("Dashboard",
            ui.card(
                ui.h4("Serie Temporal - Producción de Azúcar por Zafra"),
                ui.output_plot("plot_series"),
            ),
            ui.card(
                ui.h4("Regresión Lineal - Predicción de Producción"),
                ui.output_plot("plot_regression"),
            ),
            ui.card(
                ui.h4("Modelo de Regresión"),
                ui.output_text("model_summary"),
            ),
            ui.card(
                ui.h4("Datos Filtrados"),
                ui.output_table("data_table"),
            ),
        ),
        ui.nav_panel("Acerca de",
            ui.card(
                ui.h3("Análisis Predictivo de Producción de Azúcar - Infocaña"),
                ui.p("Este dashboard fue desarrollado como parte del proyecto final de Big Data (UNACH, Ene-Jun 2026)."),
                ui.p("Título del proyecto: Análisis predictivo de la producción de azúcar en México mediante regresión lineal y procesamiento ETL con datos abiertos de Infocaña."),
                ui.p("Fuente de datos: https://www.datos.gob.mx/dataset/avance_produccion_cana_azucar_infocana"),
                ui.h5("Metodología:"),
                ui.tags.ul(
                    ui.tags.li("ETL: Extracción de archivos CSV, transformación (limpieza, normalización), carga a CSV procesado"),
                    ui.tags.li("Regresión Lineal: Y = β₀ + β₁·X donde Y = azúcar total, X = caña molida neta o superficie cosechada"),
                    ui.tags.li("División de datos: 80% entrenamiento, 20% pruebas (orden cronológico)"),
                    ui.tags.li("Pronóstico: Extrapolación de tendencia lineal simple para la próxima zafra"),
                ),
                ui.h5("Tecnologías:"),
                ui.tags.ul(
                    ui.tags.li("Python: pandas, numpy, scikit-learn"),
                    ui.tags.li("Shiny for Python: Framework del dashboard"),
                    ui.tags.li("Matplotlib: Visualizaciones"),
                ),
            )
        ),
    ),
    title="Dashboard Azúcar - Infocaña",
)

def server(input, output, session):
    
    @reactive.effect
    def _():
        df = load_data()
        if not df.empty and 'ingenio' in df.columns:
            choices = ["Todos"] + sorted(df['ingenio'].dropna().unique().tolist())
            ui.update_selectize("ingenio", choices=choices)

    @reactive.calc
    def filtered_data():
        df = load_data()
        if df.empty:
            return df
        
        df = df.copy()
        
        if input.ingenio() != "Todos" and 'ingenio' in df.columns:
            df = df[df['ingenio'] == input.ingenio()]
        
        if 'zafra' in df.columns:
            df['zafra_num'] = df['zafra'].astype(str).str.extract(r'(\d{4})').astype(float)
            df = df[(df['zafra_num'] >= input.zafra_range()[0]) & 
                    (df['zafra_num'] <= input.zafra_range()[1])]
        
        return df

    @reactive.calc
    def data_zafra():
        df = filtered_data()
        if df.empty or 'zafra' not in df.columns:
            return pd.DataFrame()
        
        agg_cols = {
            'azucar_producida_total': 'sum',
            'cana_molida_neta': 'sum',
            'superficie_cosechada': 'sum'
        }
        df_agg = df.groupby('zafra').agg(agg_cols).reset_index()
        
        df_agg['zafra_num'] = df_agg['zafra'].astype(str).str.extract(r'(\d{4})').astype(float)
        return df_agg.sort_values('zafra_num')

    @output
    @render.plot
    def plot_series():
        import matplotlib.pyplot as plt
        df = data_zafra()
        if df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center')
            return fig
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar(df['zafra'], df['azucar_producida_total'], color='steelblue', label='Azúcar Total')
        
        if len(df) > 1:
            zafra_nums = df['zafra_num'].values.reshape(-1, 1)
            azucar_vals = df['azucar_producida_total'].values
            lr = LinearRegression()
            lr.fit(zafra_nums, azucar_vals)
            trend = lr.predict(zafra_nums)
            ax.plot(df['zafra'], trend, 'r--', linewidth=2, label='Tendencia Lineal')
            
            next_zafra = np.array([[df['zafra_num'].max() + 1]])
            pred_next = lr.predict(next_zafra)[0]
            ax.scatter([f"{int(df['zafra_num'].max() + 1)}-{int(df['zafra_num'].max() + 2)}"], 
                      [pred_next], color='green', s=100, marker='D', zorder=5, 
                      label=f'Pronóstico {int(df["zafra_num"].max() + 1)}')
        
        ax.set_xlabel('Zafra')
        ax.set_ylabel('Azúcar Producida (ton)')
        ax.set_title('Producción de Azúcar Total por Zafra')
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig

    @output
    @render.plot
    def plot_regression():
        import matplotlib.pyplot as plt
        df = filtered_data()
        if df.empty or input.x_var() not in df.columns:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center')
            return fig
        
        df_clean = df.dropna(subset=['azucar_producida_total', input.x_var()])
        
        if len(df_clean) < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'Datos insuficientes', ha='center', va='center')
            return fig
        
        X = df_clean[[input.x_var()]].values
        y = df_clean['azucar_producida_total'].values
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df_clean[input.x_var()], df_clean['azucar_producida_total'], 
                   alpha=0.6, color='blue', label='Datos')
        
        x_range = np.linspace(df_clean[input.x_var()].min(), df_clean[input.x_var()].max(), 100).reshape(-1, 1)
        y_range = lr.predict(x_range)
        ax.plot(x_range, y_range, 'r-', linewidth=2, label='Regresión Lineal')
        
        ax.set_xlabel(input.x_var())
        ax.set_ylabel('Azúcar Producida (ton)')
        ax.set_title(f'Regresión: Azúcar vs {input.x_var()}')
        ax.legend()
        plt.tight_layout()
        return fig

    @output
    @render.text
    def model_summary():
        df = filtered_data()
        if df.empty or input.x_var() not in df.columns:
            return "No hay datos disponibles"
        
        df_clean = df.dropna(subset=['azucar_producida_total', input.x_var()])
        
        if len(df_clean) < 2:
            return "Datos insuficientes para regresión"
        
        X = df_clean[[input.x_var()]].values
        y = df_clean['azucar_producida_total'].values
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        r2_train = lr.score(X_train, y_train)
        r2_test = lr.score(X_test, y_test)
        
        pred_next_zafra = lr.predict([[df_clean[input.x_var()].mean()]])[0]
        
        summary = f"""Modelo: Y = {lr.intercept_:.2f} + {lr.coef_[0]:.6f} * X

Variables:
  - Y (dependiente): Azúcar Producida Total
  - X (independiente): {input.x_var()}

R² Entrenamiento (80%): {r2_train:.4f}
R² Prueba (20%): {r2_test:.4f}
Coeficiente β₀ (intercepto): {lr.intercept_:.2f}
Coeficiente β₁ (pendiente): {lr.coef_[0]:.6f}

Predicción promedio para {input.x_var()} = {df_clean[input.x_var()].mean():.2f}: {pred_next_zafra:.2f} ton de azúcar
        """
        return summary

    @output
    @render.table
    def data_table():
        df = filtered_data()
        if df.empty:
            return pd.DataFrame()
        
        cols = ['ingenio', 'zafra', 'semana', 'azucar_producida_total', 
                'cana_molida_neta', 'superficie_cosechada', 'rendimiento']
        cols = [c for c in cols if c in df.columns]
        
        return df[cols].head(100)

app = App(app_ui, server)

if __name__ == "__main__":
    import shiny
    shiny.run_app(app, host="127.0.0.1", port=8000)