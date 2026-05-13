import os
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
CSV_DIR = BASE_DIR / "csv"
DATA_DIR = BASE_DIR / "data"

def extract():
    csv_files = list(CSV_DIR.glob("*.csv"))
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)
    df_raw = pd.concat(dfs, ignore_index=True)
    print(f"Extract: {len(csv_files)} archivos cargados, {len(df_raw)} registros")
    return df_raw

def transform(df):
    df = df.copy()
    
    df.columns = df.columns.str.lower().str.strip()
    
    if 'azucar_total' in df.columns and 'azucar_producida_total' in df.columns:
        if 'azucar_total' in df.columns:
            df = df.drop(columns=['azucar_producida_total'])
    
    col_rename = {}
    if 'azucar_total' in df.columns:
        col_rename['azucar_total'] = 'azucar_producida_total'
    df = df.rename(columns=col_rename)
    
    numeric_cols = ['azucar_producida_total', 'cana_molida_neta', 
                    'superficie_cosechada', 'cana_molida_bruta']
    for col in numeric_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    df = df.dropna(subset=['azucar_producida_total', 'cana_molida_neta'])
    
    if 'ingenio' in df.columns:
        df['ingenio'] = df['ingenio'].astype(str).str.strip().str.title()
    
    if 'zafra' in df.columns:
        df['zafra'] = df['zafra'].astype(str)
    
    df['rendimiento'] = (df['azucar_producida_total'] / df['cana_molida_neta']) * 100
    
    print(f"Transform: {len(df)} registros limpios")
    return df

def load(df):
    DATA_DIR.mkdir(exist_ok=True)
    output_path = DATA_DIR / "datos_procesados.csv"
    df.to_csv(output_path, index=False)
    print(f"Load: Datos guardados en {output_path}")
    return output_path

def aggregate_by_zafra(df):
    agg_cols = {
        'azucar_producida_total': 'sum',
        'cana_molida_neta': 'sum',
        'superficie_cosechada': 'sum',
        'rendimiento': 'mean'
    }
    df_agg = df.groupby('zafra').agg(agg_cols).reset_index()
    df_agg = df_agg.sort_values('zafra')
    
    DATA_DIR.mkdir(exist_ok=True)
    output_path = DATA_DIR / "datos_agrupados_por_zafra.csv"
    df_agg.to_csv(output_path, index=False)
    print(f"Aggregate: Datos agrupados por zafra en {output_path}")
    return df_agg

def run_etl():
    print("=" * 50)
    print("INICIANDO ETL - Infocaña")
    print("=" * 50)
    
    df_raw = extract()
    df_clean = transform(df_raw)
    load(df_clean)
    df_zafra = aggregate_by_zafra(df_clean)
    
    print("=" * 50)
    print("ETL COMPLETADO")
    print("=" * 50)
    return df_clean, df_zafra

if __name__ == "__main__":
    run_etl()