"""Utilidad para obtener el Fear & Greed Index de alternative.me."""

from typing import Optional

import pandas as pd

from .config import FGI_API_URL, FGI_API_TIMEOUT, FGI_MAX_LIMIT


def fetch_fear_greed_index(limit: int = 200) -> Optional[pd.DataFrame]:
    """
    Obtiene el Fear & Greed Index histórico desde la API de alternative.me.

    El índice mide el sentimiento del mercado crypto (basado en BTC)
    en una escala de 0 (Extreme Fear) a 100 (Extreme Greed).

    Parameters
    ----------
    limit : int, default 200
        Número de días históricos a obtener (máximo ~1000).

    Returns
    -------
    pd.DataFrame or None
        DataFrame con DatetimeIndex y columna ``fgi_value`` (int 0-100).
        Retorna None si la API falla.
    """
    import requests

    limit = min(limit, FGI_MAX_LIMIT)

    try:
        response = requests.get(
            FGI_API_URL,
            params={"limit": limit, "format": "json"},
            timeout=FGI_API_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json().get("data", [])
    except (requests.RequestException, ValueError) as e:
        print(f"⚠️ No se pudo obtener el Fear & Greed Index: {e}")
        return None

    if not data:
        print("⚠️ La API de Fear & Greed Index retornó datos vacíos.")
        return None

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
    df["fgi_value"] = df["value"].astype(int)
    df = df.set_index("date").sort_index()[["fgi_value"]]

    return df
