"""
Utility functions for Spotify Music Trends Analysis
====================================================
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# AUDIO FEATURES DEFINITIONS
# =============================================================================

AUDIO_FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

AUDIO_FEATURES_NORMALIZED = [
    'danceability', 'energy', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence'
]

FEATURE_DESCRIPTIONS = {
    'danceability': 'Primernost skladbe za ples (0-1)',
    'energy': 'Intenzivnost in aktivnost skladbe (0-1)',
    'loudness': 'Povprečna glasnost v decibelih (dB)',
    'speechiness': 'Prisotnost govorjenih besed (0-1)',
    'acousticness': 'Verjetnost akustične izvedbe (0-1)',
    'instrumentalness': 'Napoveduje ali skladba vsebuje vokale (0-1)',
    'liveness': 'Verjetnost žive izvedbe (0-1)',
    'valence': 'Pozitivnost/veselost skladbe (0-1)',
    'tempo': 'Tempo v BPM',
    'popularity': 'Priljubljenost skladbe (0-100)'
}


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def load_spotify_data(filepath: str) -> pd.DataFrame:
    """
    Naloži Spotify podatke iz CSV datoteke.
    
    Parameters
    ----------
    filepath : str
        Pot do CSV datoteke
        
    Returns
    -------
    pd.DataFrame
        Naloženi podatki
    """
    df = pd.read_csv(filepath)
    print(f"✓ Naloženih {len(df):,} skladb")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Očisti podatke - odstrani duplikate in manjkajoče vrednosti.
    
    Parameters
    ----------
    df : pd.DataFrame
        Surovi podatki
        
    Returns
    -------
    pd.DataFrame
        Očiščeni podatki
    """
    initial_count = len(df)
    
    # Odstrani duplikate
    df = df.drop_duplicates()
    
    # Odstrani vrstice z manjkajočimi vrednostmi v ključnih stolpcih
    key_columns = [col for col in AUDIO_FEATURES if col in df.columns]
    if 'popularity' in df.columns:
        key_columns.append('popularity')
    
    df = df.dropna(subset=key_columns)
    
    final_count = len(df)
    removed = initial_count - final_count
    
    print(f"✓ Odstranjenih {removed:,} vrstic ({removed/initial_count*100:.1f}%)")
    print(f"✓ Končno število skladb: {final_count:,}")
    
    return df.reset_index(drop=True)


def normalize_loudness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizira glasnost na razpon 0-1.
    
    Parameters
    ----------
    df : pd.DataFrame
        Podatki z loudness stolpcem
        
    Returns
    -------
    pd.DataFrame
        Podatki z normalizirano glasnostjo
    """
    if 'loudness' in df.columns:
        df = df.copy()
        min_loud = df['loudness'].min()
        max_loud = df['loudness'].max()
        df['loudness_normalized'] = (df['loudness'] - min_loud) / (max_loud - min_loud)
    return df


def categorize_popularity(df: pd.DataFrame, 
                          thresholds: Tuple[int, int] = (33, 66)) -> pd.DataFrame:
    """
    Kategorizira priljubljenost v razrede.
    
    Parameters
    ----------
    df : pd.DataFrame
        Podatki s popularity stolpcem
    thresholds : Tuple[int, int]
        Mejne vrednosti za kategorije
        
    Returns
    -------
    pd.DataFrame
        Podatki z dodanim stolpcem popularity_category
    """
    if 'popularity' in df.columns:
        df = df.copy()
        conditions = [
            df['popularity'] <= thresholds[0],
            (df['popularity'] > thresholds[0]) & (df['popularity'] <= thresholds[1]),
            df['popularity'] > thresholds[1]
        ]
        choices = ['Nizka', 'Srednja', 'Visoka']
        df['popularity_category'] = np.select(conditions, choices, default='Neznano')
    return df


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def calculate_correlations(df: pd.DataFrame, 
                          features: List[str] = None) -> pd.DataFrame:
    """
    Izračuna korelacijsko matriko za izbrane značilnosti.
    
    Parameters
    ----------
    df : pd.DataFrame
        Podatki
    features : List[str], optional
        Seznam značilnosti za analizo
        
    Returns
    -------
    pd.DataFrame
        Korelacijska matrika
    """
    if features is None:
        features = [f for f in AUDIO_FEATURES if f in df.columns]
    
    return df[features].corr()


def get_popularity_correlations(df: pd.DataFrame) -> pd.Series:
    """
    Vrne korelacije zvočnih značilnosti s priljubljenostjo.
    
    Parameters
    ----------
    df : pd.DataFrame
        Podatki
        
    Returns
    -------
    pd.Series
        Korelacije, urejene po absolutni vrednosti
    """
    features = [f for f in AUDIO_FEATURES if f in df.columns]
    
    if 'popularity' not in df.columns:
        raise ValueError("Stolpec 'popularity' ni prisoten v podatkih")
    
    correlations = df[features + ['popularity']].corr()['popularity'].drop('popularity')
    return correlations.sort_values(key=abs, ascending=False)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_energy_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ustvari razmerje med energijo in akustičnostjo.
    """
    df = df.copy()
    if 'energy' in df.columns and 'acousticness' in df.columns:
        df['energy_acoustic_ratio'] = df['energy'] / (df['acousticness'] + 0.01)
    return df


def create_mood_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ustvari kompozitni indeks razpoloženja.
    """
    df = df.copy()
    if all(col in df.columns for col in ['valence', 'energy', 'danceability']):
        df['mood_score'] = (df['valence'] + df['energy'] + df['danceability']) / 3
    return df


def create_club_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ustvari indeks primernosti za klube.
    """
    df = df.copy()
    if all(col in df.columns for col in ['danceability', 'energy', 'tempo']):
        # Normaliziraj tempo na 0-1 (predpostavka: 60-180 BPM)
        tempo_norm = (df['tempo'].clip(60, 180) - 60) / 120
        df['club_score'] = (df['danceability'] + df['energy'] + tempo_norm) / 3
    return df


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================

def get_summary_statistics(df: pd.DataFrame, 
                          features: List[str] = None) -> pd.DataFrame:
    """
    Vrne opisno statistiko za izbrane značilnosti.
    
    Parameters
    ----------
    df : pd.DataFrame
        Podatki
    features : List[str], optional
        Seznam značilnosti
        
    Returns
    -------
    pd.DataFrame
        Opisna statistika
    """
    if features is None:
        features = [f for f in AUDIO_FEATURES + ['popularity'] if f in df.columns]
    
    stats = df[features].describe().T
    stats['median'] = df[features].median()
    stats['skew'] = df[features].skew()
    stats['kurtosis'] = df[features].kurtosis()
    
    return stats


def identify_outliers_iqr(df: pd.DataFrame, 
                          column: str, 
                          multiplier: float = 1.5) -> pd.Series:
    """
    Identificira osamelce z IQR metodo.
    
    Parameters
    ----------
    df : pd.DataFrame
        Podatki
    column : str
        Ime stolpca
    multiplier : float
        IQR množilnik
        
    Returns
    -------
    pd.Series
        Boolean maska za osamelce
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)


def print_analysis_header(title: str):
    """Izpiše formatiran naslov sekcije."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60 + "\n")
