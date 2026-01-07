"""
Utility functions for Spotify Music Trends Analysis
====================================================
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
import warnings
import os
warnings.filterwarnings('ignore')

# Spotify API
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


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


# =============================================================================
# SPOTIFY API INTEGRATION
# =============================================================================

class SpotifyAPI:
    """
    Razred za interakcijo s Spotify Web API.
    
    Omogoča pridobivanje audio značilnosti, informacij o izvajalcih,
    top skladbah in podobnih izvajalcih.
    
    Parameters
    ----------
    client_id : str, optional
        Spotify Client ID (lahko tudi iz okoljskih spremenljivk)
    client_secret : str, optional
        Spotify Client Secret (lahko tudi iz okoljskih spremenljivk)
    """
    
    def __init__(self, client_id: str = None, client_secret: str = None):
        self.client_id = client_id or os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('SPOTIFY_CLIENT_SECRET')
        self.sp = None
        
        if self.client_id and self.client_secret:
            self._authenticate()
    
    def _authenticate(self):
        """Avtentikacija s Spotify API."""
        try:
            auth_manager = SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            self.sp = spotipy.Spotify(auth_manager=auth_manager)
            print("✅ Uspešno povezan s Spotify API")
        except Exception as e:
            print(f"❌ Napaka pri avtentikaciji: {e}")
            self.sp = None
    
    def is_connected(self) -> bool:
        """Preveri, ali je povezava vzpostavljena."""
        return self.sp is not None
    
    def search_track(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Išče skladbe po imenu/izvajalcu.
        
        Parameters
        ----------
        query : str
            Iskalni niz (ime skladbe, izvajalec)
        limit : int
            Maksimalno število rezultatov
            
        Returns
        -------
        List[Dict]
            Seznam najdenih skladb
        """
        if not self.is_connected():
            print("❌ Ni povezave s Spotify API")
            return []
        
        try:
            results = self.sp.search(q=query, limit=limit, type='track')
            tracks = []
            for item in results['tracks']['items']:
                tracks.append({
                    'id': item['id'],
                    'name': item['name'],
                    'artist': item['artists'][0]['name'],
                    'album': item['album']['name'],
                    'popularity': item['popularity'],
                    'preview_url': item['preview_url']
                })
            return tracks
        except Exception as e:
            print(f"❌ Napaka pri iskanju: {e}")
            return []
    
    def get_audio_features(self, track_ids: List[str]) -> pd.DataFrame:
        """
        ⚠️ DEPRECATED - Spotify je ukinil Audio Features API novembra 2024.
        
        Ta metoda vrne prazen DataFrame. Za audio značilnosti uporabite
        lokalni dataset, ki že vsebuje te podatke.
        
        Parameters
        ----------
        track_ids : List[str]
            Seznam Spotify track ID-jev (ignoriran)
            
        Returns
        -------
        pd.DataFrame
            Prazen DataFrame
        """
        warnings.warn(
            "⚠️ DEPRECATED: Spotify Audio Features API je bil ukinjen novembra 2024. "
            "Uporabite lokalni dataset za audio značilnosti.",
            DeprecationWarning,
            stacklevel=2
        )
        print("⚠️ DEPRECATED: Spotify Audio Features API ni več na voljo!")
        print("   Uporabite podatke iz lokalnega dataseta (../data/dataset.csv)")
        return pd.DataFrame()
    
    def get_artist_info(self, artist_id: str) -> Dict:
        """
        Pridobi informacije o izvajalcu.
        
        OPOMBA: Polje 'genres' je deprecated in bo vedno prazno.
        Za žanre uporabite lokalni dataset.
        
        Parameters
        ----------
        artist_id : str
            Spotify artist ID
            
        Returns
        -------
        Dict
            Informacije o izvajalcu
        """
        if not self.is_connected():
            print("❌ Ni povezave s Spotify API")
            return {}
        
        try:
            artist = self.sp.artist(artist_id)
            return {
                'id': artist['id'],
                'name': artist['name'],
                # 'genres' je deprecated in vrača prazen seznam
                'popularity': artist['popularity'],
                'followers': artist['followers']['total'],
                'image_url': artist['images'][0]['url'] if artist['images'] else None
            }
        except Exception as e:
            print(f"❌ Napaka pri pridobivanju izvajalca: {e}")
            return {}
    
    def get_artist_top_tracks(self, artist_id: str, country: str = 'US') -> List[Dict]:
        """
        Pridobi top skladbe izvajalca.
        
        Parameters
        ----------
        artist_id : str
            Spotify artist ID
        country : str
            Koda države za top skladbe
            
        Returns
        -------
        List[Dict]
            Seznam top skladb
        """
        if not self.is_connected():
            print("❌ Ni povezave s Spotify API")
            return []
        
        try:
            results = self.sp.artist_top_tracks(artist_id, country=country)
            tracks = []
            for track in results['tracks']:
                tracks.append({
                    'id': track['id'],
                    'name': track['name'],
                    'popularity': track['popularity'],
                    'album': track['album']['name'],
                    'duration_ms': track['duration_ms'],
                    'explicit': track.get('explicit', False)
                })
            return tracks
        except Exception as e:
            print(f"❌ Napaka pri pridobivanju top skladb: {e}")
            return []
    
    def get_related_artists(self, artist_id: str) -> List[Dict]:
        """
        Pridobi povezane izvajalce.
        
        Parameters
        ----------
        artist_id : str
            Spotify artist ID
            
        Returns
        -------
        List[Dict]
            Seznam povezanih izvajalcev
        """
        if not self.is_connected():
            print("❌ Ni povezave s Spotify API")
            return []
        
        try:
            results = self.sp.artist_related_artists(artist_id)
            related = []
            for artist in results['artists']:
                related.append({
                    'id': artist['id'],
                    'name': artist['name'],
                    'popularity': artist['popularity'],
                    'followers': artist['followers']['total']
                })
            return related
        except Exception as e:
            print(f"❌ Napaka pri pridobivanju povezanih izvajalcev: {e}")
            return []
    
    def get_new_releases(self, country: str = 'US', limit: int = 20) -> List[Dict]:
        """
        Pridobi nove izdaje albumov.
        
        Parameters
        ----------
        country : str
            Koda države
        limit : int
            Maksimalno število rezultatov
            
        Returns
        -------
        List[Dict]
            Seznam novih albumov
        """
        if not self.is_connected():
            print("❌ Ni povezave s Spotify API")
            return []
        
        try:
            results = self.sp.new_releases(country=country, limit=limit)
            albums = []
            for album in results['albums']['items']:
                albums.append({
                    'id': album['id'],
                    'name': album['name'],
                    'artist': album['artists'][0]['name'],
                    'release_date': album['release_date'],
                    'total_tracks': album['total_tracks'],
                    'album_type': album['album_type']
                })
            return albums
        except Exception as e:
            print(f"❌ Napaka pri pridobivanju novih izdaj: {e}")
            return []
    
    def get_recommendations(self, seed_tracks: List[str] = None, 
                           seed_artists: List[str] = None,
                           seed_genres: List[str] = None,
                           limit: int = 20,
                           **kwargs) -> List[Dict]:
        """
        Pridobi priporočila na podlagi seedov.
        
        Parameters
        ----------
        seed_tracks : List[str], optional
            Seznam track ID-jev za seed
        seed_artists : List[str], optional
            Seznam artist ID-jev za seed
        seed_genres : List[str], optional
            Seznam žanrov za seed
        limit : int
            Maksimalno število priporočil
        **kwargs
            Dodatni parametri (target_energy, min_danceability, itd.)
            
        Returns
        -------
        List[Dict]
            Seznam priporočenih skladb
        """
        if not self.is_connected():
            print("❌ Ni povezave s Spotify API")
            return []
        
        try:
            results = self.sp.recommendations(
                seed_tracks=seed_tracks,
                seed_artists=seed_artists,
                seed_genres=seed_genres,
                limit=limit,
                **kwargs
            )
            tracks = []
            for track in results['tracks']:
                tracks.append({
                    'id': track['id'],
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'popularity': track['popularity'],
                    'preview_url': track['preview_url']
                })
            return tracks
        except Exception as e:
            print(f"❌ Napaka pri pridobivanju priporočil: {e}")
            return []
    
    def enrich_dataframe_with_api(self, df: pd.DataFrame, 
                                  track_id_column: str = 'track_id') -> pd.DataFrame:
        """
        ⚠️ DEPRECATED - Ta metoda je bila odvisna od Audio Features API,
        ki ga je Spotify ukinil novembra 2024.
        
        Za obogatitev podatkov uporabite druge API metode:
        - get_artist_info() za podatke o izvajalcih
        - get_artist_top_tracks() za top skladbe
        - get_related_artists() za povezane izvajalce
        
        Parameters
        ----------
        df : pd.DataFrame
            Vhodni DataFrame
        track_id_column : str
            Ime stolpca s track ID-ji
            
        Returns
        -------
        pd.DataFrame
            Nespremenjen DataFrame
        """
        warnings.warn(
            "⚠️ DEPRECATED: enrich_dataframe_with_api je odvisen od Audio Features API, "
            "ki je bil ukinjen novembra 2024.",
            DeprecationWarning,
            stacklevel=2
        )
        print("⚠️ DEPRECATED: Ta metoda ni več na voljo zaradi ukinitve Audio Features API!")
        return df


def create_spotify_client(client_id: str = None, client_secret: str = None) -> SpotifyAPI:
    """
    Ustvari Spotify API odjemalca.
    
    Parameters
    ----------
    client_id : str, optional
        Spotify Client ID
    client_secret : str, optional
        Spotify Client Secret
        
    Returns
    -------
    SpotifyAPI
        Instanca SpotifyAPI razreda
    """
    return SpotifyAPI(client_id, client_secret)
