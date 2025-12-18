# 🎵 Analiza Glasbenih Trendov na Platformi Spotify

## Raziskovalni Problem

Kateri zvočni in žanrski vzorci opredeljujejo glasbene trende na Spotifyju in kako so ti vzorci povezani s priljubljenostjo skladb?

## Raziskovalna Vprašanja

1. Katere zvočne značilnosti najbolj opisujejo sodobne glasbene trende na Spotifyju?
2. Kako so zvočne značilnosti med seboj povezane?
3. Ali obstajajo značilni tipi skladb (glasbeni trendi), ki jih lahko identificiramo s pomočjo podatkovne analize?
4. Kako se posamezni glasbeni trendi razlikujejo glede na priljubljenost skladb?
5. Ali lahko na podlagi zvočnih značilnosti napovemo priljubljenost skladb?

## Struktura Projekta

```
MOSProjekt/
│
├── data/                    # Podatkovne datoteke
│   ├── raw/                 # Surovi podatki
│   └── processed/           # Obdelani podatki
│
├── notebooks/               # Jupyter notebooks
│   └── spotify_analysis.ipynb
│
├── src/                     # Python moduli
│   └── utils.py
│
├── outputs/                 # Rezultati analiz
│   ├── figures/            # Vizualizacije
│   └── models/             # Shranjeni modeli
│
├── requirements.txt         # Python dependencies
└── README.md               # Dokumentacija
```

## Namestitev

```bash
# Ustvari virtualno okolje
python -m venv venv

# Aktiviraj okolje (Windows)
venv\Scripts\activate

# Namesti potrebne knjižnice
pip install -r requirements.txt
```

## Podatkovni Viri

1. **Kaggle** - Spotify datasets (zvočne značilnosti, metapodatki, popularnost)
2. **Spotify Web API** - dodatni podatki o izvajalcih in žanrih
3. **Lastna obdelava** - združevanje, čiščenje, izpeljava novih atributov

## Metodologija

### 1. Priprava Podatkov
- Združevanje podatkov iz različnih virov
- Čiščenje in validacija
- Izbor relevantnih atributov

### 2. Eksplorativna Analiza (EDA)
- Porazdelitve zvočnih značilnosti
- Korelacijska analiza
- Vizualizacije (1D, 2D, 3D)

### 3. Statistična Analiza
- Pearsonove korelacije
- t-test (primerjava priljubljenih/manj priljubljenih)
- ANOVA (razlike med žanri)

### 4. Strojno Učenje
- **Nenadzorovano:** PCA, K-Means clustering
- **Nadzorovano:** Regresija za napoved priljubljenosti

## Zvočne Značilnosti (Audio Features)

| Atribut | Opis | Razpon |
|---------|------|--------|
| `danceability` | Primernost za ples | 0.0 - 1.0 |
| `energy` | Intenzivnost in aktivnost | 0.0 - 1.0 |
| `loudness` | Glasnost v dB | -60 - 0 |
| `speechiness` | Prisotnost govora | 0.0 - 1.0 |
| `acousticness` | Akustična narava | 0.0 - 1.0 |
| `instrumentalness` | Instrumentalna narava | 0.0 - 1.0 |
| `liveness` | Prisotnost občinstva | 0.0 - 1.0 |
| `valence` | Pozitivnost/veselost | 0.0 - 1.0 |
| `tempo` | BPM | 0 - 250 |
| `popularity` | Priljubljenost | 0 - 100 |

## Avtorji

- Projektna skupina MOS 2024/2025
