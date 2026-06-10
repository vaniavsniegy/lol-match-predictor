# League of Legends Match Outcome & Player Performance Predictor

Predicts professional LoL match results and individual player stats using XGBoost trained on Oracle's Elixir data (2014–2026).

Two prediction modes:
- **Pre-match** — based on draft and rosters only (before the game starts)
- **15-minute** — based on draft, rosters, and in-game stats at the 15-min mark

---

## Quick Start

### 1. Install dependencies

```bash
pip install pandas numpy xgboost scikit-learn joblib
```

### 2. Get the data

Download Oracle's Elixir match data CSVs and place them in the `data/` folder:
- https://oracleselixir.com/tools/downloads
- Download each year you want (2014–2026) and drop the CSV files into `data/`

### 3. Train the models

```bash
python lol_xgboost_pipeline.py
```

This reads all CSVs from `data/`, trains 4 models, and saves them to `models/`. Training takes a few minutes.

### 4. Run a demo prediction

```bash
python lol_inference.py
```

---

## Programmatic Usage

```python
from lol_inference import load_models, predict_match

loaded = load_models()
result = predict_match(loaded, match_info, mode='prematch')
# or mode='15min' if you have 15-minute stats
```

---

## Input Format (`match_info`)

```python
match_info = {
    'league':   'LCK',      # tournament league
    'split':    'Spring',   # Spring / Summer / etc.
    'playoffs': 0,          # 1 if playoff game, else 0

    # Only needed for mode='15min'
    'stats': {
        'golddiffat15': 2000,   # Blue team gold lead at 15 min
        'xpdiffat15':   1500,
        'csdiffat15':   10,
        'killsat15':    5,
        'assistsat15':  8,
        'deathsat15':   2,
    },

    'blue': {
        'team': 'T1',
        'picks': ['Gnar', 'Vi', 'Azir', 'Jinx', 'Thresh'],       # 5 picks in draft order
        'bans':  ['Zed', 'Yasuo', 'Katarina', 'LeBlanc', 'Syndra'],  # 5 bans
        'players': {
            'top': {'name': 'Zeus',     'champion': 'Gnar',
                    # Optional per-player 15-min stats (used only in mode='15min')
                    'golddiffat15': 500, 'xpdiffat15': 300, 'csdiffat15': 5,
                    'killsat15': 1, 'assistsat15': 2, 'deathsat15': 0},
            'jng': {'name': 'Oner',     'champion': 'Vi'},
            'mid': {'name': 'Faker',    'champion': 'Azir'},
            'bot': {'name': 'Gumayusi', 'champion': 'Jinx'},
            'sup': {'name': 'Keria',    'champion': 'Thresh'},
        },
    },

    'red': {
        'team': 'Gen.G',
        'picks': ['Jayce', 'Graves', 'Orianna', 'Aphelios', 'Lulu'],
        'bans':  ['Caitlyn', 'Lux', 'Ahri', 'Karma', 'Xayah'],
        'players': {
            'top': {'name': 'Doran',   'champion': 'Jayce'},
            'jng': {'name': 'Peanut',  'champion': 'Graves'},
            'mid': {'name': 'Chovy',   'champion': 'Orianna'},
            'bot': {'name': 'Peyz',    'champion': 'Aphelios'},
            'sup': {'name': 'Delight', 'champion': 'Lulu'},
        },
    },
}
```

---

## Output Format

```python
{
    'game_outcome': {
        'winner': 'Blue',
        'blue_win_probability': 0.7231,
        'red_win_probability':  0.2769,
    },
    'blue_team': {
        'top': {'playername': 'Zeus',     'champion': 'Gnar',  'dpm': 412.5, 'kda': 3.25, 'gold': 15200.0},
        'jng': {'playername': 'Oner',     'champion': 'Vi',    'dpm': 280.0, 'kda': 4.10, 'gold': 12000.0},
        'mid': {'playername': 'Faker',    'champion': 'Azir',  'dpm': 520.0, 'kda': 5.20, 'gold': 16000.0},
        'bot': {'playername': 'Gumayusi', 'champion': 'Jinx',  'dpm': 580.0, 'kda': 4.80, 'gold': 17000.0},
        'sup': {'playername': 'Keria',    'champion': 'Thresh','dpm': 150.0, 'kda': 6.10, 'gold':  8000.0},
    },
    'red_team': {
        # same structure
    },
}
```

---

## Project Structure

```
data/                          # Oracle's Elixir CSV files (not tracked in git)
models/                        # Trained .joblib files (not tracked in git)
lol_xgboost_pipeline.py        # Training script
lol_inference.py               # Inference script
project_documentation.md       # Detailed architecture and design notes
```

---

## Models

| Model | Input | Predicts |
|---|---|---|
| Pre-match outcome | Draft + both rosters | Win probability |
| 15-min outcome | Draft + both rosters + 15-min team stats | Win probability |
| Player performance (pre-game) | Draft + both rosters + player identity | DPM / KDA / Gold |
| Player performance (15-min) | Above + individual 15-min stats | DPM / KDA / Gold |

All models use XGBoost with exponential time-decay sample weights (90-day half-life) so recent matches are weighted more heavily. Champions or players with fewer than 50 appearances across the dataset are bucketed as `__rare__` to prevent overfitting on low-sample entities.

---

## Notes

- **Unknown players/champions** at inference are handled safely — they fall back to the `__rare__` bucket.
- **Unknown teams/leagues** similarly fall back without crashing.
- The `data/` and `models/` folders are excluded from git. You must download the data and retrain locally.
- For detailed architecture and design decisions, see [project_documentation.md](project_documentation.md).
