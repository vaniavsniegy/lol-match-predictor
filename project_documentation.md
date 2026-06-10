# Leveraging Historical Player Statistics and Champion Selection for Game Outcome Prediction in Professional League of Legends

## Project Goal
The primary objective is to build a machine learning pipeline that predicts professional League of Legends match outcomes and individual player performance. By using Oracle's Elixir data and combining tree-based models (XGBoost) with game-theory explanations (SHAP), the system quantifies exactly how specific variables — gold leads, champion picks, player identity — impact win probability and end-game stats.

The emphasis is on **statistical inference**: not just predicting outcomes, but understanding *why*.

---

## Pipeline Architecture

All four models share the same base dataset (`match_df` — one row per match with both teams fully joined) and the same shared categorical encoders. Game outcome models and player performance models use the same match-level feature set, making them a unified system.

```
load_data()
     │
     ├── build_match_df()        ← one row per match, both teams fully joined
     │
     ├── build_encoders()        ← one shared encoder dict for all models
     │     (champion, player, position, teamname, league, split)
     │
     ├── build_prematch_model()           ← Model 1: full draft + both rosters → win %
     ├── build_15min_model()              ← Model 2: same + 15-min team stats → win %
     ├── build_player_perf_model()        ← Model 3: match context + player id + stats → DPM/KDA/Gold
     └── build_player_perf_pregame_model() ← Model 4: same without 15-min stats
```

---

## Shared Infrastructure

### `build_match_df(df)` — Full Match Dataset
Constructs one row per match with complete information from both Blue and Red teams:
- **Blue team row**: result, 15-min stats, league/split/playoffs, blue picks (pick1-5), blue bans (ban1-5), blue teamname
- **Red team row** (joined on `gameid`): red picks, red bans, red teamname
- **Pivoted rosters** (joined on `gameid`): `blue_top/jng/mid/bot/sup` and `red_top/jng/mid/bot/sup` — player names per position for both sides

### `build_encoders(match_df, player_df)` — Shared Categorical Encoders
All categorical columns are encoded once and shared across all four models. This ensures a champion name always maps to the same integer in every model.

| Key | Covers |
|---|---|
| `__champion__` | All 20 pick/ban slots + individual player champion column |
| `__player__` | All 10 roster slots + individual playername column |
| `position` | top / jng / mid / bot / sup |
| `blue_teamname`, `red_teamname`, `league`, `split` | Team and tournament identity |
| `_rare_champs` | Set of champion names with < 50 appearances (used for bucketing) |
| `_rare_players` | Set of player names with < 50 appearances |
| `_champion_freq` | Dict of champion → game count (for inference lookup) |
| `_player_freq` | Dict of player → game count (for inference lookup) |

### Temporal Weighting
All models apply exponential decay (90-day half-life) to sample weights. A match from 3 months ago contributes half the weight of today's match, reflecting how quickly the professional meta evolves.

### Class Imbalance
Blue Side has a ~53.6% win-rate advantage. All classifiers use `scale_pos_weight = losses / wins` to prevent the model from exploiting map-side bias.

### Champion and Player Rarity
- **Rare bucketing**: any champion/player with fewer than 50 appearances across the full dataset is relabeled `__rare__`. This prevents overfitting on counterpick selections or newly released champions whose limited sample inflates apparent performance.
- **Frequency features** (`champion_freq`, `player_freq`): the raw game count is passed as a numeric feature in player performance models so the model can learn to regress toward the mean for low-sample entities.

---

## Shared Feature Set (used by all four models)

| Feature group | Columns |
|---|---|
| Blue picks | `blue_pick1` – `blue_pick5` |
| Red picks | `red_pick1` – `red_pick5` |
| Blue bans | `blue_ban1` – `blue_ban5` |
| Red bans | `red_ban1` – `red_ban5` |
| Blue roster | `blue_top`, `blue_jng`, `blue_mid`, `blue_bot`, `blue_sup` |
| Red roster | `red_top`, `red_jng`, `red_mid`, `red_bot`, `red_sup` |
| Context | `blue_teamname`, `red_teamname`, `league`, `split`, `playoffs` |

All champion and player columns are rare-bucketed and label-encoded via the shared encoders.

---

## Model 1 — Pre-Match Game Outcome

**Question:** *Before the game starts, who is likely to win based on the full draft and both rosters?*

**Features:** Full shared feature set (30 features).

**Target:** `result` (1 = Blue wins, 0 = Blue loses)

**Model:** `XGBClassifier` (200 estimators, `max_depth=4`, `lr=0.05`)

---

## Model 2 — 15-Minute Game Outcome

**Question:** *At the 15-minute mark, who is likely to win given current game state?*

**Features:** Full shared feature set + `golddiffat15`, `xpdiffat15`, `csdiffat15`, `killsat15`, `assistsat15`, `deathsat15` (36 features total).

**Target:** `result`

**Model:** `XGBClassifier` (200 estimators, `max_depth=4`, `lr=0.05`)

---

## Model 3 — Player Performance WITH 15-Min Stats

**Question:** *Given full match context and a player's early-game numbers, what will their end-game stats look like?*

**Features:** Full shared feature set + `playername`, `champion`, `position`, `player_freq`, `champion_freq` + individual `golddiffat15`, `xpdiffat15`, `csdiffat15`, `killsat15`, `assistsat15`, `deathsat15`.

**Targets (3 regressors):**
- `dpm` — damage per minute to champions
- `kda` — `(kills + assists) / max(deaths, 1)`
- `gold` — `earnedgold` (excludes passive starting gold)

**Model:** `XGBRegressor` per target (150 estimators, `max_depth=4`, `lr=0.05`)

---

## Model 4 — Player Performance WITHOUT 15-Min Stats (Pre-Game Baseline)

**Question:** *Based only on who is playing and what was drafted — before the game begins — what stats are expected?*

**Features:** Full shared feature set + `playername`, `champion`, `position`, `player_freq`, `champion_freq`.

**Targets:** `dpm`, `kda`, `gold` (same 3 regressors as Model 3).

**Purpose:** Baseline. The RMSE gap between Model 3 and Model 4 quantifies how much early-game performance adds beyond pre-game knowledge alone.

---

## Model Persistence

All models save automatically to the project root:

| File | Contents |
|---|---|
| `prematch.joblib` | `XGBClassifier` |
| `prematch_encoders.joblib` | Shared encoder dict (used by all models) |
| `min15.joblib` | `XGBClassifier` |
| `player_perf.joblib` | `{'dpm': XGBRegressor, 'kda': ..., 'gold': ...}` |
| `player_perf_pregame.joblib` | `{'dpm': XGBRegressor, 'kda': ..., 'gold': ...}` |

---

## Inference API (`lol_inference.py`)

### `load_models()` — Load all .joblib files from the project root

### `predict_match(loaded_models, match_info, mode='prematch')`
Single call that returns game outcome + all 10 players' predicted DPM / KDA / Gold.

`mode='prematch'` → uses Model 1 (game) + Model 4 (players)
`mode='15min'` → uses Model 2 (game) + Model 3 (players, with 15-min stats)

**Output:**
```python
{
    'game_outcome': {
        'winner': 'Blue',
        'blue_win_probability': 0.7231,
        'red_win_probability':  0.2769,
    },
    'blue_team': {
        'top': {'playername': 'Zeus',  'champion': 'Gnar',   'dpm': 412.5, 'kda': 3.25, 'gold': 15200.0},
        'jng': {'playername': 'Oner',  'champion': 'Vi',     'dpm': 280.0, 'kda': 4.10, 'gold': 12000.0},
        'mid': {'playername': 'Faker', 'champion': 'Azir',   'dpm': 520.0, 'kda': 5.20, 'gold': 16000.0},
        'bot': {'playername': 'Gumayusi', 'champion': 'Jinx','dpm': 580.0, 'kda': 4.80, 'gold': 17000.0},
        'sup': {'playername': 'Keria', 'champion': 'Thresh', 'dpm': 150.0, 'kda': 6.10, 'gold':  8000.0},
    },
    'red_team': { ... }
}
```

---

## Usage

**Step 1 — Train and save all models:**
```bash
python lol_xgboost_pipeline.py
```

**Step 2 — Run inference:**
```bash
python lol_inference.py
```

**Programmatic use:**
```python
from lol_inference import load_models, predict_match

loaded = load_models()
result = predict_match(loaded, match_info, mode='prematch')
```

---

## Dependencies

```bash
pip install pandas numpy xgboost scikit-learn shap joblib
```

---

## Design Decisions and Limitations

* **Both teams included:** All four models now receive both teams' rosters, picks, and bans. This eliminates the prior Blue-side-only bias and lets the model learn from the full match context.
* **Shared feature set:** Game outcome and player performance models use identical match-level features. A single `predict_match` call covers both predictions.
* **15-minute snapshot:** Game and player stats are evaluated at the 15-min mark as a deliberate design choice — this enables live in-game inference at a well-defined checkpoint.
* **No normalization:** Tree-based models are scale-invariant.
* **No hyperparameter tuning:** Defaults are reasonable but grid search or Bayesian optimization could improve all four models.

---

## Changelog

| Date | Change |
|---|---|
| 2026-06-10 | Initial pipeline with single-year data, Blue-side-only filtering, XGBoost + SHAP |
| 2026-06-10 | Expanded to all 13 years of Oracle's Elixir data (2014–2026) from `data/` folder |
| 2026-06-10 | Added champion rarity bucketing (`__rare__` for < 50 appearances) and `champion_freq` feature |
| 2026-06-10 | Added model persistence (`joblib`), DPM/KDA/Gold multi-target regression, player name features |
| 2026-06-10 | Full architecture redesign: `build_match_df` joins both teams into one row; shared `build_encoders`; game outcome and player performance models unified on the same feature set; `predict_match` returns full match prediction (game outcome + all 10 players) in one call |
| 2026-06-10 | Bug fixes: (1) `_print_result` moved before `__main__` block to prevent NameError; (2) context encoders (`blue_teamname`, `red_teamname`, `league`, `split`) now include `__rare__` so unknown values at inference don't crash; (3) `scale_pos_weight` now computed from `y_tr` (training split only) rather than full dataset |
