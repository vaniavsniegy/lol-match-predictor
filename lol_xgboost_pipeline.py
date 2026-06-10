
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.callback import TrainingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib
import glob
import os
import warnings
warnings.filterwarnings('ignore')


class _ProgressBar(TrainingCallback):
    """XGBoost callback that prints a live progress bar."""
    BAR_WIDTH = 30

    def __init__(self, n_estimators, label=''):
        self._n = n_estimators
        self._label = f' {label}' if label else ''

    def after_iteration(self, model, epoch, evals_log):
        filled = (epoch + 1) * self.BAR_WIDTH // self._n
        bar = '#' * filled + '-' * (self.BAR_WIDTH - filled)
        pct = (epoch + 1) / self._n * 100
        print(f'\r    [{bar}] {pct:5.1f}%{self._label}', end='', flush=True)
        if epoch + 1 == self._n:
            print()
        return False

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(ROOT, exist_ok=True)

POSITIONS  = ['top', 'jng', 'mid', 'bot', 'sup']

CHAMP_COLS = [
    'blue_pick1', 'blue_pick2', 'blue_pick3', 'blue_pick4', 'blue_pick5',
    'red_pick1',  'red_pick2',  'red_pick3',  'red_pick4',  'red_pick5',
    'blue_ban1',  'blue_ban2',  'blue_ban3',  'blue_ban4',  'blue_ban5',
    'red_ban1',   'red_ban2',   'red_ban3',   'red_ban4',   'red_ban5',
]
PLAYER_COLS = [
    'blue_top', 'blue_jng', 'blue_mid', 'blue_bot', 'blue_sup',
    'red_top',  'red_jng',  'red_mid',  'red_bot',  'red_sup',
]
CONTEXT_COLS  = ['blue_teamname', 'red_teamname', 'league', 'split', 'playoffs']
STAT_15MIN    = ['golddiffat15', 'xpdiffat15', 'csdiffat15', 'killsat15', 'assistsat15', 'deathsat15']

# Shared feature sets used by all models
MATCH_FEATURES       = CHAMP_COLS + PLAYER_COLS + CONTEXT_COLS
MATCH_FEATURES_15MIN = MATCH_FEATURES + STAT_15MIN

PLAYER_EXTRA_CATS = ['playername', 'champion', 'position']
PLAYER_EXTRA_NUM  = ['player_freq', 'champion_freq']


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir):
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}'")
    df = pd.concat(
        [pd.read_csv(f, low_memory=False) for f in csv_files],
        ignore_index=True
    )
    print(f"Loaded {len(csv_files)} files — {len(df):,} total rows")
    return df


# ---------------------------------------------------------------------------
# Match-level dataset (one row per match, both teams joined)
# ---------------------------------------------------------------------------

def build_match_df(df):
    """Return one row per match with full info from both Blue and Red teams."""

    def pivot_roster(side):
        p = df[(df['position'].isin(POSITIONS)) & (df['side'] == side)][
            ['gameid', 'position', 'playername']
        ]
        wide = p.pivot_table(index='gameid', columns='position',
                             values='playername', aggfunc='first')
        wide.columns.name = None
        wide.columns = [f'{side.lower()}_{c}' for c in wide.columns]
        return wide.reset_index()

    blue_roster = pivot_roster('Blue')
    red_roster  = pivot_roster('Red')

    blue = df[(df['position'] == 'team') & (df['side'] == 'Blue')].copy()
    blue = blue.rename(columns={
        'teamname': 'blue_teamname',
        'pick1': 'blue_pick1', 'pick2': 'blue_pick2', 'pick3': 'blue_pick3',
        'pick4': 'blue_pick4', 'pick5': 'blue_pick5',
        'ban1':  'blue_ban1',  'ban2':  'blue_ban2',  'ban3':  'blue_ban3',
        'ban4':  'blue_ban4',  'ban5':  'blue_ban5',
    })

    red_keep = {
        'gameid': 'gameid', 'teamname': 'red_teamname',
        'pick1': 'red_pick1', 'pick2': 'red_pick2', 'pick3': 'red_pick3',
        'pick4': 'red_pick4', 'pick5': 'red_pick5',
        'ban1':  'red_ban1',  'ban2':  'red_ban2',  'ban3':  'red_ban3',
        'ban4':  'red_ban4',  'ban5':  'red_ban5',
    }
    red = (df[(df['position'] == 'team') & (df['side'] == 'Red')]
           [list(red_keep)].rename(columns=red_keep))

    match_df = (blue
                .merge(red,         on='gameid', how='inner')
                .merge(blue_roster, on='gameid', how='left')
                .merge(red_roster,  on='gameid', how='left'))

    match_df = _add_temporal_weights(match_df)
    print(f"Match dataset: {len(match_df):,} matches")
    return match_df


def enrich_players_with_match_context(df, match_df):
    """Join each individual player row with full match context from match_df."""
    context_cols = CHAMP_COLS + PLAYER_COLS + CONTEXT_COLS + ['gameid', 'sample_weight']
    available = [c for c in context_cols if c in match_df.columns]
    players = df[df['position'].isin(POSITIONS)].copy()
    enriched = players.merge(match_df[available], on='gameid', how='inner',
                             suffixes=('', '_match'))
    return enriched


# ---------------------------------------------------------------------------
# Shared preprocessing helpers
# ---------------------------------------------------------------------------

def _add_temporal_weights(df, half_life_days=90):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    most_recent = df['date'].max()
    df['days_old'] = (most_recent - df['date']).dt.days
    decay = np.log(2) / half_life_days
    df['sample_weight'] = np.exp(-decay * df['days_old'])
    df['sample_weight'] = df['sample_weight'].fillna(df['sample_weight'].median())
    return df


def build_encoders(match_df, player_df=None):
    """Fit shared encoders for all categorical columns across all models.

    Returns a dict with:
      '__champion__' : LabelEncoder for all champion names (picks, bans, individual champion)
      '__player__'   : LabelEncoder for all player names (roster cols + individual playername)
      'position'     : LabelEncoder for position strings
      col            : LabelEncoder for each context/teamname/league/split column
      '_rare_champs' : set of rare champion names (< 50 appearances)
      '_rare_players': set of rare player names  (< 50 appearances)
    """
    encoders = {}

    # Champions — global rarity across all pick + ban slots
    all_champs = pd.concat([match_df[c].dropna() for c in CHAMP_COLS]).astype(str)
    rare_champs = set(all_champs.value_counts().pipe(lambda s: s[s < 50]).index)
    bucketed_champs = all_champs.apply(lambda v: '__rare__' if v in rare_champs else v)
    le_champ = LabelEncoder()
    le_champ.fit(pd.concat([bucketed_champs, pd.Series(['__rare__'])]).unique())
    encoders['__champion__'] = le_champ
    encoders['_rare_champs'] = rare_champs

    # Players — global rarity across all roster slots + individual playername
    player_sources = [match_df[c].dropna() for c in PLAYER_COLS]
    if player_df is not None and 'playername' in player_df.columns:
        player_sources.append(player_df['playername'].dropna())
    all_players = pd.concat(player_sources).astype(str)
    rare_players = set(all_players.value_counts().pipe(lambda s: s[s < 50]).index)
    bucketed_players = all_players.apply(lambda v: '__rare__' if v in rare_players else v)
    le_player = LabelEncoder()
    le_player.fit(pd.concat([bucketed_players, pd.Series(['__rare__'])]).unique())
    encoders['__player__'] = le_player
    encoders['_rare_players'] = rare_players

    # Position
    le_pos = LabelEncoder()
    le_pos.fit(POSITIONS)
    encoders['position'] = le_pos

    # Context columns — individual encoders; include __rare__ so unknown values at inference
    # fall back safely instead of crashing with a ValueError
    for col in ['blue_teamname', 'red_teamname', 'league', 'split']:
        if col in match_df.columns:
            le = LabelEncoder()
            vals = list(match_df[col].dropna().astype(str).unique()) + ['__rare__']
            le.fit(vals)
            encoders[col] = le

    # Frequency tables (for use at inference when freq is unknown)
    if player_df is not None:
        encoders['_champion_freq'] = player_df['champion'].value_counts().to_dict()
        encoders['_player_freq']   = player_df['playername'].value_counts().to_dict()
    else:
        encoders['_champion_freq'] = {}
        encoders['_player_freq']   = {}

    return encoders


def _apply_encoders(df, cols, encoders):
    """Encode a list of columns in df using the shared encoder dict. Returns df copy."""
    df = df.copy()
    for col in cols:
        le, rare = _resolve_encoder(col, encoders)
        if le is None:
            continue
        has_rare = '__rare__' in le.classes_
        rare_code = int(le.transform(['__rare__'])[0]) if has_rare else 0
        # Vectorized lookup: pre-build {value: code} excluding rare-bucketed values so
        # they fall through to fillna(rare_code) — avoids per-row Python calls
        lookup = {cls: int(code) for cls, code in zip(le.classes_, le.transform(le.classes_))
                  if cls not in rare}
        df[col] = df[col].astype(str).map(lookup).fillna(rare_code).astype(int)
    return df


def _resolve_encoder(col, encoders):
    """Return (LabelEncoder, rare_set) for a given column name."""
    if col in CHAMP_COLS or col == 'champion':
        return encoders['__champion__'], encoders['_rare_champs']
    if col in PLAYER_COLS or col == 'playername':
        return encoders['__player__'], encoders['_rare_players']
    if col == 'position':
        return encoders['position'], set()
    if col in encoders:
        return encoders[col], set()
    return None, set()


def encode_value(col, value, encoders):
    """Encode a single raw value for inference."""
    le, rare = _resolve_encoder(col, encoders)
    if le is None:
        return value
    value = str(value)
    if value in rare or value not in le.classes_:
        value = '__rare__'
    return int(le.transform([value])[0])


# ---------------------------------------------------------------------------
# Shared model fitting / evaluation
# ---------------------------------------------------------------------------

def _class_scale_weight(y):
    wins = y.sum()
    losses = len(y) - wins
    return losses / wins if wins > 0 else 1.0


def _fit_classifier(X_train, y_train, w_train, scale_weight, label='', n_estimators=200, lr=0.05):
    m = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=4, learning_rate=lr,
                           scale_pos_weight=scale_weight, random_state=42,
                           tree_method='hist', n_jobs=-1,
                           callbacks=[_ProgressBar(n_estimators, label)])
    m.fit(X_train, y_train, sample_weight=w_train)
    return m


def _fit_regressor(X_train, y_train, w_train, label=''):
    n = 150
    m = xgb.XGBRegressor(n_estimators=n, max_depth=4, learning_rate=0.05, random_state=42,
                          tree_method='hist', n_jobs=-1,
                          callbacks=[_ProgressBar(n, label)])
    m.fit(X_train, y_train, sample_weight=w_train)
    return m


def _evaluate_classifier(model, X_test, y_test, w_test, label):
    acc = accuracy_score(y_test, model.predict(X_test), sample_weight=w_test)
    print(f"  Weighted Accuracy ({label}): {acc:.2%}")


def _evaluate_regressor(model, X_test, y_test, w_test, label):
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test), sample_weight=w_test))
    print(f"  Weighted RMSE ({label}): {rmse:.2f}")


def _strip_callbacks(obj):
    """Remove training callbacks before saving so joblib doesn't pickle _ProgressBar."""
    if hasattr(obj, 'set_params'):
        obj.set_params(callbacks=None)
    elif isinstance(obj, dict):
        for v in obj.values():
            if hasattr(v, 'set_params'):
                v.set_params(callbacks=None)


def _save(name, obj, encoders=None):
    _strip_callbacks(obj)
    joblib.dump(obj, os.path.join(ROOT, f'{name}.joblib'))
    if encoders is not None:
        joblib.dump(encoders, os.path.join(ROOT, f'{name}_encoders.joblib'))
    print(f"  Saved → {name}.joblib")


# ---------------------------------------------------------------------------
# Model 1 — Pre-match game outcome (both teams, full draft + roster)
# ---------------------------------------------------------------------------

def build_prematch_model(match_df, encoders):
    print("\n--- Pre-Match Game Outcome (draft + full roster, both teams) ---")

    cat_cols = CHAMP_COLS + PLAYER_COLS + ['blue_teamname', 'red_teamname', 'league', 'split']
    all_cols  = MATCH_FEATURES
    target    = 'result'

    df = match_df.dropna(subset=all_cols + [target]).copy()
    df = _apply_encoders(df, cat_cols, encoders)

    X, y, w = df[all_cols], df[target], df['sample_weight']
    X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(X, y, w, test_size=0.2, random_state=42)

    print("  Training...")
    model = _fit_classifier(X_tr, y_tr, w_tr, _class_scale_weight(y_tr), label='pre-match')
    _evaluate_classifier(model, X_te, y_te, w_te, "pre-match")
    _save('prematch', model, encoders)

    return model


# ---------------------------------------------------------------------------
# Model 2 — 15-minute game outcome (same features + in-game stats)
# ---------------------------------------------------------------------------

def build_15min_model(match_df, encoders):
    print("\n--- 15-Minute Game Outcome (draft + full roster + 15-min stats) ---")

    cat_cols = CHAMP_COLS + PLAYER_COLS + ['blue_teamname', 'red_teamname', 'league', 'split']
    all_cols  = MATCH_FEATURES_15MIN
    target    = 'result'

    df = match_df.dropna(subset=all_cols + [target]).copy()
    df = _apply_encoders(df, cat_cols, encoders)

    X, y, w = df[all_cols], df[target], df['sample_weight']
    X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(X, y, w, test_size=0.2, random_state=42)

    print("  Training...")
    model = _fit_classifier(X_tr, y_tr, w_tr, _class_scale_weight(y_tr), label='15-min')
    _evaluate_classifier(model, X_te, y_te, w_te, "15-min")
    _save('min15', model)

    return model


# ---------------------------------------------------------------------------
# Shared: train DPM / KDA / gold regressors on a prepared player DataFrame
# ---------------------------------------------------------------------------

def _train_perf_regressors(players_df, features, label):
    players_df = players_df.copy()
    players_df['kda'] = (players_df['kills'] + players_df['assists']) / players_df['deaths'].clip(lower=1)

    targets = {'dpm': 'dpm', 'kda': 'kda', 'gold': 'earnedgold'}
    players_df = players_df.dropna(subset=features + list(targets.values()))

    X, w = players_df[features], players_df['sample_weight']
    models = {}

    print(f"  DPM / KDA / Gold ({label}):")
    for key, col in targets.items():
        y = players_df[col]
        X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(X, y, w, test_size=0.2, random_state=42)
        m = _fit_regressor(X_tr, y_tr, w_tr, label=f'{key} ({label})')
        _evaluate_regressor(m, X_te, y_te, w_te, f"{key}")
        models[key] = m

    return models


# ---------------------------------------------------------------------------
# Model 3 — Player performance WITH 15-min stats (same match features + player id + stats)
# ---------------------------------------------------------------------------

def build_player_perf_model(df, match_df, encoders):
    print("\n--- Player Performance WITH 15-Min Stats ---")

    players = enrich_players_with_match_context(df, match_df)
    players['champion_freq'] = players['champion'].map(players['champion'].value_counts())
    players['player_freq']   = players['playername'].map(players['playername'].value_counts())

    cat_cols = CHAMP_COLS + PLAYER_COLS + PLAYER_EXTRA_CATS + ['league', 'split',
                                                                'blue_teamname', 'red_teamname']
    players = _apply_encoders(players, cat_cols, encoders)

    features = MATCH_FEATURES + PLAYER_EXTRA_CATS + PLAYER_EXTRA_NUM + STAT_15MIN
    features = [f for f in features if f in players.columns]

    players = players.dropna(subset=features)
    models = _train_perf_regressors(players, features, 'with-stats')
    _save('player_perf', models, encoders)

    return models


# ---------------------------------------------------------------------------
# Model 4 — Player performance WITHOUT 15-min stats (pre-game, same match features)
# ---------------------------------------------------------------------------

def build_player_perf_pregame_model(df, match_df, encoders):
    print("\n--- Player Performance WITHOUT 15-Min Stats (pre-game) ---")

    players = enrich_players_with_match_context(df, match_df)
    players['champion_freq'] = players['champion'].map(players['champion'].value_counts())
    players['player_freq']   = players['playername'].map(players['playername'].value_counts())

    cat_cols = CHAMP_COLS + PLAYER_COLS + PLAYER_EXTRA_CATS + ['league', 'split',
                                                                'blue_teamname', 'red_teamname']
    players = _apply_encoders(players, cat_cols, encoders)

    features = MATCH_FEATURES + PLAYER_EXTRA_CATS + PLAYER_EXTRA_NUM
    features = [f for f in features if f in players.columns]

    players = players.dropna(subset=features)
    models = _train_perf_regressors(players, features, 'pre-game')
    _save('player_perf_pregame', models, encoders)

    return models


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df       = load_data('data')
    match_df = build_match_df(df)

    # Fit shared encoders once — all models use the same encoding
    player_df = df[df['position'].isin(POSITIONS)]
    encoders  = build_encoders(match_df, player_df)

    build_prematch_model(match_df, encoders)
    build_15min_model(match_df, encoders)
    build_player_perf_model(df, match_df, encoders)
    build_player_perf_pregame_model(df, match_df, encoders)

    print("\nAll four models trained and saved.")
