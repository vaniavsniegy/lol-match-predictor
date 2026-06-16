import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.callback import TrainingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    mean_squared_error,
)
from sklearn.preprocessing import LabelEncoder
import joblib
import glob
import os
import warnings
import json
from datetime import datetime
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
RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_ROOT, exist_ok=True)

POSITIONS  = ['top', 'jng', 'mid', 'bot', 'sup']

CHAMP_COLS = [
    'blue_top_champ', 'blue_jng_champ', 'blue_mid_champ', 'blue_bot_champ', 'blue_sup_champ',
    'red_top_champ',  'red_jng_champ',  'red_mid_champ',  'red_bot_champ',  'red_sup_champ',
    'blue_ban1',  'blue_ban2',  'blue_ban3',  'blue_ban4',  'blue_ban5',
    'red_ban1',   'red_ban2',   'red_ban3',   'red_ban4',   'red_ban5',
]
PLAYER_COLS = [
    'blue_top', 'blue_jng', 'blue_mid', 'blue_bot', 'blue_sup',
    'red_top',  'red_jng',  'red_mid',  'red_bot',  'red_sup',
]
CONTEXT_COLS  = ['blue_teamname', 'red_teamname', 'league', 'split', 'playoffs', 'patch']
STAT_15MIN    = ['golddiffat15', 'xpdiffat15', 'csdiffat15', 'killsat15', 'assistsat15', 'deathsat15']

# Shared feature sets used by all models
MATCH_FEATURES = CHAMP_COLS + PLAYER_COLS + CONTEXT_COLS
MATCH_FEATURES_15MIN = MATCH_FEATURES + STAT_15MIN

# Baseline feature groups for match-outcome experiments.
# These let us compare whether the model is learning from draft, roster/team,
# or mostly from simple side/team/context priors.
DRAFT_FEATURES = CHAMP_COLS
PLAYER_TEAM_FEATURES = PLAYER_COLS + CONTEXT_COLS
TEAM_ONLY_FEATURES = CONTEXT_COLS

FIFTEEN_MIN_ONLY_FEATURES = STAT_15MIN

PLAYER_EXTRA_CATS = ['playername', 'champion', 'position']
PLAYER_EXTRA_NUM = ['player_freq', 'champion_freq']


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

    def pivot_champions(side):
        p = df[(df['position'].isin(POSITIONS)) & (df['side'] == side)][
            ['gameid', 'position', 'champion']
        ]
        wide = p.pivot_table(index='gameid', columns='position',
                             values='champion', aggfunc='first')
        wide.columns.name = None
        wide.columns = [f'{side.lower()}_{c}_champ' for c in wide.columns]
        return wide.reset_index()

    blue_roster = pivot_roster('Blue')
    red_roster  = pivot_roster('Red')
    blue_champs = pivot_champions('Blue')
    red_champs  = pivot_champions('Red')

    blue = df[(df['position'] == 'team') & (df['side'] == 'Blue')].copy()
    blue = blue.rename(columns={
        'teamname': 'blue_teamname',
        'ban1':  'blue_ban1',  'ban2':  'blue_ban2',  'ban3':  'blue_ban3',
        'ban4':  'blue_ban4',  'ban5':  'blue_ban5',
    })

    red_keep = {
        'gameid': 'gameid', 'teamname': 'red_teamname',
        'ban1':  'red_ban1',  'ban2':  'red_ban2',  'ban3':  'red_ban3',
        'ban4':  'red_ban4',  'ban5':  'red_ban5',
    }
    red = (df[(df['position'] == 'team') & (df['side'] == 'Red')]
           [list(red_keep)].rename(columns=red_keep))

    match_df = (blue
                .merge(red,         on='gameid', how='inner')
                .merge(blue_roster, on='gameid', how='left')
                .merge(red_roster,  on='gameid', how='left')
                .merge(blue_champs, on='gameid', how='left')
                .merge(red_champs,  on='gameid', how='left'))

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
    """Build category lists for all categorical columns across all models.

    Returns a dict with:
      '__champion_cats__' : sorted list of known champion names + '__rare__'
      '__player_cats__'   : sorted list of known player names  + '__rare__'
      'position_cats'     : list of valid position strings
      '{col}_cats'        : category list for each context column
      '_rare_champs'      : set of rare champion names (< 50 appearances)
      '_rare_players'     : set of rare player names  (< 50 appearances)
    """
    encoders = {}

    # Champions — global rarity across all pick + ban slots
    all_champs = pd.concat([match_df[c].dropna() for c in CHAMP_COLS]).astype(str)
    rare_champs = set(all_champs.value_counts().pipe(lambda s: s[s < 50]).index)
    encoders['_rare_champs']      = rare_champs
    encoders['__champion_cats__'] = sorted(set(all_champs) - rare_champs) + ['__rare__']

    # Players — global rarity across all roster slots + individual playername
    player_sources = [match_df[c].dropna() for c in PLAYER_COLS]
    if player_df is not None and 'playername' in player_df.columns:
        player_sources.append(player_df['playername'].dropna())
    all_players = pd.concat(player_sources).astype(str)
    rare_players = set(all_players.value_counts().pipe(lambda s: s[s < 50]).index)
    encoders['_rare_players']    = rare_players
    encoders['__player_cats__']  = sorted(set(all_players) - rare_players) + ['__rare__']

    # Position
    encoders['position_cats'] = POSITIONS[:]

    # Context columns — include __rare__ so unknown values at inference fall back safely
    for col in ['blue_teamname', 'red_teamname', 'league', 'split']:
        if col in match_df.columns:
            encoders[f'{col}_cats'] = sorted(match_df[col].dropna().astype(str).unique().tolist()) + ['__rare__']

    # Frequency tables (for use at inference when freq is unknown)
    if player_df is not None:
        encoders['_champion_freq'] = player_df['champion'].value_counts().to_dict()
        encoders['_player_freq']   = player_df['playername'].value_counts().to_dict()
    else:
        encoders['_champion_freq'] = {}
        encoders['_player_freq']   = {}

    return encoders


def _get_cats(col, encoders):
    """Return (categories_list, rare_set) for a categorical column, or (None, None) for numeric."""
    if col in CHAMP_COLS or col == 'champion':
        return encoders['__champion_cats__'], encoders['_rare_champs']
    if col in PLAYER_COLS or col == 'playername':
        return encoders['__player_cats__'], encoders['_rare_players']
    if col == 'position':
        return encoders['position_cats'], set()
    cats_key = f'{col}_cats'
    if cats_key in encoders:
        return encoders[cats_key], set()
    return None, None


def _apply_encoders(df, cols, encoders):
    """Set categorical columns to pd.Categorical dtype with rare-value bucketing."""
    df = df.copy()
    for col in cols:
        cats, rare = _get_cats(col, encoders)
        if cats is None:
            continue
        cats_set = set(cats)
        df[col] = pd.Categorical(
            df[col].astype(str).apply(lambda v: '__rare__' if v in rare or v not in cats_set else v),
            categories=cats
        )
    return df


# ---------------------------------------------------------------------------
# Shared model fitting / evaluation
# ---------------------------------------------------------------------------
def make_json_safe(obj):
    """
    Recursively convert NumPy / pandas objects into JSON-serializable Python types.
    """
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if pd.isna(obj):
        return None

    return obj

def save_results_json(results, filename='training_results.json'):
    """
    Save model metrics and metadata to a JSON file.
    """
    path = os.path.join(RESULTS_ROOT, filename)

    safe_results = make_json_safe(results)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(safe_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved evaluation results → {path}")

def make_shared_match_split(
    match_df,
    features=None,
    target='result',
    test_size=0.2,
    random_state=42,
):
    """
    Create one shared train/test split at the match level.

    This makes the full model and all baseline models evaluate on the same
    games, instead of each model creating its own split.
    """
    if features is None:
        features = MATCH_FEATURES_15MIN

    required_cols = list(dict.fromkeys(features + [target, 'sample_weight']))
    required_cols = [c for c in required_cols if c in match_df.columns]

    valid = match_df.dropna(subset=required_cols).copy()

    train_idx, test_idx = train_test_split(
        valid.index,
        test_size=test_size,
        random_state=random_state,
        stratify=valid[target],
    )

    print(
        f"Shared split: {len(train_idx):,} train matches, "
        f"{len(test_idx):,} test matches"
    )

    return train_idx, test_idx


def evaluate_classifier_metrics_from_predictions(
    y_test,
    y_prob,
    y_pred,
    label='',
    sample_weight=None,
):
    """
    Evaluate binary classification predictions where:
        1 = Blue win
        0 = Red win

    This function assumes predictions are already computed.
    """
    metrics = {
        'accuracy': accuracy_score(
            y_test,
            y_pred,
            sample_weight=sample_weight,
        ),
        'balanced_accuracy': balanced_accuracy_score(
            y_test,
            y_pred,
            sample_weight=sample_weight,
        ),
        'roc_auc': roc_auc_score(
            y_test,
            y_prob,
            sample_weight=sample_weight,
        ),
        'log_loss': log_loss(
            y_test,
            y_prob,
            sample_weight=sample_weight,
        ),
        'brier_score': brier_score_loss(
            y_test,
            y_prob,
            sample_weight=sample_weight,
        ),
        'blue_side_baseline_accuracy': (
            float(np.average(y_test, weights=sample_weight))
            if sample_weight is not None
            else float(np.mean(y_test))
        ),
    }

    cm = confusion_matrix(
        y_test,
        y_pred,
        sample_weight=sample_weight,
    )

    print(f"\n=== {label} Metrics ===")
    print(f"Accuracy:                  {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy:         {metrics['balanced_accuracy']:.4f}")
    print(f"ROC AUC:                   {metrics['roc_auc']:.4f}")
    print(f"Log loss:                  {metrics['log_loss']:.4f}")
    print(f"Brier score:               {metrics['brier_score']:.4f}")
    print(f"Blue-side baseline acc.:   {metrics['blue_side_baseline_accuracy']:.4f}")

    print("\nConfusion matrix:")
    print("Rows = actual, columns = predicted")
    print("[[Actual Red predicted Red,  Actual Red predicted Blue],")
    print(" [Actual Blue predicted Red, Actual Blue predicted Blue]]")
    print(cm)

    return metrics

def evaluate_classifier_once(
    model,
    X_test,
    y_test,
    w_test,
    label='',
):
    """
    Predict once, then compute both unweighted and weighted metrics.
    Returns predictions so group evaluations can reuse them.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print("\nUnweighted evaluation:")
    metrics_unweighted = evaluate_classifier_metrics_from_predictions(
        y_test=y_test,
        y_prob=y_prob,
        y_pred=y_pred,
        label=f"{label} / unweighted",
        sample_weight=None,
    )

    print("\nWeighted evaluation:")
    metrics_weighted = evaluate_classifier_metrics_from_predictions(
        y_test=y_test,
        y_prob=y_prob,
        y_pred=y_pred,
        label=f"{label} / weighted",
        sample_weight=w_test,
    )

    return {
        'y_prob': y_prob,
        'y_pred': y_pred,
        'metrics': {
            'unweighted': metrics_unweighted,
            'weighted': metrics_weighted,
        },
    }


def evaluate_by_group_from_predictions(
    y_test,
    y_prob,
    y_pred,
    group_values,
    group_name='group',
    sample_weight=None,
    min_group_size=20,
):
    """
    Print and return grouped metrics using already-computed predictions.
    """
    eval_df = pd.DataFrame({
        'y_true': np.asarray(y_test),
        'y_prob': np.asarray(y_prob),
        'y_pred': np.asarray(y_pred),
        group_name: np.asarray(group_values),
    })

    if sample_weight is not None:
        eval_df['sample_weight'] = np.asarray(sample_weight)

    print(f"\nMetrics by {group_name}:")

    group_results = {}

    for group, sub in eval_df.groupby(group_name):
        if pd.isna(group):
            continue

        if len(sub) < min_group_size:
            continue

        weights = (
            sub['sample_weight'].values
            if sample_weight is not None
            else None
        )

        has_both_classes = sub['y_true'].nunique() == 2

        acc = accuracy_score(
            sub['y_true'],
            sub['y_pred'],
            sample_weight=weights,
        )

        bal_acc = balanced_accuracy_score(
            sub['y_true'],
            sub['y_pred'],
            sample_weight=weights,
        )

        auc = (
            roc_auc_score(
                sub['y_true'],
                sub['y_prob'],
                sample_weight=weights,
            )
            if has_both_classes
            else np.nan
        )

        ll = log_loss(
            sub['y_true'],
            sub['y_prob'],
            sample_weight=weights,
            labels=[0, 1],
        )

        brier = brier_score_loss(
            sub['y_true'],
            sub['y_prob'],
            sample_weight=weights,
        )

        blue_baseline = (
            float(np.average(sub['y_true'], weights=weights))
            if weights is not None
            else float(np.mean(sub['y_true']))
        )

        row = {
            'n': len(sub),
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'roc_auc': auc,
            'log_loss': ll,
            'brier_score': brier,
            'blue_side_baseline_accuracy': blue_baseline,
        }

        group_results[str(group)] = row

        print(
            f"{group}: "
            f"n={len(sub):5d}, "
            f"acc={acc:.4f}, "
            f"bal_acc={bal_acc:.4f}, "
            f"auc={auc:.4f}, "
            f"logloss={ll:.4f}, "
            f"brier={brier:.4f}, "
            f"blue_base={blue_baseline:.4f}"
        )

    return group_results

def evaluate_standard_groups(
    model_eval,
    original_test_data,
    y_test,
    w_test,
    label='',
):
    """
    Evaluate grouped metrics using predictions already computed by _evaluate_classifier().

    Groups:
        - year
        - year_month
        - patch, if available
        - league
    """
    y_prob = model_eval['y_prob']
    y_pred = model_eval['y_pred']

    test_dates = None
    if 'date' in original_test_data.columns:
        test_dates = pd.to_datetime(
            original_test_data['date'],
            errors='coerce',
        )

    group_specs = []

    if test_dates is not None:
        group_specs.append((
            'year',
            test_dates.dt.year,
        ))

        group_specs.append((
            'year_month',
            test_dates.dt.to_period('M').astype(str),
        ))

    if 'patch' in original_test_data.columns:
        group_specs.append((
            'patch',
            original_test_data['patch'].astype(str),
        ))

    if 'league' in original_test_data.columns:
        group_specs.append((
            'league',
            original_test_data['league'],
        ))

    grouped_results = {}

    for group_name, group_values in group_specs:
        print(f"\n--- {label}: unweighted by {group_name} ---")
        unweighted = evaluate_by_group_from_predictions(
            y_test=y_test,
            y_prob=y_prob,
            y_pred=y_pred,
            group_values=group_values,
            group_name=group_name,
            sample_weight=None,
            min_group_size=20,
        )

        print(f"\n--- {label}: weighted by {group_name} ---")
        weighted = evaluate_by_group_from_predictions(
            y_test=y_test,
            y_prob=y_prob,
            y_pred=y_pred,
            group_values=group_values,
            group_name=group_name,
            sample_weight=w_test,
            min_group_size=20,
        )

        grouped_results[group_name] = {
            'unweighted': unweighted,
            'weighted': weighted,
        }

    return grouped_results

def _class_scale_weight(y):
    wins = y.sum()
    losses = len(y) - wins
    return losses / wins if wins > 0 else 1.0


def _fit_classifier(X_train, y_train, w_train, scale_weight, label='', n_estimators=200, lr=0.05):
    m = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=4, learning_rate=lr,
                           scale_pos_weight=scale_weight, random_state=42,
                           tree_method='hist', enable_categorical=True, n_jobs=-1,
                           callbacks=[_ProgressBar(n_estimators, label)])
    m.fit(X_train, y_train, sample_weight=w_train)
    return m


def _fit_regressor(X_train, y_train, w_train, label=''):
    n = 150
    m = xgb.XGBRegressor(n_estimators=n, max_depth=4, learning_rate=0.05, random_state=42,
                          tree_method='hist', enable_categorical=True, n_jobs=-1,
                          callbacks=[_ProgressBar(n, label)])
    m.fit(X_train, y_train, sample_weight=w_train)
    return m


def _evaluate_classifier(model, X_test, y_test, w_test, label):
    return evaluate_classifier_once(
        model=model,
        X_test=X_test,
        y_test=y_test,
        w_test=w_test,
        label=label,
    )

def train_classifier_on_features(
    match_df,
    encoders,
    features,
    train_idx,
    test_idx,
    label,
    target='result',
    save_name=None,
):
    """
    Train and evaluate an XGBoost classifier on a chosen feature set
    using a shared train/test split.

    This is used for:
        - full pre-match model
        - full 15-minute model
        - draft-only baseline
        - player/team-only baseline
        - team-only baseline
        - 15-minute-only baseline
    """
    features = [f for f in features if f in match_df.columns]
    all_cols = features + [target, 'sample_weight']

    data = match_df.loc[train_idx.union(test_idx)].dropna(subset=all_cols).copy()

    train_ids = data.index.intersection(train_idx)
    test_ids = data.index.intersection(test_idx)

    if len(train_ids) == 0 or len(test_ids) == 0:
        raise ValueError(
            f"{label}: empty train/test split after dropping missing values."
        )

    cat_cols = [
        c for c in features
        if c in CHAMP_COLS
        or c in PLAYER_COLS
        or c in ['blue_teamname', 'red_teamname', 'league', 'split']
    ]

    encoded = _apply_encoders(data, cat_cols, encoders)

    X_train = encoded.loc[train_ids, features]
    y_train = encoded.loc[train_ids, target]
    w_train = encoded.loc[train_ids, 'sample_weight']

    X_test = encoded.loc[test_ids, features]
    y_test = encoded.loc[test_ids, target]
    w_test = encoded.loc[test_ids, 'sample_weight']

    print(f"\n--- Training {label} ---")
    print(f"Features: {len(features)}")
    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows:  {len(X_test):,}")

    model = _fit_classifier(
        X_train,
        y_train,
        w_train,
        _class_scale_weight(y_train),
        label=label,
    )

    model_eval = _evaluate_classifier(
        model,
        X_test,
        y_test,
        w_test,
        label=label,
    )

    # Grouped evaluation using the same predictions.
    original_test_data = data.loc[test_ids]

    grouped_metrics = evaluate_standard_groups(
        model_eval=model_eval,
        original_test_data=original_test_data,
        y_test=y_test,
        w_test=w_test,
        label=label,
    )

    metrics = model_eval['metrics']
    metrics['groups'] = grouped_metrics

    if save_name is not None:
        _save(save_name, model, encoders)

    return model, metrics

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

def build_prematch_model(match_df, encoders, train_idx=None, test_idx=None):
    print("\n--- Pre-Match Game Outcome: Full Model ---")

    if train_idx is None or test_idx is None:
        train_idx, test_idx = make_shared_match_split(
            match_df,
            features=MATCH_FEATURES,
        )

    model, metrics = train_classifier_on_features(
        match_df=match_df,
        encoders=encoders,
        features=MATCH_FEATURES,
        train_idx=train_idx,
        test_idx=test_idx,
        label='pre-match full model',
        save_name='prematch',
    )

    return model, metrics

def build_prematch_baselines(match_df, encoders, train_idx, test_idx):
    """
    Train pre-match baseline models on the same split as the full model.
    """
    baselines = {}

    baseline_specs = {
        'prematch_draft_only': DRAFT_FEATURES,
        'prematch_player_team_only': PLAYER_TEAM_FEATURES,
        'prematch_team_only': TEAM_ONLY_FEATURES,
    }

    for name, features in baseline_specs.items():
        model, metrics = train_classifier_on_features(
            match_df=match_df,
            encoders=encoders,
            features=features,
            train_idx=train_idx,
            test_idx=test_idx,
            label=name,
            save_name=name,
        )

        baselines[name] = {
            'model': model,
            'metrics': metrics,
            'features': features,
        }

    return baselines

# ---------------------------------------------------------------------------
# Model 2 — 15-minute game outcome (same features + in-game stats)
# ---------------------------------------------------------------------------

def build_15min_model(match_df, encoders, train_idx=None, test_idx=None):
    print("\n--- 15-Minute Game Outcome: Full Model ---")

    if train_idx is None or test_idx is None:
        train_idx, test_idx = make_shared_match_split(
            match_df,
            features=MATCH_FEATURES_15MIN,
        )

    model, metrics = train_classifier_on_features(
        match_df=match_df,
        encoders=encoders,
        features=MATCH_FEATURES_15MIN,
        train_idx=train_idx,
        test_idx=test_idx,
        label='15-min full model',
        save_name='min15',
    )

    return model, metrics

def build_15min_baselines(match_df, encoders, train_idx, test_idx):
    """
    Train 15-minute baseline models on the same split as the full model.
    """
    baselines = {}

    baseline_specs = {
        'min15_stats_only': FIFTEEN_MIN_ONLY_FEATURES,
        'min15_prematch_only': MATCH_FEATURES,
        'min15_draft_plus_stats': DRAFT_FEATURES + STAT_15MIN,
        'min15_player_team_plus_stats': PLAYER_TEAM_FEATURES + STAT_15MIN,
    }

    for name, features in baseline_specs.items():
        model, metrics = train_classifier_on_features(
            match_df=match_df,
            encoders=encoders,
            features=features,
            train_idx=train_idx,
            test_idx=test_idx,
            label=name,
            save_name=name,
        )

        baselines[name] = {
            'model': model,
            'metrics': metrics,
            'features': features,
        }

    return baselines

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
    df = load_data('data')
    match_df = build_match_df(df)

    player_df = df[df['position'].isin(POSITIONS)]
    encoders = build_encoders(match_df, player_df)

    train_idx, test_idx = make_shared_match_split(
        match_df,
        features=MATCH_FEATURES_15MIN,
        target='result',
        test_size=0.2,
        random_state=42,
    )

    results = {
        'run_metadata': {
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'num_raw_rows': int(len(df)),
            'num_matches': int(len(match_df)),
            'num_train_matches': int(len(train_idx)),
            'num_test_matches': int(len(test_idx)),
            'target': 'result',
            'positive_class': 'Blue win',
            'test_size': 0.2,
            'random_state': 42,
            'uses_temporal_sample_weight_training': True,
            'reports_weighted_and_unweighted_metrics': True,
        },
        'models': {},
    }

    # Full match-outcome models.
    prematch_model, prematch_metrics = build_prematch_model(
        match_df,
        encoders,
        train_idx,
        test_idx,
    )

    results['models']['prematch_full'] = {
        'features': MATCH_FEATURES,
        'metrics': prematch_metrics,
    }

    min15_model, min15_metrics = build_15min_model(
        match_df,
        encoders,
        train_idx,
        test_idx,
    )

    results['models']['min15_full'] = {
        'features': MATCH_FEATURES_15MIN,
        'metrics': min15_metrics,
    }

    # Baselines.
    prematch_baselines = build_prematch_baselines(
        match_df,
        encoders,
        train_idx,
        test_idx,
    )

    for name, info in prematch_baselines.items():
        results['models'][name] = {
            'features': info['features'],
            'metrics': info['metrics'],
        }

    min15_baselines = build_15min_baselines(
        match_df,
        encoders,
        train_idx,
        test_idx,
    )

    for name, info in min15_baselines.items():
        results['models'][name] = {
            'features': info['features'],
            'metrics': info['metrics'],
        }

    # Player-performance models still print RMSE.
    # You can later refactor these to return metrics too.
    build_player_perf_model(df, match_df, encoders)
    build_player_perf_pregame_model(df, match_df, encoders)

    save_results_json(results, filename='training_results_latest.json')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_results_json(results, filename=f'training_results_{timestamp}.json')

    print("\nAll models and baselines trained and saved.")
