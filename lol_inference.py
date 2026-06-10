"""
lol_inference.py — Load all saved models and run full match predictions.

Run lol_xgboost_pipeline.py first to generate the .joblib files.

Usage:
    python lol_inference.py
    from lol_inference import load_models, predict_match
"""

import os
import joblib
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

POSITIONS = ['top', 'jng', 'mid', 'bot', 'sup']


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

def load_models():
    """Load all four trained models and the shared encoders from the project root."""
    names = ['prematch', 'min15', 'player_perf', 'player_perf_pregame']
    loaded = {}

    for name in names:
        path = os.path.join(ROOT, f'{name}.joblib')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}\nRun lol_xgboost_pipeline.py first."
            )
        loaded[name] = joblib.load(path)
        print(f"Loaded: {name}.joblib")

    # Encoders are saved alongside prematch and player_perf (they're identical)
    enc_path = os.path.join(ROOT, 'prematch_encoders.joblib')
    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"Missing encoders: {enc_path}")
    loaded['encoders'] = joblib.load(enc_path)
    print("Loaded: prematch_encoders.joblib")

    return loaded


# ---------------------------------------------------------------------------
# Encoding helper (mirrors pipeline)
# ---------------------------------------------------------------------------

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


def _enc(col, value, encoders):
    """Encode a single value using the shared encoder dict."""
    value = str(value)
    if col in CHAMP_COLS or col == 'champion':
        le, rare = encoders['__champion__'], encoders['_rare_champs']
    elif col in PLAYER_COLS or col == 'playername':
        le, rare = encoders['__player__'], encoders['_rare_players']
    elif col == 'position':
        le, rare = encoders['position'], set()
    elif col in encoders:
        le, rare = encoders[col], set()
    else:
        return value  # numeric — pass through
    if (value in rare or value not in le.classes_) and '__rare__' in le.classes_:
        value = '__rare__'
    elif value not in le.classes_:
        return 0  # position encoder: unknown value, fallback to 0
    return int(le.transform([value])[0])


# ---------------------------------------------------------------------------
# Flatten match_info → flat feature dict
# ---------------------------------------------------------------------------

def _flatten(match_info):
    """Convert the nested match_info dict into a flat feature dict."""
    blue = match_info['blue']
    red  = match_info['red']
    flat = {
        'blue_teamname': blue['team'],
        'red_teamname':  red['team'],
        'league':   match_info.get('league', 'unknown'),
        'split':    match_info.get('split',  'unknown'),
        'playoffs': int(match_info.get('playoffs', 0)),
    }
    for i, champ in enumerate(blue.get('picks', []), 1):
        flat[f'blue_pick{i}'] = champ
    for i, champ in enumerate(red.get('picks', []), 1):
        flat[f'red_pick{i}'] = champ
    for i, champ in enumerate(blue.get('bans', []), 1):
        flat[f'blue_ban{i}'] = champ
    for i, champ in enumerate(red.get('bans', []), 1):
        flat[f'red_ban{i}'] = champ
    for pos in POSITIONS:
        flat[f'blue_{pos}'] = blue['players'][pos]['name']
        flat[f'red_{pos}']  = red['players'][pos]['name']
    return flat


def _build_row(feature_cols, flat, encoders):
    """Build an encoded single-row DataFrame for model prediction."""
    row = {}
    for col in feature_cols:
        row[col] = _enc(col, flat.get(col, 0), encoders)
    return pd.DataFrame([row]).apply(pd.to_numeric, errors='coerce').fillna(0)


# ---------------------------------------------------------------------------
# Core prediction functions
# ---------------------------------------------------------------------------

def _predict_outcome(model, feature_cols, flat, encoders):
    X = _build_row(feature_cols, flat, encoders)
    prob_blue = float(model.predict_proba(X)[0][1])
    return {
        'winner':               'Blue' if prob_blue >= 0.5 else 'Red',
        'blue_win_probability': round(prob_blue, 4),
        'red_win_probability':  round(1 - prob_blue, 4),
    }


def _predict_player(perf_models, feature_cols, flat, encoders, player_info, pos):
    """Predict DPM/KDA/gold for one player, mixing match context with player-specific fields."""
    player_flat = dict(flat)
    player_flat['playername']    = player_info['name']
    player_flat['champion']      = player_info['champion']
    player_flat['position']      = pos
    player_flat['player_freq']   = encoders['_player_freq'].get(player_info['name'], 1)
    player_flat['champion_freq'] = encoders['_champion_freq'].get(player_info['champion'], 1)
    # Optional 15-min individual stats
    for stat in ['golddiffat15', 'xpdiffat15', 'csdiffat15', 'killsat15', 'assistsat15', 'deathsat15']:
        if stat in player_info:
            player_flat[stat] = player_info[stat]

    X = _build_row(feature_cols, player_flat, encoders)
    return {
        'playername': player_info['name'],
        'champion':   player_info['champion'],
        'dpm':   round(float(perf_models['dpm'].predict(X)[0]),  2),
        'kda':   round(float(perf_models['kda'].predict(X)[0]),  3),
        'gold':  round(float(perf_models['gold'].predict(X)[0]), 0),
    }


# ---------------------------------------------------------------------------
# Unified predict_match
# ---------------------------------------------------------------------------

def predict_match(loaded_models, match_info, mode='prematch'):
    """Full match prediction: game outcome + all 10 players' DPM / KDA / gold.

    Parameters
    ----------
    loaded_models : dict returned by load_models()
    match_info    : nested dict describing the match (see example below)
    mode          : 'prematch' (uses draft + roster only)
                    '15min'    (also uses 15-min stats from match_info['stats'])

    match_info structure
    --------------------
    {
        'league': 'LCK', 'split': 'Spring', 'playoffs': 0,
        'stats': {                           # only needed for mode='15min'
            'golddiffat15': 2000, 'xpdiffat15': 1500, 'csdiffat15': 10,
            'killsat15': 5, 'assistsat15': 8, 'deathsat15': 2,
        },
        'blue': {
            'team': 'T1',
            'picks': ['Gnar', 'Vi', 'Azir', 'Jinx', 'Thresh'],   # draft order
            'bans':  ['Zed', 'Yasuo', 'Katarina', 'LeBlanc', 'Syndra'],
            'players': {
                'top': {'name': 'Zeus',     'champion': 'Gnar',
                        'golddiffat15': 500, 'xpdiffat15': 300, 'csdiffat15': 5,
                        'killsat15': 1, 'assistsat15': 2, 'deathsat15': 0},
                'jng': {'name': 'Oner',     'champion': 'Vi', ...},
                'mid': {'name': 'Faker',    'champion': 'Azir', ...},
                'bot': {'name': 'Gumayusi', 'champion': 'Jinx', ...},
                'sup': {'name': 'Keria',    'champion': 'Thresh', ...},
            }
        },
        'red': {
            'team': 'Gen.G',
            'picks': ['Jayce', 'Graves', 'Orianna', 'Aphelios', 'Lulu'],
            'bans':  ['Caitlyn', 'Lux', 'Ahri', 'Karma', 'Xayah'],
            'players': {
                'top': {'name': 'Doran',   'champion': 'Jayce', ...},
                'jng': {'name': 'Peanut',  'champion': 'Graves', ...},
                'mid': {'name': 'Chovy',   'champion': 'Orianna', ...},
                'bot': {'name': 'Peyz',    'champion': 'Aphelios', ...},
                'sup': {'name': 'Delight', 'champion': 'Lulu', ...},
            }
        }
    }

    Returns
    -------
    {
        'game_outcome': {'winner': 'Blue', 'blue_win_probability': 0.72, ...},
        'blue_team': {
            'top': {'playername': 'Zeus', 'champion': 'Gnar', 'dpm': 412.5, 'kda': 3.25, 'gold': 15200.0},
            ...
        },
        'red_team': { ... }
    }
    """
    encoders = loaded_models['encoders']
    flat     = _flatten(match_info)

    # Add team-level 15-min stats to flat dict if provided
    for stat, val in match_info.get('stats', {}).items():
        flat[stat] = val

    # --- Game outcome ---
    if mode == '15min':
        game_model = loaded_models['min15']
        perf_model_key = 'player_perf'
    else:
        game_model = loaded_models['prematch']
        perf_model_key = 'player_perf_pregame'

    game_feature_cols = list(game_model.get_booster().feature_names)
    outcome = _predict_outcome(game_model, game_feature_cols, flat, encoders)

    # --- Player performance ---
    perf_models      = loaded_models[perf_model_key]
    perf_feature_cols = list(next(iter(perf_models.values())).get_booster().feature_names)

    result = {'game_outcome': outcome, 'blue_team': {}, 'red_team': {}}
    for side_key, side_label in [('blue', 'blue_team'), ('red', 'red_team')]:
        for pos in POSITIONS:
            player_info = match_info[side_key]['players'][pos]
            result[side_label][pos] = _predict_player(
                perf_models, perf_feature_cols, flat, encoders, player_info, pos
            )

    return result


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def _print_result(result):
    o = result['game_outcome']
    print(f"  Winner: {o['winner']}  "
          f"(Blue {o['blue_win_probability']:.1%} / Red {o['red_win_probability']:.1%})")
    for side in ['blue_team', 'red_team']:
        label = 'Blue' if side == 'blue_team' else 'Red'
        print(f"  {label} team:")
        for pos, stats in result[side].items():
            print(f"    {pos:3s}  {stats['playername']:12s}  {stats['champion']:12s}"
                  f"  DPM={stats['dpm']:6.0f}  KDA={stats['kda']:.2f}  Gold={stats['gold']:,.0f}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loaded = load_models()
    print()

    match_info = {
        'league': 'LCK', 'split': 'Spring', 'playoffs': 0,
        'stats': {
            'golddiffat15': 2000, 'xpdiffat15': 1500, 'csdiffat15': 10,
            'killsat15': 5, 'assistsat15': 8, 'deathsat15': 2,
        },
        'blue': {
            'team': 'T1',
            'picks': ['Gnar', 'Vi', 'Azir', 'Jinx', 'Thresh'],
            'bans':  ['Zed', 'Yasuo', 'Katarina', 'LeBlanc', 'Syndra'],
            'players': {
                'top': {'name': 'Zeus',     'champion': 'Gnar',
                        'golddiffat15': 500,  'xpdiffat15': 300, 'csdiffat15': 5,
                        'killsat15': 1, 'assistsat15': 2, 'deathsat15': 0},
                'jng': {'name': 'Oner',     'champion': 'Vi',
                        'golddiffat15': 200,  'xpdiffat15': 100, 'csdiffat15': 3,
                        'killsat15': 2, 'assistsat15': 4, 'deathsat15': 1},
                'mid': {'name': 'Faker',    'champion': 'Azir',
                        'golddiffat15': 800,  'xpdiffat15': 600, 'csdiffat15': 12,
                        'killsat15': 2, 'assistsat15': 1, 'deathsat15': 0},
                'bot': {'name': 'Gumayusi', 'champion': 'Jinx',
                        'golddiffat15': 400,  'xpdiffat15': 400, 'csdiffat15': 15,
                        'killsat15': 0, 'assistsat15': 1, 'deathsat15': 1},
                'sup': {'name': 'Keria',    'champion': 'Thresh',
                        'golddiffat15': 100,  'xpdiffat15': 100, 'csdiffat15': 0,
                        'killsat15': 0, 'assistsat15': 0, 'deathsat15': 0},
            },
        },
        'red': {
            'team': 'Gen.G',
            'picks': ['Jayce', 'Graves', 'Orianna', 'Aphelios', 'Lulu'],
            'bans':  ['Caitlyn', 'Lux', 'Ahri', 'Karma', 'Xayah'],
            'players': {
                'top': {'name': 'Doran',   'champion': 'Jayce',
                        'golddiffat15': -500, 'xpdiffat15': -300, 'csdiffat15': -5,
                        'killsat15': 0, 'assistsat15': 1, 'deathsat15': 1},
                'jng': {'name': 'Peanut',  'champion': 'Graves',
                        'golddiffat15': -200, 'xpdiffat15': -100, 'csdiffat15': -3,
                        'killsat15': 1, 'assistsat15': 2, 'deathsat15': 2},
                'mid': {'name': 'Chovy',   'champion': 'Orianna',
                        'golddiffat15': -800, 'xpdiffat15': -600, 'csdiffat15': -12,
                        'killsat15': 0, 'assistsat15': 2, 'deathsat15': 2},
                'bot': {'name': 'Peyz',    'champion': 'Aphelios',
                        'golddiffat15': -400, 'xpdiffat15': -400, 'csdiffat15': -15,
                        'killsat15': 1, 'assistsat15': 1, 'deathsat15': 0},
                'sup': {'name': 'Delight', 'champion': 'Lulu',
                        'golddiffat15': -100, 'xpdiffat15': -100, 'csdiffat15': 0,
                        'killsat15': 0, 'assistsat15': 3, 'deathsat15': 0},
            },
        },
    }

    print("=== Pre-Match Prediction ===")
    result_pre = predict_match(loaded, match_info, mode='prematch')
    _print_result(result_pre)

    print("\n=== 15-Minute Prediction ===")
    result_15 = predict_match(loaded, match_info, mode='15min')
    _print_result(result_15)
