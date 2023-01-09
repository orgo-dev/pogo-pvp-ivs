from streamlit_javascript import st_javascript
import streamlit as st, pandas as pd
from config import PATH_DATA

MAX_LEVEL = 50
LEAGUE_CPS = dict(
    Little=500,
    Great=1500,
    Ultra=2500,
    Master=9e9,
)
IVS_ALL = [
    dict(iv_attack=a, iv_defense=d, iv_stamina=s)
    for a in range(16)
    for d in range(16)
    for s in range(16)
]


@st.cache
def load_app_db_constants():

    # pokemon stats
    df_pokes = pd.read_csv(f"{PATH_DATA}/pokemon.csv")
    POKE_STATS = df_pokes.set_index("pokemon")[
        ["base_attack", "base_defense", "base_stamina"]
    ].T.to_dict()

    # level stuff
    df_levels = pd.read_csv(f"{PATH_DATA}/levels.csv")
    df_levels["cp_coef"] = df_levels["cp_multiplier"] ** 2 * 0.1
    LEVELS = df_levels.level.to_list()
    CP_MULTS = df_levels.set_index("level").cp_multiplier.to_dict()
    cp_coefs = df_levels.cp_coef.values
    CP_COEF_PCTS = cp_coefs / cp_coefs.max()

    xls_default = df_levels[df_levels.level <= MAX_LEVEL]
    xls_best_buddy = df_levels[df_levels.level.between(MAX_LEVEL - 0.5, MAX_LEVEL)].copy()
    xls_best_buddy.level = xls_best_buddy.level + 1
    xls_cols = ["level"] + [
        f"{x}_xl_candy_total" for x in ["regular", "lucky", "shadow", "purified"]
    ]
    DF_XL_COSTS = pd.concat([xls_default, xls_best_buddy])[xls_cols]

    rename_move_cols = {
        "move": "Move",
        "type": "Type",
        "stab_bonus": "STAB",
        "damage": "Damage",
        "energy_gain": "Energy Gain",
        "turns": "Turns",
        "damage_per_turn": "DPT",
        "energy_per_turn": "EPT",
        "energy": "Energy",
        "damage_per_energy": "Damage Per Energy",
        "effect": "Effect",
        "stab_dpt": "STAB DPT",
        "stab_dpe": "STAB DPE",
        "archetype": "Archetype",
        "notes": "Notes",
    }

    df_fast = pd.read_csv(f"{PATH_DATA}/pokemon_fast_moves.csv")
    DF_POKEMON_FAST_MOVES = (
        df_fast.assign(damage=df_fast["damage"] * df_fast["stab_bonus"])
        .assign(damage_per_turn=df_fast["damage_per_turn"] * df_fast["stab_bonus"])
        .drop(["stab_bonus"], axis=1)
        .set_index("pokemon", drop=False)
        .rename(rename_move_cols, axis=1)
    )

    df_charged = pd.read_csv(f"{PATH_DATA}/pokemon_charged_moves.csv")
    DF_POKEMON_CHARGED_MOVES = (
        df_charged.assign(damage=df_charged["damage"] * df_charged["stab_bonus"])
        .assign(
            damage_per_energy=df_charged["damage_per_energy"] * df_charged["stab_bonus"]
        )
        .drop(["stab_bonus"], axis=1)
        .set_index("pokemon", drop=False)
        .rename(rename_move_cols, axis=1)
    )

    DF_POKEMON_TYPES = (
        pd.read_csv(f"{PATH_DATA}/pokemon_types.csv")
        .set_index("pokemon")
        .rename({"type": "Type"}, axis=1)
    )

    DF_POKEMON_TYPE_EFFECTIVENESS = (
        pd.read_csv(f"{PATH_DATA}/pokemon_types_effectiveness.csv")
        .set_index("pokemon")
        .rename({"attacking_type": "Type", "multiplier": "Effectiveness"}, axis=1)
    )

    return (
        POKE_STATS,
        # LEVELS,
        CP_MULTS,
        CP_COEF_PCTS,
        DF_XL_COSTS,
        DF_POKEMON_FAST_MOVES,
        DF_POKEMON_CHARGED_MOVES,
        DF_POKEMON_TYPES,
        DF_POKEMON_TYPE_EFFECTIVENESS,
    )


(
    ALL_POKEMON_STATS,
    # LEVELS,
    CP_MULTS,
    CP_COEF_PCTS,
    DF_XL_COSTS,
    DF_POKEMON_FAST_MOVES,
    DF_POKEMON_CHARGED_MOVES,
    DF_POKEMON_TYPES,
    DF_POKEMON_TYPE_EFFECTIVENESS,
) = load_app_db_constants()


def get_poke_fast_moves(pokemon):
    return (
        DF_POKEMON_FAST_MOVES.loc[pokemon]
        .drop(["pokemon", "move_kind"], axis=1)
        .reset_index(drop=True)
    )


def get_poke_charged_moves(pokemon):
    return (
        DF_POKEMON_CHARGED_MOVES.loc[pokemon]
        .drop(["pokemon", "move_kind"], axis=1)
        .reset_index(drop=True)
    )


def listify(o=None):
    if o is None:
        res = []
    elif isinstance(o, list):
        res = o
    elif isinstance(o, str):
        res = [o]
    else:
        res = [o]
    return res


def get_query_params_url(params_list, params_dict, **kwargs):
    """
    Create url params from alist of parameters and a dictionary with values.

    Args:
        params_list (str) :
            A list of parameters to get the value of from `params_dict`
        parmas_dict (dict) :
            A dict with values for the `parmas_list .
        **kwargs :
            Extra keyword args to add to the url
    """
    # get the url params
    url_params = {k: listify(params_dict[k])[0] for k in params_list}
    url_params.update(kwargs)
    url_params = parse_url_parameters(url_params)
    url_params_str = "&".join(f"{k}={v}" for k, v in url_params.items())

    # get the base url
    base_url = str(
        st_javascript(
            "await fetch('').then(r => window.parent.location.href)", key=url_params_str
        )
    ).split("?")[0]

    return f"{base_url}?{url_params_str}"


def parse_url_parameters(url_params):
    for k, v in url_params.items():
        if isinstance(v, str):
            url_params[k] = v.replace(" ", "%20")
        elif isinstance(v, (list, tuple, set)):
            url_params[k] = re.sub(r"[\'\"[\](){} ]", "", str(v))
    return url_params
