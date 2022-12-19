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

    return (
        POKE_STATS,
        # LEVELS,
        CP_MULTS,
        CP_COEF_PCTS,
        DF_XL_COSTS,
        DF_POKEMON_FAST_MOVES,
        DF_POKEMON_CHARGED_MOVES,
    )


(
    ALL_POKEMON_STATS,
    # LEVELS,
    CP_MULTS,
    CP_COEF_PCTS,
    DF_XL_COSTS,
    DF_POKEMON_FAST_MOVES,
    DF_POKEMON_CHARGED_MOVES,
) = load_app_db_constants()
