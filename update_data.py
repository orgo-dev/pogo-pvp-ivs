import os, json, fire, requests, pathlib, collections, pandas as pd, numpy as np
from config import PATH_DATA


def main():
    df_pokemons = update_pokemon()
    df_moves = update_moves()
    df_pokemon_types = update_pokemon_types(df_pokemons)
    update_pokemon_types_effectiveness(df_pokemons, df_pokemon_types)
    update_pokemon_moves(df_pokemons, df_moves, df_pokemon_types)


def update_pokemon():

    # read and save gamemaster pokemon json
    gamemaster_pokemon_json_url = "https://raw.githubusercontent.com/pvpoke/pvpoke/master/src/data/gamemaster/pokemon.json"
    pokemons_json = requests.get(gamemaster_pokemon_json_url).json()
    with open(PATH_DATA / "pokemon.json", "w", encoding="utf-8") as f:
        json.dump(pokemons_json, f, indent=4)

    # convert json to df
    df_pokemons = pd.concat(
        [pd.DataFrame({k: [v] for k, v in p.items()}) for p in pokemons_json],
        ignore_index=True,
    )

    # add FRUSTRATION move to shadow pokemon charged moves
    mask_shadow_pokemon = df_pokemons["speciesName"].str.contains("\(Shadow\)")
    frustration_move = mask_shadow_pokemon.map({False: [], True: ["FRUSTRATION"]})
    df_pokemons["chargedMoves"] = df_pokemons["chargedMoves"] + frustration_move

    # add RETURN move to purifiable pokemon charged moves
    # 1. get shadow species id without the "_shadow", e.g. "mewtwo_shadow" > "mewtwo"
    # 2. map those to ["RETURN"], otherwise []
    # 3. add to current charged moves
    purifiable_pokes = collections.defaultdict(lambda: [])
    for p in df_pokemons[mask_shadow_pokemon]["speciesId"]:
        purifiable_pokes[p[:-7]] = ["RETURN"]
    purify_move = df_pokemons["speciesId"].map(purifiable_pokes)
    df_pokemons["chargedMoves"] = df_pokemons["chargedMoves"] + purify_move

    # add and rename columns
    df_pokemons[["base_attack", "base_defense", "base_stamina"]] = df_pokemons[
        "baseStats"
    ].apply(pd.Series)[["atk", "def", "hp"]]
    df_pokemons[["family_id", "evolutions", "parent"]] = df_pokemons["family"].apply(
        pd.Series
    )[["id", "evolutions", "parent"]]
    df_pokemons.rename(
        columns={
            "speciesName": "pokemon",
            "speciesId": "pokemon_id",
            "fastMoves": "fast_moves",
            "chargedMoves": "charged_moves",
            "defaultIVs": "default_ivs",
            "buddyDistance": "buddy_distance",
            "thirdMoveCost": "third_move_cost",
            "released": "released",
            "family": "family",
            "eliteMoves": "elite_moves",
            "searchPriority": "search_priority",
            "legacyMoves": "legacy_moves",
            "levelFloor": "level_floor",
        },
        inplace=True,
    )

    # convert obj cols to strs for csv
    obj_cols = [
        "types",
        "fast_moves",
        "charged_moves",
        "elite_moves",
        "legacy_moves",
        "evolutions",
        "tags",
    ]
    df_pokemons_csv = df_pokemons.copy(deep=True)
    with pd.option_context("mode.chained_assignment", None):
        df_pokemons_csv[obj_cols].fillna("[]", inplace=True)
        df_pokemons_csv[obj_cols] = df_pokemons_csv[obj_cols].astype(str)

    # output to csv
    pokemons_out_cols = [
        "dex",
        "pokemon_id",
        "pokemon",
        "base_attack",
        "base_defense",
        "base_stamina",
        "types",
        "fast_moves",
        "charged_moves",
        "elite_moves",
        "legacy_moves",
        "buddy_distance",
        "third_move_cost",
        "released",
        "family_id",
        "evolutions",
        "parent",
        "tags",
    ]
    df_pokemons_csv[pokemons_out_cols].to_csv(PATH_DATA / "pokemon.csv", index=False)

    return df_pokemons


def get_unstacked_df(df, repeat_cols, lst_col, rename=None):
    return pd.DataFrame(
        {col: np.repeat(df[col].values, df[lst_col].str.len()) for col in repeat_cols}
    ).assign(**{rename or lst_col: np.concatenate(df[lst_col].values)})


def update_pokemon_types(df_pokemons):
    df_pokemon_types = get_unstacked_df(df_pokemons, ["pokemon"], "types", "type")
    df_pokemon_types.to_csv(PATH_DATA / "pokemon_types.csv", index=False)
    return df_pokemon_types


def update_pokemon_types_effectiveness(df_pokemons, df_pokemon_types):
    df_type_effectiveness = pd.read_csv(PATH_DATA / "type_effectiveness.csv")
    pokemon_type_effectiveness_groups = df_pokemon_types.merge(
        df_type_effectiveness, "left", left_on="type", right_on="defending_type"
    ).groupby(["pokemon", "attacking_type"], sort=False)
    df = (
        (
            pokemon_type_effectiveness_groups.min()["multiplier"]
            * pokemon_type_effectiveness_groups.max()["multiplier"]
        )
        .round(3)
        .reset_index()
    )
    df.to_csv(PATH_DATA / "pokemon_types_effectiveness.csv", index=False)


def update_moves():
    gamemaster_moves_json_url = "https://raw.githubusercontent.com/pvpoke/pvpoke/master/src/data/gamemaster/moves.json"
    moves_json = requests.get(gamemaster_moves_json_url).json()
    with open(PATH_DATA / "moves.json", "w", encoding="utf-8") as f:
        json.dump(moves_json, f, indent=4)

    # convert json to df
    df_moves = pd.concat(
        [pd.DataFrame({k: [v] for k, v in m.items()}) for m in moves_json],
        ignore_index=True,
    )

    # rename cols
    df_moves.rename(
        columns={
            "moveId": "move_id",
            "name": "move",
            "power": "damage",
            "energyGain": "energy_gain",
            "buffTarget": "buff_target",
            "buffApplyChance": "buff_apply_chance",
            "buffsSelf": "buffs_self",
            "buffsOpponent": "buffs_opponent",
        },
        inplace=True,
    )

    # add calculated cols
    df_moves["move_kind"] = (df_moves["energy"] == 0).map(
        {True: "fast", False: "charged"}
    )
    mask_fast = df_moves["move_kind"] == "fast"
    mask_charged = df_moves["move_kind"] == "charged"
    df_moves["turns"] = df_moves["cooldown"] // 500
    df_moves.loc[mask_fast, "damage_per_turn"] = (
        df_moves["damage"] / df_moves["turns"]
    ).round(2)
    df_moves.loc[mask_fast, "energy_per_turn"] = (
        df_moves["energy_gain"] / df_moves["turns"]
    ).round(2)
    df_moves.loc[mask_charged, "damage_per_energy"] = (
        df_moves["damage"] / df_moves["energy"]
    ).round(2)
    df_moves["effect"] = df_moves.apply(lambda row: parse_move_buffs_effect(row), axis=1)

    # convert obj cols to strs for db
    obj_cols = [
        "buffs",
        "fast_moves",
        "charged_moves",
        "elite_moves",
        "legacy_moves",
        "evolutions",
        "tags",
    ]
    df_moves_csv = df_moves.copy(deep=True)
    df_moves_csv[["buffs", "buffs_self", "buffs_opponent"]] = df_moves_csv[
        ["buffs", "buffs_self", "buffs_opponent"]
    ].astype(str)

    moves_out_cols = [
        "move_id",
        "move",
        "move_kind",
        "type",
        "damage",
        "energy_gain",
        "turns",
        "damage_per_turn",
        "energy_per_turn",
        "energy",
        "damage_per_energy",
        "effect",
        "archetype",
        "buffs",
        "buff_target",
        "buff_apply_chance",
        "buffs_self",
        "buffs_opponent",
    ]

    # output to csv
    df_moves_csv[moves_out_cols].to_csv(PATH_DATA / "moves.csv", index=False)

    return df_moves


def update_pokemon_moves(df_pokemons, df_moves, df_pokemon_types):

    # elite moves
    mask_elite_moves = df_pokemons["elite_moves"].notna()
    df_pokemon_elite_moves = get_unstacked_df(
        df_pokemons[mask_elite_moves], ["pokemon"], "elite_moves", "elite_move_id"
    )

    # legacy moves
    mask_legacy_moves = df_pokemons["legacy_moves"].notna()
    df_pokemon_legacy_moves = get_unstacked_df(
        df_pokemons[mask_legacy_moves], ["pokemon"], "legacy_moves", "legacy_move_id"
    )

    # fast moves
    df_pokemon_fast_moves = (
        get_unstacked_df(df_pokemons, ["pokemon"], "fast_moves", "fast_move")
        .merge(df_moves, how="left", left_on="fast_move", right_on="move_id")
        .merge(
            df_pokemon_elite_moves,
            how="left",
            left_on=["pokemon", "move_id"],
            right_on=["pokemon", "elite_move_id"],
        )
        .merge(
            df_pokemon_legacy_moves,
            how="left",
            left_on=["pokemon", "move_id"],
            right_on=["pokemon", "legacy_move_id"],
        )
        .merge(
            df_pokemon_types,
            how="left",
            on=["pokemon", "type"],
            indicator=True,
        )
        .rename(columns={"_merge": "stab_bonus"})
    )
    df_pokemon_fast_moves["energy"] = df_pokemon_fast_moves["energy"].replace(0.0, np.nan)
    df_pokemon_fast_moves["stab_bonus"] = df_pokemon_fast_moves["stab_bonus"].map(
        {"left_only": 1.0, "both": 1.2}
    )
    df_pokemon_fast_moves["notes"] = df_pokemon_fast_moves.apply(
        lambda row: parse_pokemon_move_notes(row), axis=1
    )
    fast_out_cols = [
        "pokemon",
        "move",
        "move_kind",
        "type",
        "stab_bonus",
        "damage",
        "energy_gain",
        "turns",
        "damage_per_turn",
        "energy_per_turn",
        "archetype",
        "notes",
    ]
    df_pokemon_fast_moves.sort_values(["pokemon", "move"], inplace=True)
    df_pokemon_fast_moves[fast_out_cols].to_csv(
        PATH_DATA / "pokemon_fast_moves.csv", index=False
    )

    # charged moves
    df_pokemon_charged_moves = (
        get_unstacked_df(df_pokemons, ["pokemon"], "charged_moves", "charged_move")
        .merge(df_moves, how="left", left_on="charged_move", right_on="move_id")
        .merge(
            df_pokemon_elite_moves,
            how="left",
            left_on=["pokemon", "move_id"],
            right_on=["pokemon", "elite_move_id"],
        )
        .merge(
            df_pokemon_legacy_moves,
            how="left",
            left_on=["pokemon", "move_id"],
            right_on=["pokemon", "legacy_move_id"],
        )
        .merge(
            df_pokemon_types,
            how="left",
            on=["pokemon", "type"],
            indicator=True,
        )
        .rename(columns={"_merge": "stab_bonus"})
    )
    df_pokemon_charged_moves["stab_bonus"] = df_pokemon_charged_moves["stab_bonus"].map(
        {"left_only": 1.0, "both": 1.2}
    )
    df_pokemon_charged_moves["notes"] = df_pokemon_charged_moves.apply(
        lambda row: parse_pokemon_move_notes(row), axis=1
    )
    charged_out_cols = [
        "pokemon",
        "move",
        "move_kind",
        "type",
        "stab_bonus",
        "damage",
        "energy",
        "damage_per_energy",
        "effect",
        "archetype",
        "notes",
    ]
    df_pokemon_charged_moves.sort_values(["pokemon", "move"], inplace=True)
    df_pokemon_charged_moves[charged_out_cols].to_csv(
        PATH_DATA / "pokemon_charged_moves.csv", index=False
    )

    # combined moves
    df_pokemon_combined_moves = pd.concat(
        [df_pokemon_fast_moves, df_pokemon_charged_moves]
    )
    df_pokemon_combined_moves["move_kind_sort"] = df_pokemon_combined_moves[
        "move_kind"
    ].map({"fast": 0, "charged": 1})
    df_pokemon_combined_moves.sort_values(
        ["pokemon", "move_kind_sort", "move"], inplace=True
    )
    combined_out_cols = [
        "pokemon",
        "move",
        "move_kind",
        "type",
        "stab_bonus",
        "damage",
        "energy_gain",
        "turns",
        "damage_per_turn",
        "energy_per_turn",
        "energy",
        "damage_per_energy",
        "effect",
        "archetype",
        "notes",
    ]
    df_pokemon_combined_moves[combined_out_cols].to_csv(
        PATH_DATA / "pokemon_moves.csv", index=False
    )


def parse_move_buffs_effect(r):
    if r["buffs"] is np.nan:
        return ""
    # get chance
    str_chance = (
        str(float(r["buff_apply_chance"]) * 100).replace("100.0", "100") + "% chance "
    )

    # get buff text
    if r["buff_target"] == "both":
        buff_text = [
            f"{buff:+} {stat}"
            for buff, stat in zip(r.buffs_self, ["Atk", "Def"])
            if buff != 0
        ]
        buff_text += ["self,"]
        buff_text += [
            f"{buff:+} {stat}"
            for buff, stat in zip(r.buffs_opponent, ["Atk", "Def"])
            if buff != 0
        ]
        buff_text += ["opponent"]
    else:
        buff_text = [
            f"{buff:+} {stat}" for buff, stat in zip(r.buffs, ["Atk", "Def"]) if buff != 0
        ]
        buff_text += [r.buff_target]
    str_buff = " ".join(buff_text)
    return str_chance + str_buff


def parse_pokemon_move_notes(r):
    notes = []
    for tag in ["elite", "legacy"]:
        if r[f"{tag}_move_id"] is not np.nan:
            notes.append(tag)
    return ", ".join(notes) if notes else ""


if __name__ == "__main__":
    fire.Fire(main)
