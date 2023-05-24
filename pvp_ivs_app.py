import streamlit as st
import math, re, sqlite3, pandas as pd, numpy as np
from bisect import bisect
from st_aggrid import (
    GridOptionsBuilder,
    AgGrid,
    GridUpdateMode,
    DataReturnMode,
    JsCode,
    ColumnsAutoSizeMode,
)
from app_utils import (
    get_query_params_url,
    get_poke_fast_moves,
    get_poke_charged_moves,
    load_app_db_constants,
    MAX_LEVEL,
    LEAGUE_CPS,
    IVS_ALL,
)


def calc_level_stats(
    level,
    iv_attack,
    iv_defense,
    iv_stamina,
    base_attack,
    base_defense,
    base_stamina,
    cp_mults,
):
    cp_multiplier = cp_mults[level]
    level_attack = (base_attack + iv_attack) * cp_multiplier
    level_defense = (base_defense + iv_defense) * cp_multiplier
    level_stamina = (base_stamina + iv_stamina) * cp_multiplier
    cp = max(
        10, math.floor(0.1 * level_attack * level_defense**0.5 * level_stamina**0.5)
    )
    return dict(
        level=level,
        level_attack=level_attack,
        level_defense=level_defense,
        level_stamina=math.floor(level_stamina),
        cp=cp,
    )


def calc_poke_level_stats(
    pokemon,
    level,
    iv_attack,
    iv_defense,
    iv_stamina,
    all_pokemon_stats,
    cp_mults,
):
    pokemon_dict = dict(
        pokemon=pokemon,
        level=level,
        iv_attack=iv_attack,
        iv_defense=iv_defense,
        iv_stamina=iv_stamina,
    )
    stats = calc_level_stats(
        level,
        iv_attack,
        iv_defense,
        iv_stamina,
        **all_pokemon_stats[pokemon],
        cp_mults=cp_mults,
    )
    return {**pokemon_dict, **stats}


def find_league_iv_level_stats(
    league,
    pokemon,
    iv_attack,
    iv_defense,
    iv_stamina,
    all_pokemon_stats,
    cp_mults,
    cp_coef_pcts,
    max_level=MAX_LEVEL,
    league_cps=LEAGUE_CPS,
    add_non_best_buddy_stats=True,
    log=False,
):
    stats_args = dict(
        pokemon=pokemon,
        iv_attack=iv_attack,
        iv_defense=iv_defense,
        iv_stamina=iv_stamina,
        all_pokemon_stats=all_pokemon_stats,
        cp_mults=cp_mults,
    )
    max_cp = league_cps[league]
    cp_high = calc_poke_level_stats(**stats_args, level=55)["cp"]
    pct_to_cp_high = max_cp / cp_high
    stats, rtn = [], []
    level = min(bisect(cp_coef_pcts, pct_to_cp_high) / 2 + 1, max_level)

    for i in range(100):
        stats.append(calc_poke_level_stats(**stats_args, level=level))

        if stats[i]["cp"] == max_cp:
            rtn.append(stats[i])
            break

        elif stats[i]["cp"] > max_cp:
            if i > 0 and stats[i]["cp"] > stats[i - 1]["cp"]:
                rtn.append(stats[i - 1])
                break
            level -= 0.5

        elif stats[i]["cp"] < max_cp:
            if stats[i]["level"] == max_level + 1:
                rtn.append(stats[i])
                break
            elif i > 0 and stats[i]["cp"] < stats[i - 1]["cp"]:
                rtn.append(stats[i])
                break
            level += 0.5

    if rtn[0]["level"] > 50 and add_non_best_buddy_stats:
        rtn.append(calc_poke_level_stats(**stats_args, level=rtn[0]["level"] - 1))

    return [dict(league=league, **x) for x in rtn]


def get_league_pokemon_all_ivs_stats(
    league,
    pokemon,
    all_pokemon_stats,
    cp_mults,
    cp_coef_pcts,
    IVS_ALL=IVS_ALL,
):
    results = []
    for ivs in IVS_ALL:
        results.extend(
            find_league_iv_level_stats(
                league,
                pokemon,
                all_pokemon_stats=all_pokemon_stats,
                cp_mults=cp_mults,
                cp_coef_pcts=cp_coef_pcts,
                **ivs,
            )
        )
    return pd.DataFrame(results)


def add_pokemon_iv_notes(row):
    notes = []
    if row["is_max_stats"]:
        notes.append("#1 stats")
    if row["is_max_bulk"]:
        notes.append("#1 bulk")
    if row["is_level_max_stats"]:
        notes.append("level max stats")
    if row["is_max_attack"]:
        notes.append("max attack")
    if row["is_max_defense"]:
        notes.append("max defense")
    if row["is_max_stamina"]:
        notes.append("max stamina")
    if row["iv_attack"] == 15 and row["iv_defense"] == 15 and row["iv_stamina"] == 15:
        notes.append("100% iv")
    if row["level"] > 50:
        notes.append("best buddy")
    return ", ".join(notes)


def parse_ivs(ivs):
    if not ivs:
        return []
    iv_names = ["iv_attack", "iv_defense", "iv_stamina"]
    return [
        dict(zip(iv_names, map(int, re.findall(r"\d+", iv)))) for iv in ivs.split(",")
    ]


def get_pareto_efficient_stats(stats):
    is_inefficient = np.ones(stats.shape[0], dtype=bool)
    for i, s in enumerate(stats):
        other_stats = np.r_[stats[:i], stats[i + 1 :]]
        is_inefficient[i] = np.any(
            # check any other stats where all are >= than current
            np.all(other_stats >= s, axis=1)
            # but make sure the stats aren't all equal
            & np.any(other_stats != s, axis=1)
        )
    return ~is_inefficient


def get_league_pokemon_df(
    league,
    pokemon,
    ivs,
    all_pokemon_stats,
    cp_mults,
    cp_coef_pcts,
    df_xl_costs,
):
    # stats df
    df = get_league_pokemon_all_ivs_stats(
        league, pokemon, all_pokemon_stats, cp_mults, cp_coef_pcts
    )

    # add ivs col
    iv_cols = ["iv_attack", "iv_defense", "iv_stamina"]
    df["IVs"] = df[iv_cols].apply(lambda row: "/".join(row.values.astype(str)), axis=1)

    # stats/bulk products
    df["stats_prod"] = (
        df[["level_attack", "level_defense", "level_stamina"]].product(axis=1).round(1)
    )
    df["bulk_prod"] = df[["level_defense", "level_stamina"]].product(axis=1).round(1)
    df["pct_max_stats_product"] = (df["stats_prod"] / df["stats_prod"].max() * 100).round(
        1
    )

    # ranks
    rank_cols = ["stats_prod", "level_attack", "level_defense", "cp", "iv_stamina"]
    rank_indices = df.sort_values(rank_cols, ascending=False).index
    df["rank"] = rank_indices.argsort() + 1
    rank_cols = ["bulk_prod", "level_attack", "level_defense", "cp", "iv_stamina"]
    df["rank_bulk"] = df.sort_values(rank_cols, ascending=False).index.argsort() + 1

    # add max stat bools
    df["is_max_iv"] = (
        (df["iv_attack"] == 15) & (df["iv_defense"] == 15) & (df["iv_stamina"] == 15)
    )
    df["is_max_stats"] = df["stats_prod"] == df["stats_prod"].max()
    df["is_max_bulk"] = df["bulk_prod"] == df["bulk_prod"].max()
    df["is_level_max_stats"] = (
        df.groupby(["level"])["stats_prod"].rank(method="min", ascending=False) == 1
    )
    df["is_level_max_bulk"] = (
        df.groupby(["level"])["bulk_prod"].rank(method="min", ascending=False) == 1
    )
    df["is_max_attack"] = df["level_attack"] == df["level_attack"].max()
    df["is_max_defense"] = df["level_defense"] == df["level_defense"].max()
    df["is_max_stamina"] = df["level_stamina"] == df["level_stamina"].max()

    # add pareto efficient bools for max level and best buddy
    stat_cols = ["level_attack", "level_defense", "level_stamina"]
    df["is_efficient_best_buddy"] = get_pareto_efficient_stats(df[stat_cols].values)
    if df["level"].max() > MAX_LEVEL:
        max_level_idx = df[df.level <= MAX_LEVEL].index
        df.loc[max_level_idx, "is_efficient_max_level"] = get_pareto_efficient_stats(
            df.loc[max_level_idx, stat_cols].values
        )
        df["is_efficient_max_level"].fillna(False, inplace=True)
    else:
        df["is_efficient_max_level"] = df["is_efficient_best_buddy"]
    display_true = {True: "True", False: ""}
    df[f"Efficient @{MAX_LEVEL}"] = df["is_efficient_max_level"].map(display_true)
    df[f"Efficient @{MAX_LEVEL+1}"] = df["is_efficient_best_buddy"].map(display_true)

    # add r1 cmp
    rank1_attack = df.at[rank_indices[0], "level_attack"]
    df["r1_cmp"] = np.where(
        df["level_attack"] == rank1_attack,
        "T",
        np.where(df["level_attack"] > rank1_attack, "W", "L"),
    )

    # add notes
    df["notes"] = df.apply(add_pokemon_iv_notes, axis=1)

    # add xl costs
    df = df.merge(df_xl_costs, how="left", on="level")

    # add col for selected ivs
    df_input_ivs = (
        pd.DataFrame(parse_ivs(ivs)).assign(Input="True")
        if ivs
        else pd.DataFrame(columns=["iv_attack", "iv_defense", "iv_stamina", "Input"])
    )
    df = df.merge(df_input_ivs, how="left", on=["iv_attack", "iv_defense", "iv_stamina"])
    df["Input"].fillna("", inplace=True)

    # return sorted, renamed cols
    return (
        df.sort_values("rank")
        .reset_index(drop=True)
        .rename(
            {
                "rank": "Rank",
                "level": "Level",
                "cp": "CP",
                "iv_attack": "IV Atk",
                "iv_defense": "IV Def",
                "iv_stamina": "IV HP",
                "level_attack": "Atk",
                "level_defense": "Def",
                "level_stamina": "HP",
                "stats_prod": "Stats Prod",
                "pct_max_stats_product": "Pct Max Stats",
                "bulk_prod": "Bulk Prod",
                "rank_bulk": "Rank Bulk",
                "notes": "Notes",
                "r1_cmp": "R1 CMP",
                "regular_xl_candy_total": "Regular XLs",
                "lucky_xl_candy_total": "Lucky XLs",
                "shadow_xl_candy_total": "Shadow XLs",
                "purified_xl_candy_total": "Purified XLs",
            },
            axis=1,
        )[
            [
                "Rank",
                "Level",
                "CP",
                "IVs",
                "IV Atk",
                "IV Def",
                "IV HP",
                "Stats Prod",
                "R1 CMP",
                "Pct Max Stats",
                "Bulk Prod",
                "Rank Bulk",
                "Atk",
                "Def",
                "HP",
                f"Efficient @{MAX_LEVEL}",
                f"Efficient @{MAX_LEVEL+1}",
                "Notes",
                "Input",
                "Regular XLs",
                "Lucky XLs",
                "Shadow XLs",
                "Purified XLs",
                "is_max_iv",
                "is_max_stats",
                "is_max_bulk",
                "is_level_max_stats",
                "is_level_max_bulk",
                "is_max_attack",
                "is_max_defense",
                "is_max_stamina",
                "is_efficient_max_level",
                "is_efficient_best_buddy",
            ]
        ]
    )


def app(app="GBL IV Stats", **kwargs):
    (
        ALL_POKEMON_STATS,
        CP_MULTS,
        CP_COEF_PCTS,
        DF_XL_COSTS,
        DF_POKEMON_FAST_MOVES,
        DF_POKEMON_CHARGED_MOVES,
        DF_POKEMON_TYPES,
        DF_POKEMON_TYPE_EFFECTIVENESS,
    ) = load_app_db_constants()

    pokemons = list(ALL_POKEMON_STATS.keys())

    # get eligible pokemon
    with st.sidebar:

        # inputs
        leagues = ["Little", "Great", "Ultra", "Master"]
        default_league = kwargs.get("league", ["Great"])[0]
        default_league_idx = (
            1 if default_league not in leagues else leagues.index(default_league)
        )
        league = st.selectbox("Select a league", leagues, default_league_idx)

        default_pokemon = kwargs.get("pokemon", ["Swampert"])[0]
        default_pokemon_idx = (
            0 if default_pokemon not in pokemons else pokemons.index(default_pokemon)
        )
        pokemon = st.selectbox("Select a Pokemon", pokemons, default_pokemon_idx)
        default_ivs = kwargs.get("input_ivs", [""])[0]
        input_ivs = st.text_input(
            "Input IVs split by a comma (e.g. '1/2/3,15/15/15')", default_ivs
        )
        st.markdown("---")

        # searches to find IVs
        st.markdown("Search options")

        # stats / bulk product rank
        show_ranks_below_cols = st.columns(2)
        with show_ranks_below_cols[0]:
            default_stats_rank = int(kwargs.get("stats_rank", [20])[0])
            stats_rank = st.number_input(
                "Stats Prod Rank <=",
                min_value=0,
                max_value=99999,
                value=default_stats_rank,
            )
        with show_ranks_below_cols[1]:
            default_bulk_rank = int(kwargs.get("bulk_rank", [0])[0])
            bulk_rank = st.number_input(
                "Bulk Prod Rank <=", min_value=0, max_value=99999, value=default_bulk_rank
            )

        # level max stats / hundo ivs
        search_options_row1 = st.columns(3)
        with search_options_row1[0]:
            default_all_ivs = kwargs.get("all_ivs", ["False"])[0] == "True"
            all_ivs = st.checkbox("All IVs", default_all_ivs)
        with search_options_row1[1]:
            default_level_max_stats = (
                kwargs.get("level_max_stats", ["False"])[0] == "True"
            )
            level_max_stats = st.checkbox("Level Maxes", default_level_max_stats)
        with search_options_row1[2]:
            default_show_100 = kwargs.get("show_100", ["False"])[0] == "True"
            show_100 = st.checkbox("15/15/15 IV", default_show_100)

        iv_stat_cols = ["Atk", "Def", "HP"]

        # max stat cols
        search_options_row2 = st.columns(3)
        with search_options_row2[0]:
            default_max_atk = kwargs.get("max_atk", ["False"])[0] == "True"
            max_atk = st.checkbox("Max Atk", default_max_atk)
        with search_options_row2[1]:
            default_max_def = kwargs.get("max_def", ["False"])[0] == "True"
            max_def = st.checkbox("Max Def", default_max_def)
        with search_options_row2[2]:
            default_max_hp = kwargs.get("max_hp", ["False"])[0] == "True"
            max_hp = st.checkbox("Max HP", default_max_hp)

        # 15 iv cols
        search_options_row3 = st.columns(3)
        with search_options_row3[0]:
            default_iv_atk_15 = kwargs.get("iv_atk_15", ["False"])[0] == "True"
            iv_atk_15 = st.checkbox("15 Atk IV", default_iv_atk_15)
        with search_options_row3[1]:
            default_iv_def_15 = kwargs.get("iv_def_15", ["False"])[0] == "True"
            iv_def_15 = st.checkbox("15 Def IV", default_iv_def_15)
        with search_options_row3[2]:
            default_iv_hp_15 = kwargs.get("iv_hp_15", ["False"])[0] == "True"
            iv_hp_15 = st.checkbox("15 HP IV", default_iv_hp_15)

        # 0 iv cols
        search_options_row4 = st.columns(3)
        with search_options_row4[0]:
            default_iv_atk_0 = kwargs.get("iv_atk_0", ["False"])[0] == "True"
            iv_atk_0 = st.checkbox("0 Atk IV", default_iv_atk_0)
        with search_options_row4[1]:
            default_iv_def_0 = kwargs.get("iv_def_0", ["False"])[0] == "True"
            iv_def_0 = st.checkbox("0 Def IV", default_iv_def_0)
        with search_options_row4[2]:
            default_iv_hp_0 = kwargs.get("iv_hp_0", ["False"])[0] == "True"
            iv_hp_0 = st.checkbox("0 HP IV", default_iv_hp_0)

        # exact iv cols
        show_exact_iv_cols = st.columns(3)
        with show_exact_iv_cols[0]:
            default_iv_atk_exact = int(kwargs.get("iv_atk_exact", [-1])[0])
            iv_atk_exact = st.number_input(
                "Atk IV =", min_value=-1, max_value=15, value=default_iv_atk_exact
            )
        with show_exact_iv_cols[1]:
            default_iv_def_exact = int(kwargs.get("iv_def_exact", [-1])[0])
            iv_def_exact = st.number_input(
                "Def IV =", min_value=-1, max_value=15, value=default_iv_def_exact
            )
        with show_exact_iv_cols[2]:
            default_iv_hp_exact = int(kwargs.get("iv_hp_exact", [-1])[0])
            iv_hp_exact = st.number_input(
                "HP IV =", min_value=-1, max_value=15, value=default_iv_hp_exact
            )
        st.markdown("---")

        # ~~~FILTERS~~~
        st.markdown("Filters Options")
        filter_check_boxes = st.columns(3)
        with filter_check_boxes[0]:
            default_filter_inputs = kwargs.get("filter_inputs", ["False"])[0] == "True"
            filter_inputs = st.checkbox("Filter inputs", default_filter_inputs)
        with filter_check_boxes[1]:
            default_efficient_max_level = (
                kwargs.get("efficient_max_level", ["True"])[0] == "True"
            )
            efficient_max_level = st.checkbox(
                f"Efficient@{MAX_LEVEL}", default_efficient_max_level
            )
        with filter_check_boxes[2]:
            default_efficient_best_buddy = (
                kwargs.get("efficient_best_buddy", ["False"])[0] == "True"
            )
            efficient_best_buddy = st.checkbox(
                f"Efficient@{MAX_LEVEL+1}", default_efficient_best_buddy
            )

        # min iv cols
        min_ivs_cols = st.columns(3)
        with min_ivs_cols[0]:
            default_iv_atk_ge = int(kwargs.get("iv_atk_ge", [0])[0])
            iv_atk_ge = st.number_input(
                "Atk IV >=", min_value=0, max_value=15, value=default_iv_atk_ge
            )
        with min_ivs_cols[1]:
            default_iv_def_ge = int(kwargs.get("iv_def_ge", [0])[0])
            iv_def_ge = st.number_input(
                "Def IV >=", min_value=0, max_value=15, value=default_iv_def_ge
            )
        with min_ivs_cols[2]:
            default_iv_hp_ge = int(kwargs.get("iv_hp_ge", [0])[0])
            iv_hp_ge = st.number_input(
                "HP IV >=", min_value=0, max_value=15, value=default_iv_hp_ge
            )

        # min stats cols
        min_stats_cols = st.columns(3)
        with min_stats_cols[0]:
            default_stats_atk_ge = float(kwargs.get("stats_atk_ge", [0.0])[0])
            stats_atk_ge = st.number_input(
                "Atk Stat >=", min_value=0.0, max_value=9001.0, value=default_stats_atk_ge
            )
        with min_stats_cols[1]:
            default_stats_def_ge = float(kwargs.get("stats_def_ge", [0.0])[0])
            stats_def_ge = st.number_input(
                "Def Stat >=", min_value=0.0, max_value=9001.0, value=default_stats_def_ge
            )
        with min_stats_cols[2]:
            default_stats_hp_ge = int(kwargs.get("stats_hp_ge", [0])[0])
            stats_hp_ge = st.number_input(
                "HP Stat >=", min_value=0, max_value=9001, value=default_stats_hp_ge
            )

        filter_level_cp = st.columns(3)
        with filter_level_cp[0]:
            default_cp_ge = int(kwargs.get("min_cp", [0])[0])
            min_cp = st.number_input(
                "CP >=", min_value=0, max_value=9999, value=default_cp_ge
            )
        with filter_level_cp[1]:
            default_level_ge = int(kwargs.get("level_ge", [0])[0])
            level_ge = st.number_input(
                "Level >=", min_value=0, max_value=50, value=default_level_ge
            )
        with filter_level_cp[2]:
            default_level_le = int(kwargs.get("level_le", [51])[0])
            level_le = st.number_input(
                "Level <=", min_value=0, max_value=51, value=default_level_le
            )
        st.markdown("---")

        # ~~~ COLUMN OPTIONS ~~~
        st.markdown("Column Options")
        default_show_r1_cmp = kwargs.get("show_cmp", ["False"])[0] == "True"
        show_cmp = st.checkbox("Show CMP vs Rank 1 column", default_show_r1_cmp)
        default_show_individual_ivs = (
            kwargs.get("show_individual_ivs", ["False"])[0] == "True"
        )
        default_show_prod_cols = kwargs.get("show_prod_cols", ["False"])[0] == "True"
        show_prod_cols = st.checkbox("Show stats/bulk columns", default_show_prod_cols)
        show_individual_ivs = st.checkbox(
            "Show individual IV columns", default_show_individual_ivs
        )
        st.markdown("Show XL costs")

        # xl cost cols
        xl_cost_cols = st.columns(4)
        with xl_cost_cols[0]:
            default_show_xl_regular = (
                kwargs.get("show_xl_regular", ["False"])[0] == "True"
            )
            show_xl_regular = st.checkbox("Regular", default_show_xl_regular)
        with xl_cost_cols[1]:
            default_show_xl_lucky = kwargs.get("show_xl_lucky", ["False"])[0] == "True"
            show_xl_lucky = st.checkbox("Lucky", default_show_xl_lucky)
        with xl_cost_cols[2]:
            default_show_xl_shadow = kwargs.get("show_xl_shadow", ["False"])[0] == "True"
            show_xl_shadow = st.checkbox("Shadow", default_show_xl_shadow)
        with xl_cost_cols[3]:
            default_show_xl_purified = (
                kwargs.get("show_xl_purified", ["False"])[0] == "True"
            )
            show_xl_purified = st.checkbox("Purified", default_show_xl_purified)
        st.markdown("---")

    # main app
    st.subheader(pokemon)
    ivs = parse_ivs(input_ivs)

    # get df with all rows
    df = get_league_pokemon_df(
        league, pokemon, input_ivs, ALL_POKEMON_STATS, CP_MULTS, CP_COEF_PCTS, DF_XL_COSTS
    )

    # filter df based on options selected
    mask_t = pd.Series(True, df.index)
    mask_f = ~mask_t

    # mask searched
    mask_searched = (
        (df["Rank"] <= stats_rank)
        | (df["Rank Bulk"] <= bulk_rank)
        | (mask_t if all_ivs else mask_f)
        | (df["is_level_max_stats"] if level_max_stats else mask_f)
        | (df["is_max_iv"] if show_100 else mask_f)
        | (df["is_max_attack"] if max_atk else mask_f)
        | (df["is_max_defense"] if max_def else mask_f)
        | (df["is_max_stamina"] if max_hp else mask_f)
        | ((df["IV Atk"] == 15) if iv_atk_15 else mask_f)
        | ((df["IV Def"] == 15) if iv_def_15 else mask_f)
        | ((df["IV HP"] == 15) if iv_hp_15 else mask_f)
        | ((df["IV Atk"] == 0) if iv_atk_0 else mask_f)
        | ((df["IV Def"] == 0) if iv_def_0 else mask_f)
        | ((df["IV HP"] == 0) if iv_hp_0 else mask_f)
        | ((df["IV Atk"] == iv_atk_exact) if iv_atk_exact >= 0 else mask_f)
        | ((df["IV Def"] == iv_def_exact) if iv_def_exact >= 0 else mask_f)
        | ((df["IV HP"] == iv_hp_exact) if iv_hp_exact >= 0 else mask_f)
    )

    # mask filter
    mask_filter = (
        (df["is_efficient_max_level"] if efficient_max_level else mask_t)
        & (df["is_efficient_best_buddy"] if efficient_best_buddy else mask_t)
        & (df["IV Atk"] >= iv_atk_ge)
        & (df["IV Def"] >= iv_def_ge)
        & (df["IV HP"] >= iv_hp_ge)
        & (df["Atk"] >= stats_atk_ge)
        & (df["Def"] >= stats_def_ge)
        & (df["HP"] >= stats_hp_ge)
        & (df["Level"] >= level_ge)
        & (df["Level"] <= level_le)
        & (df["CP"] >= min_cp)
    )

    # mask inputs
    mask_inputs = (df["Input"] == "True") & (mask_filter if filter_inputs else mask_t)
    mask_inputs_with_filter = mask_inputs & (mask_filter if filter_inputs else mask_t)

    # combining both and filtering
    mask = (mask_searched & mask_filter) | mask_inputs_with_filter
    df = df[mask].reset_index(drop=True)

    # add after filters efficient col
    stat_cols = ["Atk", "Def", "HP"]
    df["is_efficient_filtered"] = get_pareto_efficient_stats(df[stat_cols].values)
    display_true = {True: "True", False: ""}
    df[f"Efficient @Filters"] = df["is_efficient_filtered"].map(display_true)

    # set order of columns
    df_col_order = (
        "Rank,Level,CP,IVs,IV Atk,IV Def,IV HP,R1 CMP,Pct Max Stats,Stats Prod,Bulk Prod,"
        "Rank Bulk,Atk,Def,HP,Efficient @50,Efficient @51,Efficient @Filters,Notes,Input,"
        "Regular XLs,Lucky XLs,Shadow XLs,Purified XLs"
    ).split(",")
    df = df[df_col_order]

    # caption for results
    st.caption(
        "Results from "
        f"{mask_inputs.sum()} inputs, "
        f"{mask_searched.sum()} matching search criteria, "
        f"{(mask_inputs | mask_searched).sum() - len(df)} removed by filters, "
        f"leaving {len(df)} IVs remaining."
    )

    # setup default preselected_ivs
    input_ivs_rows = ",".join([str(i) for i in df[df["Input"] == "True"].index])
    default_preselected_ivs = kwargs.get("preselected_ivs", [input_ivs_rows])[0]
    default_preselected_ivs_ints = [
        int(i) for i in default_preselected_ivs.split(",") if i
    ]

    # build ivs output
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection("multiple", pre_selected_rows=default_preselected_ivs_ints)
    gb.configure_column("Rank", width=70)
    gb.configure_column(
        "Level",
        type=["numericColumn", "numberColumnFilter", "customNumericFormat"],
        precision=1,
        width=70,
    )
    gb.configure_column("CP", width=60)
    gb.configure_column("IVs", width=70, hide=show_individual_ivs)
    gb.configure_column("IV Atk", width=70, hide=not show_individual_ivs)
    gb.configure_column("IV Def", width=70, hide=not show_individual_ivs)
    gb.configure_column("IV HP", width=70, hide=not show_individual_ivs)
    gb.configure_column("R1 CMP", width=80, hide=not show_cmp)
    gb.configure_column(
        "Pct Max Stats",
        type=["numericColumn", "numberColumnFilter", "customNumericFormat"],
        precision=1,
        width=110,
    )
    gb.configure_column(
        "Stats Prod",
        type=["numericColumn", "numberColumnFilter", "customNumericFormat"],
        precision=1,
        hide=not show_prod_cols,
        width=95,
    )
    gb.configure_column(
        "Bulk Prod",
        type=["numericColumn", "numberColumnFilter", "customNumericFormat"],
        precision=1,
        hide=not show_prod_cols,
        width=95,
    )
    gb.configure_column("Rank Bulk", width=105, hide=not show_prod_cols)
    gb.configure_column(
        "Atk",
        type=["numericColumn", "numberColumnFilter", "customNumericFormat"],
        precision=2,
        width=60,
    )
    gb.configure_column(
        "Def",
        type=["numericColumn", "numberColumnFilter", "customNumericFormat"],
        precision=2,
        width=60,
    )
    gb.configure_column(
        "HP",
        type=["numericColumn", "numberColumnFilter", "customNumericFormat"],
        precision=0,
        width=60,
    )
    gb.configure_column("Efficient @50", width=110)
    gb.configure_column("Efficient @51", width=110)
    gb.configure_column("Efficient @Filters", width=130)
    gb.configure_column("Notes", width=220)
    gb.configure_column("Input", width=70, hide=not bool(ivs))
    gb.configure_column("Regular XLs", width=90, hide=not show_xl_regular)
    gb.configure_column("Lucky XLs", width=90, hide=not show_xl_lucky)
    gb.configure_column("Shadow XLs", width=90, hide=not show_xl_shadow)
    gb.configure_column("Purified XLs", width=90, hide=not show_xl_purified)

    grid_options = gb.build()
    ivs_response = AgGrid(
        df,
        gridOptions=grid_options,
        # columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        height=600,
        custom_css={
            ".ag-theme-streamlit-dark": {
                "--ag-grid-size": "3px",
            }
        },
    )
    selected_ivs = ivs_response["selected_rows"]
    preselected_ivs = ",".join(
        [r["_selectedRowNodeInfo"]["nodeId"] for r in selected_ivs]
    )

    # fast moves
    st.caption(
        "Fast moves (damage includes STAB bonus) - Click a move to get count and turn details"
    )
    df_fast = get_poke_fast_moves(pokemon, DF_POKEMON_FAST_MOVES)
    gb_fast = GridOptionsBuilder.from_dataframe(df_fast)

    default_preselected_fast = kwargs.get("preselected_fast", [""])[0]
    default_preselected_fast_ints = [
        int(i) for i in default_preselected_fast.split(",") if i
    ]
    gb_fast.configure_selection(
        "single", use_checkbox=False, pre_selected_rows=default_preselected_fast_ints
    )

    go_fast = gb_fast.build()
    fast_moves_response = AgGrid(
        df_fast,
        gridOptions=go_fast,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        custom_css={
            ".ag-theme-streamlit-dark": {
                "--ag-grid-size": "3px",
            }
        },
    )
    fast_move = fast_moves_response["selected_rows"]
    preselected_fast = ",".join([r["_selectedRowNodeInfo"]["nodeId"] for r in fast_move])

    # chargedmoves
    st.caption("Charged moves (damage includes STAB bonus)")
    df_charged = get_poke_charged_moves(pokemon, DF_POKEMON_CHARGED_MOVES)
    if len(fast_move):
        df_charged["Count"] = (df_charged["Energy"] / fast_move[0]["Energy Gain"]).apply(
            math.ceil
        )
        df_charged["Turns"] = df_charged["Count"] * fast_move[0]["Turns"]
    else:
        df_charged["Count"] = ""
        df_charged["Turns"] = ""
    gb_charged = GridOptionsBuilder.from_dataframe(df_charged)
    default_preselected_charged = kwargs.get("preselected_charged", [""])[0]
    default_preselected_charged_ints = [
        int(i) for i in default_preselected_charged.split(",") if i
    ]
    gb_charged.configure_selection(
        "multiple", use_checkbox=False, pre_selected_rows=default_preselected_charged_ints
    )
    go_charged = gb_charged.build()
    charged_moves_response = AgGrid(
        df_charged,
        gridOptions=go_charged,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        custom_css={
            ".ag-theme-streamlit-dark": {
                "--ag-grid-size": "3px",
            }
        },
    )
    charged_moves = charged_moves_response["selected_rows"]
    preselected_charged = ",".join(
        [r["_selectedRowNodeInfo"]["nodeId"] for r in charged_moves]
    )

    # format text to import into pvpoke
    st.caption(
        """
        Pvpoke matrix import text - Click to select IVs, a fast move, and charged moves 
        to geterate an import string for Pvpoke matrix.Note that pvpoke only allows 
        100 inputs at a time, so line numbers (not IV ranks) are shown on the left to 
        help with copy/pasting chunks at a time.
        """
    )

    # format pokemon name
    pvpoke_text = {}
    pvpoke_text["pokemon"] = (
        pokemon.lower()
        .replace("'", "")
        .replace(" ", "_")
        .replace("(shadow)", "shadow-shadow")
        .replace("(", "")
        .replace(")", "")
    )
    # format moves
    selected_moves = [m["Move"] for m in fast_move[:1]] + [
        m["Move"] for m in charged_moves[:2]
    ]
    pvpoke_text["moves"] = ",".join(
        [
            m.upper().replace(" ", "_").replace("(", "").replace(")", "")
            for m in selected_moves
        ]
    )

    if len(selected_ivs) and len(fast_move) and len(charged_moves):
        pvpoke_text_lines = []
        for row in selected_ivs:
            pvpoke_text.update(row)
            txt = "{pokemon},{moves},{Level},{IV Atk},{IV Def},{IV HP}".format(
                **pvpoke_text
            )
            pvpoke_text_lines.append(txt)
        pvpoke_lines_cols = st.columns([1, 30])
        with pvpoke_lines_cols[0]:
            line_numbers = "\n".join([str(i + 1) for i in range(len(pvpoke_text_lines))])
            st.code(line_numbers, language=None)
        with pvpoke_lines_cols[1]:
            st.code("\n".join(pvpoke_text_lines), language=None)

    # create sharable url
    with st.sidebar:
        params_list = [
            "app",
            "league",
            "pokemon",
            "input_ivs",
            "stats_rank",
            "bulk_rank",
            "all_ivs",
            "level_max_stats",
            "show_100",
            "max_atk",
            "max_def",
            "max_hp",
            "iv_atk_15",
            "iv_def_15",
            "iv_hp_15",
            "iv_atk_0",
            "iv_def_0",
            "iv_hp_0",
            "iv_atk_exact",
            "iv_def_exact",
            "iv_hp_exact",
            "filter_inputs",
            "efficient_max_level",
            "efficient_best_buddy",
            "iv_atk_ge",
            "iv_def_ge",
            "iv_hp_ge",
            "stats_atk_ge",
            "stats_def_ge",
            "stats_hp_ge",
            "min_cp",
            "level_ge",
            "level_le",
            "show_cmp",
            "show_individual_ivs",
            "show_prod_cols",
            "show_xl_regular",
            "show_xl_lucky",
            "show_xl_shadow",
            "show_xl_purified",
            "preselected_ivs",
            "preselected_fast",
            "preselected_charged",
        ]
        url = get_query_params_url(params_list, {**kwargs, **locals()})
        st.markdown(f"[Share this app's output]({url})")
        st.markdown("---")

        # help strings
        with st.expander("Help"):

            # about the app
            st.subheader("About this app.")
            st.markdown(
                "Hello there! I built this app to make it easier to search, filter, "
                "compare, and sim IVs."
            )

            # searching and filtering
            st.subheader("Searching and Filtering")
            st.markdown(
                "This app outputs IVs to a table in two steps:\n\n"
                "1. Find all IVs that match inputs or search options\n"
                "2. Exclude any IVs that don't match the filter options"
            )

            # efficient ivs example
            example_efficient_url = get_query_params_url(
                "league,pokemon,stats_rank,efficient_max_level,efficient_best_buddy".split(
                    ","
                ),
                dict(
                    league="Ultra",
                    pokemon="Talonflame",
                    stats_rank=25,
                    efficient_max_level=False,
                    efficient_best_buddy=False,
                ),
            )
            st.subheader("Efficient IVs")
            st.markdown(
                "Efficient IVs are IVs that result in stats for a pokemon plus league "
                "where no other IVs dominate in all 3 of attack, defense, and hp stats. "
                "In masters league, 15/15/15 IVs are usually the only efficient IVs "
                "(except in cases of the hp stat rounding which makes 15/15/14 equal) "
                "since no other set of IVs are strictly superior. In little/great/ultra "
                "leagues its harder to figure out which IVs are efficient, so filters "
                "and columns are included here.\n"
                f"* `Efficient @{MAX_LEVEL}` means IVs that are efficient when "
                "excluding IVs that max out with best buddy levels\n"
                f"* `Efficient @{MAX_LEVEL+1}` means IVs that are efficient when "
                "including IVs that max out with best buddy levels. "
                f"If a pokemon doesn't have any IVs that max out above {MAX_LEVEL}, "
                f"then both @{MAX_LEVEL} and @{MAX_LEVEL+1} will be the same. "
                f"[Ultra league Talonflame]({example_efficient_url}) provides a good "
                "example of the difference between the two.\n"
                "* The `Efficient @Filters` column indicates IVs that are efficient from "
                "of all the IVs remaining in the output table (so after all searches + "
                "filters have been applied). This can be useful for tasks like comparing "
                "a ton of input IVs or pokemon with IV floors (e.g. 10+ for raids)."
            )

        st.markdown("---")


if __name__ == "__main__":
    query_params = st.experimental_get_query_params()
    app(**query_params)
