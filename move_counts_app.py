import streamlit as st
from st_keyup import st_keyup
import math, re, sqlite3, pandas as pd, numpy as np
from config import PATH_DATA
from app_utils import get_query_params_url, load_app_db_constants


@st.cache_data(ttl=3600)
def get_pokemons(DF_POKEMON_FAST_MOVES):
    # pokemon with valid fast moves
    valid_pokemon = DF_POKEMON_FAST_MOVES[
        DF_POKEMON_FAST_MOVES["Energy Gain"] != 0
    ].pokemon.unique()
    pokemons = [
        p
        for p in pd.read_csv(f"{PATH_DATA}/pokemon.csv").pokemon.to_list()
        if "(Shadow" not in p
        and "(Mega" not in p
        and p in valid_pokemon
        and p != "Smeargle"
    ]
    return sorted(pokemons)


def create_move_count_print_str(pokemon, df):
    fm_len = max(df.fast_move.str.len().max(), 6) + 1
    cm_len = df.charged_move.str.len().max()
    print_rows = [" " * (cm_len) + " |"]
    for i, (fm, df_fm) in enumerate(df.groupby("fast_move")):
        print_rows[0] += f"{fm:>{fm_len}} |"
        for j, row in df_fm.reset_index(drop=True).iterrows():
            if i == 0:
                print_rows.append(f"{row.charged_move:>{cm_len}} |")
            counts = f"{row.count_1:>2} {row.count_2:>2} {row.count_3:>2}"
            print_rows[j + 1] += f"{counts:>{fm_len}} |"
    return "\n".join([pokemon] + print_rows)


@st.cache_data(ttl=3600)
def get_all_pokemon_move_counts(
    all_pokemons,
    DF_POKEMON_FAST_MOVES,
    DF_POKEMON_CHARGED_MOVES,
):
    df_fast = (
        DF_POKEMON_FAST_MOVES[
            (DF_POKEMON_FAST_MOVES["pokemon"].isin(all_pokemons))
            & (DF_POKEMON_FAST_MOVES["Energy Gain"] != 0)
            & (~DF_POKEMON_FAST_MOVES["Move"].str.contains("Hidden Power"))
        ]
        .rename(
            {
                "Move": "fast_move",
                "Type": "fast_type",
                "Damage": "fast_damage",
                "Archetype": "fast_archetype",
                "Notes": "fast_notes",
                "Turns": "fast_turns",
            },
            axis=1,
        )
        .drop(["move_kind"], axis=1)
        .reset_index(drop=True)
    )
    df_charged = (
        DF_POKEMON_CHARGED_MOVES[DF_POKEMON_CHARGED_MOVES["pokemon"].isin(all_pokemons)]
        .rename(
            {
                "Move": "charged_move",
                "Type": "charged_type",
                "Damage": "charged_damage",
                "Archetype": "charged_archetype",
                "Notes": "charged_notes",
            },
            axis=1,
        )
        .drop(["move_kind"], axis=1)
        .reset_index(drop=True)
    )
    df_combined_moves = df_fast.merge(df_charged, how="left", on="pokemon")

    # get counts
    df_combined_moves["count_1"] = (
        df_combined_moves["Energy"] / df_combined_moves["Energy Gain"]
    ).apply(math.ceil)
    df_combined_moves["count_2"] = (
        2 * df_combined_moves["Energy"] / df_combined_moves["Energy Gain"]
    ).apply(math.ceil) - df_combined_moves["count_1"]
    df_combined_moves["count_3"] = (
        (3 * df_combined_moves["Energy"] / df_combined_moves["Energy Gain"]).apply(
            math.ceil
        )
        - df_combined_moves["count_1"]
        - df_combined_moves["count_2"]
    )
    df_combined_moves["charged_turns_1"] = (
        df_combined_moves["count_1"] * df_combined_moves["fast_turns"]
    )

    pokemon_moves_str_dict = {}
    for p, foo in df_combined_moves.groupby("pokemon"):
        pokemon_moves_str_dict[p] = create_move_count_print_str(p, foo)
    return pokemon_moves_str_dict


def app(**kwargs):
    print("RUNNING move_counts_app app")
    (
        ALL_POKEMON_STATS,
        POKE_DEX_IDS,
        POKE_PARENTS,
        POKE_CHILDREN,
        CP_MULTS,
        CP_COEF_PCTS,
        DF_XL_COSTS,
        DF_POKEMON_FAST_MOVES,
        DF_POKEMON_CHARGED_MOVES,
        DF_POKEMON_TYPES,
        DF_POKEMON_TYPE_EFFECTIVENESS,
    ) = load_app_db_constants()

    # load things
    all_pokemons = get_pokemons(DF_POKEMON_FAST_MOVES)
    all_pokemons_move_counts = get_all_pokemon_move_counts(
        all_pokemons,
        DF_POKEMON_FAST_MOVES,
        DF_POKEMON_CHARGED_MOVES,
    )

    # sidebar for inputs
    with st.sidebar:
        default_display_limit = 10
        default_limit_outputs = kwargs.get("limit_outputs", ["True"])[0] == "True"
        limit_outputs = st.checkbox(
            f"Limit outputs to first {default_display_limit}, making app respond a little faster.",
            default_limit_outputs,
        )
        display_limit = default_display_limit if limit_outputs else None

    # real time filtering input
    st.markdown("Enter text to filter pokemon.")
    search_text = st_keyup(label="", value="swamp")
    search_text_letters = "".join(filter(str.isalpha, search_text))
    search_pattern = re.compile(".*".join(search_text_letters.lower()))
    selected_pokemon = [p for p in all_pokemons if search_pattern.match(p.lower())]
    st.markdown("---")

    # display move counts for matching pokemon

    for p in selected_pokemon[:display_limit]:
        pokemon_move_count_text = all_pokemons_move_counts.get(p)
        if pokemon_move_count_text:
            st.text(pokemon_move_count_text)
            st.markdown("---")

    # create sharable url
    with st.sidebar:
        # help strings
        with st.expander("About this app"):

            # about the app
            st.markdown(
                "Hello there! I built this app to make searching pokemon move counts "
                "super fast and super simple."
            )

            # searching and filtering
            st.subheader("Searching")
            st.markdown(
                "Simply just type text and pokemon with names matching the text "
                "will be shown. The names will match as long as the characters "
                "are in the right order. For example, if you type in `swmp`, you will "
                "see results for `Swampert` since the letters appear in the same order "
                "as they do in the name."
            )

            # searching and filtering
            st.subheader("Outputs")
            st.markdown(
                "The outputs show fast moves along the top as columns and charged moves "
                "along the bottom as rows. The numbers in each cell represent how many "
                "fast moves are required for the first, second, and third charged move."
            )

        # sharing app
        params_list = ["app"]
        url = get_query_params_url(params_list, {**kwargs, **locals()})
        st.markdown(f"[Share this app]({url})")
        st.markdown("---")


if __name__ == "__main__":
    app(**query_params)
