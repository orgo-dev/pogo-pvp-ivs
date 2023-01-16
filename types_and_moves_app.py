import math
import sqlite3
import pandas as pd
import streamlit as st
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
    ALL_POKEMON_STATS,
    DF_POKEMON_TYPES,
    DF_POKEMON_TYPE_EFFECTIVENESS,
)


def get_poke_types_dfs(pokemon):
    types_poke = DF_POKEMON_TYPES.loc[pokemon].reset_index(drop=True)
    types_poke = types_poke[types_poke["Type"] != "none"]
    df = DF_POKEMON_TYPE_EFFECTIVENESS.loc[pokemon]
    types_resist = df[df["Effectiveness"] < 1].reset_index(drop=True)
    types_weak = df[df["Effectiveness"] > 1].reset_index(drop=True)
    return types_poke, types_resist, types_weak


def app(**kwargs):

    pokemons = list(ALL_POKEMON_STATS.keys())

    # get eligible pokemon
    default_pokemon = kwargs.get("pokemon", ["Swampert"])[0]
    default_pokemon_idx = (
        0 if default_pokemon not in pokemons else pokemons.index(default_pokemon)
    )
    pokemon = st.selectbox("Select a Pokemon", pokemons, default_pokemon_idx)

    # types display
    image_type = JsCode(
        """function (params) {
            var element = document.createElement("span");
            var imageElement = document.createElement("img");
            imageElement.src = "https://storage.googleapis.com/nianticweb-media/pokemongo/types/" + params.value + ".png";
            imageElement.width="20";
            element.appendChild(imageElement);
            element.appendChild(document.createTextNode(" " + params.value.charAt(0).toUpperCase() + params.value.slice(1)));
            return element;
    }"""
    )

    types_poke, types_resist, types_weak = get_poke_types_dfs(pokemon)
    types_columns = st.columns(3)
    types_css = {
        ".ag-theme-streamlit-dark": {
            "--ag-borders": "none",
            "--ag-border-color": "none",
            "--ag-header-column-resize-handle-display": "none",
            "--ag-header-background-color": "none",
            "--ag-header-foreground-color": "white",
            "--ag-grid-size": "4px",
        },
    }

    # selected pokemon types
    with types_columns[0]:
        st.caption(f"Pokemon Types")
        gb_types_poke = GridOptionsBuilder.from_dataframe(types_poke)
        gb_types_poke.configure_column("Type", cellRenderer=image_type)
        go_types_poke = gb_types_poke.build()
        AgGrid(
            types_poke,
            gridOptions=go_types_poke,
            allow_unsafe_jscode=True,
            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
            custom_css=types_css,
        )

    # selected pokemon resisted types
    with types_columns[1]:
        st.caption(f"Resisted Types")
        gb_types_resist = GridOptionsBuilder.from_dataframe(types_resist)
        gb_types_resist.configure_column("Type", cellRenderer=image_type)
        gb_types_resist.configure_column(
            "Effectiveness", type=["numericColumn", "customNumericFormat"], precision=3
        )
        gb_types_resist.configure_grid_options(autoSizePadding=2)
        go_types_resist = gb_types_resist.build()
        AgGrid(
            types_resist,
            gridOptions=go_types_resist,
            allow_unsafe_jscode=True,
            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
            custom_css=types_css,
        )

    # selected pokemon weak types
    with types_columns[2]:
        st.caption(f"Weak Types")
        gb_types_weak = GridOptionsBuilder.from_dataframe(types_weak)
        gb_types_weak.configure_column("Type", cellRenderer=image_type)
        gb_types_weak.configure_column(
            "Effectiveness", type=["numericColumn", "customNumericFormat"], precision=2
        )
        go_types_weak = gb_types_weak.build()
        AgGrid(
            types_weak,
            gridOptions=go_types_weak,
            allow_unsafe_jscode=True,
            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
            custom_css=types_css,
        )

    # fast moves
    st.caption(
        "Fast moves (damage includes STAB bonus) - Click a move to get count and turn details"
    )
    df_fast = get_poke_fast_moves(pokemon)
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

    # charged moves
    st.caption("Charged moves (damage includes STAB bonus)")
    df_charged = get_poke_charged_moves(pokemon)
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

    # create sharable url
    with st.sidebar:
        params_list = ["app", "pokemon", "preselected_fast", "preselected_charged"]
        url = get_query_params_url(params_list, {**kwargs, **locals()})
        st.markdown(f"[Share this app's output]({url})")
        st.markdown("---")


if __name__ == "__main__":
    app(**query_params)
