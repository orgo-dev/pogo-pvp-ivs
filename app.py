import streamlit as st

st.set_page_config(
    page_title="Pogo PVP IVs",
    page_icon="https://www.google.com/s2/favicons?domain=pokemongolive.com",
    layout="wide",
    initial_sidebar_state="expanded",
)

from pvp_ivs_app import app as gbl_iv_stats
from pvp_ivs_app_new import app as gbl_iv_stats_new
from move_counts_app import app as move_counts
from types_and_moves_app import app as types_and_moves
import updated_timestamp  # hack to make sure streamlit reruns when data is updated
from streamlit.runtime.state.session_state_proxy import get_session_state
# class for multipage drop down
class MultiPage:
    def __init__(self) -> None:
        self.pages = {}

    def add_page(self, title, func) -> None:
        self.pages[title] = func

    def run(self, **kwargs):
        # dropdown to selec the app to run
        pages = list(self.pages.keys())
        default_page = kwargs.get("app")
        default_page_index = pages.index(default_page[0]) if default_page else 0
        app = st.sidebar.selectbox("Apps", self.pages.keys(), default_page_index)
        kwargs["app"] = app
        st.sidebar.markdown("---")

        # run the app function, passing url parameters
        self.pages[app](**kwargs)


# create app and sidebar with list of apps
app = MultiPage()
app.add_page("GBL IV Stats", gbl_iv_stats)
app.add_page("GBL IV Stats - new", gbl_iv_stats_new)
app.add_page("Move Counts", move_counts)
app.add_page("Pokemon Types and Moves", types_and_moves)

qp = {k:st.query_params.get_all(k) for k in st.query_params}
app.run(**qp)
