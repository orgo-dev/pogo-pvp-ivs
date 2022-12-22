from streamlit_javascript import st_javascript


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
