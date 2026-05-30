def _user_namespace():
    try:
        from IPython import get_ipython
    except ImportError:
        return globals()
    shell = get_ipython()
    return shell.user_ns if shell is not None else globals()


def _dropdown(variable_name, description, options):
    try:
        import ipywidgets as ipw
        from IPython.display import display
    except ImportError as exc:
        raise ImportError(
            "ipywidgets is required for the demo dropdowns. Install it with "
            "`pip install ipywidgets`."
        ) from exc

    widget = ipw.Dropdown(
        options=options,
        description=description,
        style={"description_width": "initial"},
        layout=ipw.Layout(width="max-content"),
    )
    namespace = _user_namespace()
    namespace[variable_name] = widget.value

    def update_value(change):
        namespace[variable_name] = change["new"]

    widget.observe(update_value, names="value")
    display(widget)


def dataset_dropdown():
    _dropdown(
        "dataset",
        "Dataset",
        [
            ("Cosmopedia", "cosmopedia"),
            ("FineWeb-Edu", "fineweb-edu"),
        ],
    )


def model_dropdown():
    _dropdown(
        "model",
        "Model",
        [
            ("Qwen3.5-2B (runs on 15 GB GPU, e.g., Colab)", "Qwen/Qwen3.5-2B"),
            ("Qwen3.5-4B", "Qwen/Qwen3.5-4B"),
            ("Qwen3.5-9B", "Qwen/Qwen3.5-9B"),
        ],
    )


def steps_dropdown():
    _dropdown(
        "steps",
        "Training steps",
        [(str(value), value) for value in [20, 40, 60, 80, 100]],
    )


def micro_batch_size_dropdown():
    _dropdown(
        "micro_batch_size",
        "Micro batch size",
        [(str(value), value) for value in [1, 2, 4]],
    )
