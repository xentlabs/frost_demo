# Xent Games Frost Demo

This repo contains a Jupyter notebook that allows you to train a model on a Xent Game using the efficient [Frost algorithm](https://arxiv.org/abs/2605.27701v1).

## Run On Colab

Simply open the [Colab notebook](https://colab.research.google.com/drive/1aL8m1fLpYHidrZZ6GwYjMwfOOfgw20ot) and follow the instructions there.

## Run Locally

Install the dependencies into a venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name xent-games-frost --display-name "Xent Games Frost"
```

For improved performance on Linux, you can additionally install `causal-conv1d`.

Open `frost_demo.ipynb` in Jupyter, VS Code, or another notebook UI and select the `Xent Games Frost` kernel. If you don't see the kernel, you might have to restart the program first.

To launch it with JupyterLab:

```bash
python -m jupyter lab frost_demo.ipynb
```