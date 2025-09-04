# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair==5.5.0",
#     "marimo",
#     "matplotlib==3.10.6",
#     "numpy==2.3.2",
#     "pandas==2.3.2",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np 
    import pandas as pd 
    import textwrap 
    import matplotlib.pyplot as plt 
    import seaborn as sns 
    import altair as alt

    plt.style.use('seaborn-v0_8-darkgrid') 

    SAFE_NS = {
        # constants
        "pi": np.pi,
        "e": np.e,
        # basic ops
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "arcsin": np.arcsin,
        "arccos": np.arccos,
        "arctan": np.arctan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "sign": np.sign,
        "heaviside": np.heaviside,
        "where": np.where,
        # handy
        "linspace": np.linspace,
        "clip": np.clip,
    }
    return alt, mo, np, pd


@app.cell
def _(mo):
    mo.md(
        r"""
    # Average Velocity Explorer 
    Interactively approximate velocity from a position function $x(t)$ by taking average slopes over equal-time segments.


    **How it works**
    - Choose or type a function for $x(t)$.

    - Pick a time window and sample density for plotting.

    - Set N, the number of segment boundary points (including start and end). This creates N−1 equal-time segments.

    - Note, $N\geq 2$ in order to work properly.

    - For each segment $[t_i, t_{i+1}]$, the app computes the average velocity: $$\bar v_i = \frac{x(t_{i+1}) - x(t_i)}{t_{i+1}-t_i}.$$


    Notes:

    - `N = 2` → one segment using only the initial and final points (overall average velocity).

    - `N = 3` → two segments (splits the interval in half).
    - Larger `N` gives a piecewise-constant estimate that approaches the instantaneous derivative.
    """
    )
    return


@app.cell
def _(mo, np):
    # Define the UI settings like number of points to plot, range of t, etc.
    preset = mo.ui.dropdown(
        {
            "t": lambda t: t,
            "t**2": lambda t: pow(t,2), 
            "sin(2*pi*t)": lambda t: np.sin(2*np.pi*t),
            "exp(-t)*sin(2*pi*t)": lambda t: np.exp(-t)*np.sin(2*np.pi*t), 
            "t**3 - 3*t": lambda t: pow(t,3) - 3*t,
        }, 
        value="sin(2*pi*t)", 
        label="Presets", 
    )
    custom_expr = mo.ui.text(value="", placeholder="Or type any Python/Numpy expr in t, e.g. sin(t)+0.1*t**2")


    t_min = mo.ui.number(-1.0, label="t_min")
    t_max = mo.ui.number(1.0, label="t_max")
    M = mo.ui.slider(start=20, stop=2000, step=10, value=100, include_input=True, label="plot samples (M)")
    N = mo.ui.slider(start=2, stop=M.stop, step=1, value=2, include_input=True, label="N (segment boundary points)")

    # Optional checkboxing that does not have an purpose right now.
    show_secants = mo.ui.checkbox(True, label="Draw secant lines on x(t) plot")
    show_gradient = mo.ui.checkbox(True, label="Show numerical gradient on v(t) plot")

    # Settings for how the objects are displayed. This creates a single row comprised of objects in three columns.
    mo.hstack([
        mo.vstack([preset, custom_expr], gap="0.5rem"),
        mo.vstack([t_min, t_max, M], gap="0.5rem"),
        mo.vstack([N, show_secants, show_gradient], gap="0.5rem"),
    ])
    return M, N, preset, t_max, t_min


@app.cell
def _(M, N, np, pd, preset, t_max, t_min):
    # Create function arrays and associated derivative calculations. 
    t = np.linspace(t_min.value, t_max.value, M.value)
    x = preset.value(t)

    # Index t using N-points. Take the floor value if not cleanly divisible.
    ind_step = int(M.value/N.value)
    indices = np.arange(ind_step, M.value, ind_step)

    # The initial and final point also should be included for averaging later.
    indices = np.concat([np.array([0]), indices])
    if indices[-1] != len(t):
        indices = np.concat([indices, np.array([int(len(t)-1)])])

    # Calculate average distance for each of the subpoint intervals. 
    dt = t[ind_step] - t[0]
    dxdt = []
    for i in range(len(indices)-1):
        this_dxdt = (x[i+1] - x[i])/dt
        dxdt.append(this_dxdt)
    dxdt=np.array(dxdt)

    # Now combine date into separate dataframes for future use with Altair plotting.  
    df_pos = pd.DataFrame({"t":t, "x": x})
    df_dpos = pd.DataFrame({"t":t[indices[:len(indices)-1]], "x":x[indices[:len(indices)-1]], "dxdt":dxdt})
    return df_dpos, df_pos


@app.cell
def _(alt, df_dpos, df_pos):
    # Create plot using altair. Altair is optimized for interactivity and thus offers faster rendering time.  
    def _():
        # Chart 1 is x(t)
        line1 = alt.Chart(df_pos).mark_line().encode(
            x = alt.X('t', title='t'),
            y = alt.Y('x', title='x'),
        ).properties(
            title="x(t) with Marked Points for Averaging"
        )

        # Plot points to mark N-boundaries 
        scatter1 = alt.Chart(df_dpos).mark_point(color='green').encode(
            x='t', 
            y='x',
        )

        # Layer the line and scatter plot 
        chart1 = line1 + scatter1 

        # Create Chart 2 showing the averages
        chart2 = alt.Chart(df_dpos).mark_line().encode(
            x=alt.X('t', title='t'),
            y=alt.Y('dxdt', title='dx/dt'),
        ).properties(
            title="Collection of Averages"
        )

        return chart1 | chart2


    # # The same effect can be achieved using a conventional matplotlib plot. The fig object must be the final return rather than plt.show(). Additionally, matplotlib is less optimized so real-time plot updates are much slower. 
    # def _():
    #     fig,(ax1, ax2) = plt.subplots(1,2, figsize=(10,4)) 
    #     # Plot x(t) first
    #     ax1.plot(t, x)
    #     plt.grid(alpha=0.5)
    #     ax1.set_xlabel("t")
    #     ax1.set_ylabel("x")
    #     plt.tight_layout()

    #     # Plot grid points to use for averaging
    #     ax1.scatter(t[indices], x[indices], c='g', s=30)

    #     # Plot the collection of averages
    #     ax2.plot(t[indices[:-1]], dxdt)




    #     return fig

    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
