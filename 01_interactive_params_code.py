# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair==6.0.0",
#     "marimo",
#     "numpy==2.3.4",
#     "pandas==2.3.3",
# ]
# ///

import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import altair as alt
    import pandas as pd
    return alt, mo, np, pd


@app.cell
def _(mo):
    mo.md("""
    # Marimo Interactive Parameters Playground

    Welcome to the interactive parameters demo! This dashboard showcases Marimo's reactive notebook capabilities.
    **Change any parameter and watch everything update instantly** - no need to re-run cells manually.

    Explore the tabs below to see different interactive features in action.
    """)
    return


@app.cell
def _(mo, np):
    # Wave Function Controls
    amplitude = mo.ui.slider(0.1, 5.0, value=1.0, step=0.1, label="Amplitude")
    frequency = mo.ui.slider(0.1, 5.0, value=1.0, step=0.1, label="Frequency")
    phase = mo.ui.slider(0, 2*np.pi, value=0, step=0.1, label="Phase Shift")

    wave_type = mo.ui.dropdown(
        options={
            "sine": "Sine Wave",
            "cosine": "Cosine Wave", 
            "square": "Square Wave",
            "sawtooth": "Sawtooth Wave",
            "polynomial": "Polynomial (x²)",
            "exponential": "Exponential"
        },
        value="sine",
        label="Function Type"
    )

    show_grid = mo.ui.checkbox(value=True, label="Show Grid")
    show_points = mo.ui.checkbox(value=False, label="Show Points")

    x_range = mo.ui.range_slider(0, 4*np.pi, value=[0, 4*np.pi], step=0.1, label="X-axis Range")
    return (
        amplitude,
        frequency,
        phase,
        show_grid,
        show_points,
        wave_type,
        x_range,
    )


@app.cell
def _(
    alt,
    amplitude,
    frequency,
    mo,
    np,
    pd,
    phase,
    show_grid,
    show_points,
    wave_type,
    x_range,
):
    # Wave Function Logic
    x = np.linspace(x_range.value[0], x_range.value[1], 500)

    if wave_type.selected_key == "sine":
        y = amplitude.value * np.sin(frequency.value * x + phase.value)
        equation = rf"$y = {amplitude.value:.1f} \sin({frequency.value:.1f}x + {phase.value:.2f})$"
    elif wave_type.selected_key == "cosine":
        y = amplitude.value * np.cos(frequency.value * x + phase.value)
        equation = rf"$y = {amplitude.value:.1f} \cos({frequency.value:.1f}x + {phase.value:.2f})$"
    elif wave_type.selected_key == "square":
        y = amplitude.value * np.sign(np.sin(frequency.value * x + phase.value))
        equation = rf"$y = {amplitude.value:.1f} \cdot \mathrm{{sign}}(\sin({frequency.value:.1f}x + {phase.value:.2f}))$"
    elif wave_type.selected_key == "sawtooth":
        y = amplitude.value * (2 * (frequency.value * x / (2*np.pi) + phase.value/(2*np.pi) - np.floor(frequency.value * x / (2*np.pi) + phase.value/(2*np.pi) + 0.5)))
        equation = rf"$y = {amplitude.value:.1f} \cdot \mathrm{{sawtooth}}({frequency.value:.1f}x + {phase.value:.2f})$"
    elif wave_type.selected_key == "polynomial":
        y = amplitude.value * ((x - phase.value) / frequency.value) ** 2
        equation = rf"$y = {amplitude.value:.1f} \cdot \left(\frac{{x - {phase.value:.2f}}}{{{frequency.value:.1f}}}\right)^2$"
    else:  # exponential
        y = amplitude.value * np.exp(frequency.value * (x - phase.value) / 10)
        equation = rf"$y = {amplitude.value:.1f} \cdot e^{{{frequency.value:.1f}(x - {phase.value:.2f})/10}}$"

    wave_df = pd.DataFrame({"x": x, "y": y})

    wave_chart = alt.Chart(wave_df).mark_line(color='steelblue', size=2).encode(
        x=alt.X('x:Q', title='x'),
        y=alt.Y('y:Q', title='y', scale=alt.Scale(domain=[wave_df['y'].min() - 0.5, wave_df['y'].max() + 0.5]))
    ).properties(
        width=600,
        height=300,
        title=f"{wave_type.value.title()} Function"
    )

    if show_points.value:
        points = alt.Chart(wave_df.iloc[::10]).mark_point(color='red', size=30).encode(
            x='x:Q',
            y='y:Q'
        )
        wave_chart = wave_chart + points

    wave_chart = wave_chart.configure_axis(
        grid=show_grid.value
    )

    wave_tab = mo.vstack([
        mo.md(f"## Wave Function Explorer\n\n{equation}"),
        mo.hstack([
            mo.vstack([amplitude, frequency, phase], align="start"),
            mo.vstack([wave_type, show_grid, show_points], align="start"),
        ], justify="start", gap=2),
        mo.vstack([x_range]),
        mo.ui.altair_chart(wave_chart)
    ])
    return (wave_tab,)


@app.cell
def _(mo):
    # Distribution Controls
    sample_size = mo.ui.slider(10, 10000, value=500, step=10, label="Sample Size")

    dist1_type = mo.ui.dropdown(
        options={
            "normal": "Normal",
            "uniform": "Uniform",
            "exponential": "Exponential",
            "lognormal": "Log-Normal"
        },
        value="normal",
        label="Distribution 1"
    )

    dist1_param1 = mo.ui.number(start=-10, stop=10, value=0, step=0.1, label="Parameter 1 (μ or min)")
    dist1_param2 = mo.ui.number(start=0.1, stop=10, value=1, step=0.1, label="Parameter 2 (σ or max)")

    dist2_type = mo.ui.dropdown(
        options={
            "normal": "Normal",
            "uniform": "Uniform",
            "exponential": "Exponential",
            "lognormal": "Log-Normal"
        },
        value="uniform",
        label="Distribution 2"
    )

    dist2_param1 = mo.ui.number(start=-10, stop=10, value=-2, step=0.1, label="Parameter 1 (μ or min)")
    dist2_param2 = mo.ui.number(start=0.1, stop=10, value=2, step=0.1, label="Parameter 2 (σ or max)")

    seed = mo.ui.number(start=0, stop=10000, value=42, step=1, label="Random Seed")

    show_comparison = mo.ui.checkbox(value=True, label="Show Comparison")
    return (
        dist1_param1,
        dist1_param2,
        dist1_type,
        dist2_param1,
        dist2_param2,
        dist2_type,
        sample_size,
        seed,
        show_comparison,
    )


@app.cell
def _(
    alt,
    dist1_param1,
    dist1_param2,
    dist1_type,
    dist2_param1,
    dist2_param2,
    dist2_type,
    mo,
    np,
    pd,
    sample_size,
    seed,
    show_comparison,
):
    # Distribution Logic
    np.random.seed(seed.value)
    def generate_distribution(dist_type, param1, param2, size):
        if dist_type == "normal":
            return np.random.normal(param1, param2, size)
        elif dist_type == "uniform":
            return np.random.uniform(param1, param2, size)
        elif dist_type == "exponential":
            return np.random.exponential(param2, size) + param1
        else:  # lognormal
            return np.random.lognormal(param1, param2, size)

    data1 = generate_distribution(dist1_type.selected_key, dist1_param1.value, dist1_param2.value, sample_size.value)
    data2 = generate_distribution(dist2_type.selected_key, dist2_param1.value, dist2_param2.value, sample_size.value)

    # Create combined dataframe
    df1 = pd.DataFrame({"value": data1, "distribution": "Distribution 1"})
    df2 = pd.DataFrame({"value": data2, "distribution": "Distribution 2"})

    if show_comparison.value:
        combined_df = pd.concat([df1, df2])
        dist_chart = alt.Chart(combined_df).mark_bar(opacity=0.6).encode(
            x=alt.X('value:Q', bin=alt.Bin(maxbins=40), title='Value', scale=alt.Scale(domain=[-8, 8])),
            y=alt.Y('count()', title='Frequency'),
            color=alt.Color('distribution:N', scale=alt.Scale(scheme='category10'))
        ).properties(
            width=600,
            height=300,
            title="Distribution Comparison"
        )
    else:
        dist_chart = alt.Chart(df1).mark_bar(color='steelblue').encode(
            x=alt.X('value:Q', bin=alt.Bin(maxbins=40), title='Value', scale=alt.Scale(domain=[-8, 8])),
            y=alt.Y('count()', title='Frequency')
        ).properties(
            width=600,
            height=300,
            title=f"{dist1_type.value.title()} Distribution"
        )

    # Summary statistics
    stats1 = pd.DataFrame({
        "Metric": ["Mean", "Std Dev", "Min", "Max", "Median"],
        "Distribution 1": [
            f"{np.mean(data1):.3f}",
            f"{np.std(data1):.3f}",
            f"{np.min(data1):.3f}",
            f"{np.max(data1):.3f}",
            f"{np.median(data1):.3f}"
        ]
    })

    if show_comparison.value:
        stats1["Distribution 2"] = [
            f"{np.mean(data2):.3f}",
            f"{np.std(data2):.3f}",
            f"{np.min(data2):.3f}",
            f"{np.max(data2):.3f}",
            f"{np.median(data2):.3f}"
        ]

    dist_tab = mo.vstack([
        mo.md("## Distribution Generator & Comparison"),
        mo.hstack([sample_size, seed, show_comparison], justify="start"),
        mo.hstack([
            mo.vstack([
                mo.md("### Distribution 1"),
                dist1_type,
                dist1_param1,
                dist1_param2
            ], align="start"),
            mo.vstack([
                mo.md("### Distribution 2"),
                dist2_type,
                dist2_param1,
                dist2_param2
            ], align="start") if show_comparison.value else mo.md("")
        ], justify="start", gap=2),
        mo.ui.altair_chart(dist_chart),
        mo.md("### Summary Statistics"),
        mo.ui.table(stats1)
    ])
    return (dist_tab,)


@app.cell
def _(mo):
    # Widget Gallery Controls
    gallery_slider = mo.ui.slider(0, 100, value=50, label="Slider")
    gallery_text = mo.ui.text(value="Hello Marimo!", label="Text Input")
    gallery_number = mo.ui.number(start=0, stop=100, value=42, label="Number")
    gallery_dropdown = mo.ui.dropdown(
        options=["Option A", "Option B", "Option C"],
        value="Option A",
        label="Dropdown"
    )
    gallery_radio = mo.ui.radio(
        options=["Red", "Green", "Blue"],
        value="Red",
        label="Radio Buttons"
    )
    gallery_checkbox = mo.ui.checkbox(value=True, label="Checkbox")
    gallery_multiselect = mo.ui.multiselect(
        options=["Apple", "Banana", "Cherry", "Date"],
        value=["Apple"],
        label="Multi-select"
    )
    gallery_date = mo.ui.date(label="Date Picker")
    return (
        gallery_checkbox,
        gallery_date,
        gallery_dropdown,
        gallery_multiselect,
        gallery_number,
        gallery_radio,
        gallery_slider,
        gallery_text,
    )


@app.cell
def _(
    gallery_checkbox,
    gallery_date,
    gallery_dropdown,
    gallery_multiselect,
    gallery_number,
    gallery_radio,
    gallery_slider,
    gallery_text,
    mo,
):
    # Widget Gallery Display
    widget_tab = mo.vstack([
        mo.md("## Interactive Widget Gallery"),
        mo.md("Explore all available Marimo UI widgets. Each widget's current value is displayed below it."),

        mo.hstack([
            mo.vstack([
                gallery_slider,
                mo.md(f"**Value:** `{gallery_slider.value}`"),
            ], align="start"),
            mo.vstack([
                gallery_number,
                mo.md(f"**Value:** `{gallery_number.value}`"),
            ], align="start"),
        ], justify="start", gap=2),

        mo.hstack([
            mo.vstack([
                gallery_text,
                mo.md(f"**Value:** `'{gallery_text.value}'`"),
            ], align="start"),
            mo.vstack([
                gallery_dropdown,
                mo.md(f"**Value:** `'{gallery_dropdown.value}'`"),
            ], align="start"),
        ], justify="start", gap=2),

        mo.hstack([
            mo.vstack([
                gallery_radio,
                mo.md(f"**Value:** `'{gallery_radio.value}'`"),
            ], align="start"),
            mo.vstack([
                gallery_checkbox,
                mo.md(f"**Value:** `{gallery_checkbox.value}`"),
            ], align="start"),
        ], justify="start", gap=2),

        mo.vstack([
            gallery_multiselect,
            mo.md(f"**Value:** `{gallery_multiselect.value}`"),
        ], align="start"),

        mo.vstack([
            gallery_date,
            mo.md(f"**Value:** `{gallery_date.value}`"),
        ], align="start"),

        mo.md("""
        ### Live Demo
        The equation below uses values from the widgets above:
        """),
        mo.md(
            f"""
            If **{gallery_text.value}** chooses **{gallery_dropdown.value}** 
            with intensity **{gallery_slider.value}**, and the color is **{gallery_radio.value}**, 
            then the result is: **{gallery_number.value * gallery_slider.value / 100:.2f}**
            """
        )
    ])
    return (widget_tab,)


@app.cell
def _(dist_tab, mo, wave_tab, widget_tab):
    # Assemble the dashboard with tabs
    tabs = mo.ui.tabs({
        "Wave Functions": wave_tab,
        "Distributions": dist_tab,
        "Widget Gallery": widget_tab
    })

    tabs
    return


if __name__ == "__main__":
    app.run()
