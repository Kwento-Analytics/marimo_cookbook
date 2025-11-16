# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair==5.5.0",
#     "marimo",
#     "numpy==2.3.4",
#     "pandas==2.3.3",
#     "vega-datasets==0.9.0",
#     "pyarrow",
# ]
# ///

import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import pandas as pd
    import numpy as np
    from vega_datasets import data

    return alt, data, mo, np, pd


@app.cell
def _(mo):
    mo.md(
        """
    # Interactive Data Selection & Brushing

    This demo showcases Altair's powerful selection capabilities integrated with Marimo.
    **Select data points by brushing (click and drag) or clicking** on the charts below,
    then watch as the selection propagates across all visualizations!

    Features:
    - **Brush selection** - Click and drag to select rectangular regions
    - **Point selection** - Click individual points
    - **Linked views** - Selection syncs across multiple charts
    - **Live statistics** - Compare selected vs all data
    - **Filters** - Narrow down the dataset before selecting
    """
    )
    return


@app.cell
def _(data):
    # Load the cars dataset
    cars = data.cars()
    # Clean column names for easier access
    cars = cars.rename(
        columns={
            "Miles_per_Gallon": "MPG",
            "Cylinders": "Cylinders",
            "Displacement": "Displacement",
            "Horsepower": "Horsepower",
            "Weight_in_lbs": "Weight",
            "Acceleration": "Acceleration",
            "Year": "Year",
            "Origin": "Origin",
            "Name": "Name",
        }
    )
    cars["Year"] = cars["Year"].dt.year
    return (cars,)


@app.cell
def _(cars, mo, np):
    # Filter Controls
    mo.md("## 🎛️ Data Filters")

    year_range = mo.ui.range_slider(
        start=int(cars["Year"].min()),
        stop=int(cars["Year"].max()),
        value=[int(cars["Year"].min()), int(cars["Year"].max())],
        step=1,
        label="Year Range",
    )

    origin_filter = mo.ui.multiselect(
        options=["USA", "Europe", "Japan"],
        value=["USA", "Europe", "Japan"],
        label="Origin",
    )

    cylinders_filter = mo.ui.multiselect(
        options=sorted(cars["Cylinders"].dropna().unique().astype(int).tolist()),
        value=sorted(cars["Cylinders"].dropna().unique().astype(int).tolist()),
        label="Cylinders",
    )

    mpg_range = mo.ui.range_slider(
        start=float(np.floor(cars["MPG"].min())),
        stop=float(np.ceil(cars["MPG"].max())),
        value=[float(np.floor(cars["MPG"].min())), float(np.ceil(cars["MPG"].max()))],
        step=1.0,
        label="MPG Range",
    )
    return cylinders_filter, mpg_range, origin_filter, year_range


@app.cell
def _(cylinders_filter, mo, mpg_range, origin_filter, year_range):
    # Display filters
    mo.hstack(
        [
            mo.vstack([year_range, mpg_range], align="start"),
            mo.vstack([origin_filter, cylinders_filter], align="start"),
        ],
        justify="start",
        gap=3,
    )
    return


@app.cell
def _(cars, cylinders_filter, mpg_range, origin_filter, year_range):
    # Apply filters to create filtered dataset
    filtered_cars = cars[
        (cars["Year"] >= year_range.value[0])
        & (cars["Year"] <= year_range.value[1])
        & (cars["Origin"].isin(origin_filter.value))
        & (cars["Cylinders"].isin(cylinders_filter.value))
        & (cars["MPG"] >= mpg_range.value[0])
        & (cars["MPG"] <= mpg_range.value[1])
    ].copy()
    return (filtered_cars,)


@app.cell
def _(filtered_cars, mo):
    mo.md(
        f"""
    ### 📊 Dataset Overview: {len(filtered_cars)} cars after filtering
    """
    )
    return


@app.cell
def _(alt, filtered_cars, mo):
    # Create brush and click selections
    brush = alt.selection_interval(name="brush")
    click = alt.selection_point(name="click")

    # Main scatter plot: Horsepower vs MPG
    base_scatter1 = (
        alt.Chart(filtered_cars)
        .mark_point(size=40, filled=True)
        .encode(
            x=alt.X("Horsepower:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("MPG:Q", scale=alt.Scale(zero=False)),
            color=alt.condition(
                brush | click,
                alt.Color("Origin:N", scale=alt.Scale(scheme="category10")),
                alt.value("lightgray"),
            ),
            opacity=alt.condition(brush | click, alt.value(1.0), alt.value(0.3)),
            tooltip=["Name:N", "Horsepower:Q", "MPG:Q", "Origin:N", "Year:O"],
        )
        .add_params(brush, click)
        .properties(
            width=400, height=300, title="Horsepower vs MPG (Brush or Click to Select)"
        )
    )

    scatter1_ui = mo.ui.altair_chart(base_scatter1)
    return brush, click, scatter1_ui


@app.cell
def _(alt, brush, click, filtered_cars, mo):
    # Second linked scatter plot: Weight vs Acceleration
    base_scatter2 = (
        alt.Chart(filtered_cars)
        .mark_point(size=40, filled=True)
        .encode(
            x=alt.X("Weight:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("Acceleration:Q", scale=alt.Scale(zero=False)),
            color=alt.condition(
                brush | click,
                alt.Color("Origin:N", scale=alt.Scale(scheme="category10")),
                alt.value("lightgray"),
            ),
            opacity=alt.condition(brush | click, alt.value(1.0), alt.value(0.3)),
            tooltip=["Name:N", "Weight:Q", "Acceleration:Q", "Origin:N", "Year:O"],
        )
        .add_params(brush, click)
        .properties(
            width=400, height=300, title="Weight vs Acceleration (Linked Selection)"
        )
    )

    scatter2_ui = mo.ui.altair_chart(base_scatter2)
    return (scatter2_ui,)


@app.cell
def _(mo, scatter1_ui, scatter2_ui):
    # Display linked charts side by side
    mo.md("## 🔗 Linked Scatter Plots")
    mo.hstack([scatter1_ui, scatter2_ui], justify="center")
    return


@app.cell
def _(pd, scatter1_ui):
    # Get selected data - mo.ui.altair_chart returns a DataFrame of selected rows
    selection = scatter1_ui.value

    if (
        selection is not None
        and isinstance(selection, pd.DataFrame)
        and len(selection) > 0
    ):
        selected_cars = selection
    else:
        selected_cars = pd.DataFrame()

    num_selected = len(selected_cars)
    return num_selected, selected_cars


@app.cell
def _(mo, num_selected):
    # Clear selection button and selection count
    clear_button = mo.ui.button(label="Clear Selection", value=0)

    mo.hstack(
        [mo.md(f"### Selected: **{num_selected}** cars"), clear_button],
        justify="start",
        gap=2,
    )
    return


@app.cell
def _(mo, num_selected, selected_cars):
    # Display selected data
    mo.md("## Selected Data")

    if num_selected == 0:
        mo.md(
            "*No data selected. Brush or click on the charts above to select points.*"
        )

    # Marimo has issues wraping table displays in a control statement, so here we display the table no matter what.
    # Show key columns for selected cars
    mo.ui.table(selected_cars)
    return


@app.cell
def _(filtered_cars, mo, num_selected, pd, selected_cars):
    # Summary statistics comparison
    mo.md("## Statistics: Selected vs All Data")

    stats_comparison = pd.DataFrame(
        {
            "Metric": [
                "Count",
                "Avg MPG",
                "Avg Horsepower",
                "Avg Weight",
                "Avg Acceleration",
            ],
            "All Data": [
                len(filtered_cars),
                f"{filtered_cars['MPG'].mean():.2f}",
                f"{filtered_cars['Horsepower'].mean():.2f}",
                f"{filtered_cars['Weight'].mean():.2f}",
                f"{filtered_cars['Acceleration'].mean():.2f}",
            ],
            "Selected": [
                num_selected,
                f"{selected_cars['MPG'].mean():.2f}",
                f"{selected_cars['Horsepower'].mean():.2f}",
                f"{selected_cars['Weight'].mean():.2f}",
                f"{selected_cars['Acceleration'].mean():.2f}",
            ],
        }
    )
    mo.ui.table(stats_comparison)
    return


@app.cell
def _(alt, filtered_cars, mo, pd, selected_cars):
    # Histogram comparison: Selected vs All
    mo.md("## 📈 Distribution Comparison")

    # Create comparison histograms for MPG
    all_data_df = filtered_cars[["MPG"]].copy()
    all_data_df["Dataset"] = "All Data"

    selected_data_df = selected_cars[["MPG"]].copy()
    selected_data_df["Dataset"] = "Selected"

    combined_hist_data = pd.concat([all_data_df, selected_data_df])

    histogram = (
        alt.Chart(combined_hist_data)
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("MPG:Q", bin=alt.Bin(maxbins=20), title="Miles per Gallon"),
            y=alt.Y("count()", title="Frequency"),
            color=alt.Color("Dataset:N", scale=alt.Scale(scheme="set1")),
        )
        .properties(
            width=600, height=250, title="MPG Distribution: Selected vs All Data"
        )
    )

    mo.ui.altair_chart(histogram)
    return


@app.cell
def _(mo, num_selected, selected_cars):
    # Export functionality
    mo.md("## 💾 Export Selected Data")

    # Create CSV download
    csv_data = selected_cars.to_csv(index=False)
    mo.md(
        f"""
    **{num_selected} cars selected** and ready to export.

    *Note: In a future version, you'll be able to download this data as CSV. 
    For now, you can copy the data from the table above.*
    """
    )
    return


if __name__ == "__main__":
    app.run()
