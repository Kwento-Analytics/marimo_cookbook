# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair==5.5.0",
#     "marimo",
#     "numpy==2.3.4",
#     "pandas==2.3.3",
#     "scikit-learn==1.7.2",
#     "tensorflow==2.20.0",
#     "matplotlib==3.10.7",
#     "pillow==12.0.0",
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
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import tensorflow as tf
    from io import BytesIO
    import base64
    from PIL import Image
    return BytesIO, PCA, TSNE, alt, mo, pd, plt, tf


@app.cell
def _(mo):
    mo.md("""
    # 👗 Fashion MNIST Cluster Visualization

    Explore the Fashion MNIST dataset through interactive clustering!
    This demo shows how t-SNE reduces 28×28 images (784 dimensions) down to 2D,
    revealing natural groupings of similar clothing items.

    **How to use:**
    - 🖱️ **Brush select** regions in the cluster plot
    - 👀 **View images** of selected items below
    - 🎨 **Filter by category** to explore specific clothing types
    - 📊 **See statistics** comparing your selection to the full dataset
    """)
    return


@app.cell
def _(mo, tf):
    # Load Fashion MNIST dataset
    mo.md("### 📦 Loading Fashion MNIST dataset...")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Use a subset for faster processing (first 5000 samples)
    n_samples = 5000
    x_subset = x_train[:n_samples]
    y_subset = y_train[:n_samples]

    # Category names
    category_names = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot"
    }

    mo.md(f"✅ Loaded {n_samples} Fashion MNIST samples")
    return category_names, x_subset, y_subset


@app.cell
def _(mo):
    # Controls
    mo.md("## 🎛️ Visualization Controls")

    # Dimensionality reduction method
    reduction_method = mo.ui.dropdown(
        options={
            "tsne": "t-SNE (slower, better separation)",
            "pca": "PCA (faster, linear)"
        },
        value="pca",
        label="Reduction Method"
    )

    # Perplexity for t-SNE
    perplexity = mo.ui.slider(
        start=5,
        stop=50,
        value=30,
        step=5,
        label="t-SNE Perplexity (only affects t-SNE)"
    )

    # Number of images to display
    n_images_display = mo.ui.slider(
        start=5,
        stop=50,
        value=20,
        step=5,
        label="Max images to display"
    )

    # Category filter
    category_filter = mo.ui.multiselect(
        options=[
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ],
        value=[
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ],
        label="Filter Categories"
    )

    # Compute button
    compute_button = mo.ui.button(
        label="🔄 Compute Embedding",
        value=0
    )

    return (
        category_filter,
        compute_button,
        n_images_display,
        perplexity,
        reduction_method,
    )


@app.cell
def _(
    category_filter,
    compute_button,
    mo,
    n_images_display,
    perplexity,
    reduction_method,
):
    # Display controls
    mo.hstack([
        mo.vstack([
            reduction_method,
            perplexity,
        ], align="start"),
        mo.vstack([
            category_filter,
            n_images_display,
            compute_button
        ], align="start")
    ], justify="start", gap=3)
    return


@app.cell
def _(PCA, TSNE, compute_button, mo, perplexity, reduction_method, x_subset):
    # Perform dimensionality reduction
    mo.md("### 🔄 Computing embedding...")

    # Trigger recomputation when button is clicked or method changes
    _trigger = (compute_button.value, reduction_method.selected_key, perplexity.value)

    # Flatten images for dimensionality reduction
    x_flat = x_subset.reshape(len(x_subset), -1)

    if reduction_method.selected_key == "tsne":
        mo.md(f"Computing t-SNE with perplexity={perplexity.value}... (this may take 30-60 seconds)")
        reducer = TSNE(n_components=2, perplexity=perplexity.value, random_state=42)
        embedding = reducer.fit_transform(x_flat)
    else:  # PCA
        mo.md("Computing PCA... (this takes ~2 seconds)")
        reducer = PCA(n_components=2, random_state=42)
        embedding = reducer.fit_transform(x_flat)

    mo.md("✅ Embedding complete!")
    return (embedding,)


@app.cell
def _(category_filter, category_names, embedding, pd, y_subset):
    # Create DataFrame with embeddings and labels
    df_fashion = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'category_id': y_subset,
        'category': [category_names[y] for y in y_subset],
        'index': range(len(y_subset))
    })

    # Apply category filter
    df_filtered = df_fashion[df_fashion['category'].isin(category_filter.value)].copy()

    return (df_filtered,)


@app.cell
def _(df_filtered, mo):
    mo.md(f"""
    ### 📊 Dataset: {len(df_filtered)} items after filtering
    """)
    return


@app.cell
def _(alt, df_filtered, mo):
    # Create cluster visualization
    brush = alt.selection_interval(name="brush")

    cluster_chart = alt.Chart(df_filtered).mark_point(size=30, filled=True).encode(
        x=alt.X('x:Q', scale=alt.Scale(zero=False), title='Dimension 1'),
        y=alt.Y('y:Q', scale=alt.Scale(zero=False), title='Dimension 2'),
        color=alt.condition(
            brush,
            alt.Color('category:N', scale=alt.Scale(scheme='category10'), legend=alt.Legend(title="Category")),
            alt.value('lightgray')
        ),
        opacity=alt.condition(brush, alt.value(0.8), alt.value(0.2)),
        tooltip=['category:N', 'index:Q']
    ).add_params(brush).properties(
        width=600,
        height=450,
        title='Fashion MNIST Cluster (Brush to Select Items)'
    )

    cluster_ui = mo.ui.altair_chart(cluster_chart)
    return (cluster_ui,)


@app.cell
def _(cluster_ui, mo):
    mo.md("## 🔍 Interactive Cluster Plot")
    cluster_ui
    return


@app.cell
def _(cluster_ui, pd):
    # Get selected data
    selection = cluster_ui.value

    if selection is not None and isinstance(selection, pd.DataFrame) and len(selection) > 0:
        selected_items = selection
    else:
        selected_items = pd.DataFrame()

    num_selected = len(selected_items)
    return num_selected, selected_items


@app.cell
def _(mo, num_selected):
    mo.md(f"""
    ### 🎯 Selected: **{num_selected}** items
    """)
    return


@app.cell
def _(BytesIO, mo, n_images_display, plt, selected_items, x_subset):
    # Display selected images
    mo.md("## 🖼️ Selected Images")


    # Get indices of selected items
    selected_indices = selected_items['index'].values

    # Limit number of images to display
    display_indices = selected_indices[:n_images_display.value]

    # Create image grid
    n_display = len(display_indices)
    n_cols = 5
    n_rows = (n_display + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, ax in enumerate(axes.flat):
        if idx < n_display:
            img_idx = display_indices[idx]
            img = x_subset[img_idx]
            category = selected_items[selected_items['index'] == img_idx]['category'].values[0]

            ax.imshow(img, cmap='gray')
            ax.set_title(f"{category}\n(idx: {img_idx})", fontsize=8)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()

    # Convert plot to image for display
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    # Display using mo.image
    mo.image(src=buf.getvalue())
    return


@app.cell
def _(df_filtered, mo, pd, selected_items):
    # Category breakdown
    mo.md("## 📊 Category Distribution")


    # Count categories in selection
    selected_counts = selected_items['category'].value_counts().reset_index()
    selected_counts.columns = ['Category', 'Selected Count']

    # Count categories in full filtered dataset
    all_counts = df_filtered['category'].value_counts().reset_index()
    all_counts.columns = ['Category', 'All Count']

    # Merge
    comparison = pd.merge(all_counts, selected_counts, on='Category', how='left').fillna(0)
    comparison['Selected Count'] = comparison['Selected Count'].astype(int)
    comparison['Selected %'] = (comparison['Selected Count'] / comparison['Selected Count'].sum() * 100).round(1)
    comparison['All %'] = (comparison['All Count'] / comparison['All Count'].sum() * 100).round(1)

    mo.ui.table(comparison)
    return


@app.cell
def _(alt, df_filtered, mo, pd, selected_items):
    # Category comparison bar chart
    # Prepare data for visualization
    selected_cat = selected_items['category'].value_counts().reset_index()
    selected_cat.columns = ['category', 'count']
    selected_cat['dataset'] = 'Selected'

    all_cat = df_filtered['category'].value_counts().reset_index()
    all_cat.columns = ['category', 'count']
    all_cat['dataset'] = 'All Data'

    # Normalize counts to percentages for fair comparison
    selected_cat['percentage'] = (selected_cat['count'] / selected_cat['count'].sum() * 100)
    all_cat['percentage'] = (all_cat['count'] / all_cat['count'].sum() * 100)

    combined = pd.concat([all_cat, selected_cat])

    bar_chart = alt.Chart(combined).mark_bar(opacity=0.7).encode(
        x=alt.X('category:N', title='Category', sort='-y'),
        y=alt.Y('percentage:Q', title='Percentage (%)'),
        color=alt.Color('dataset:N', scale=alt.Scale(scheme='set2')),
        xOffset='dataset:N'
    ).properties(
        width=600,
        height=300,
        title='Category Distribution: Selected vs All Data'
    )

    mo.ui.altair_chart(bar_chart)
    return


@app.cell
def _(mo, num_selected, selected_items):
    # Summary statistics
    mo.md("## 📈 Summary")
    most_common = selected_items['category'].value_counts().index[0]
    most_common_count = selected_items['category'].value_counts().values[0]

    mo.md(f"""
    - **Total selected:** {num_selected} items
    - **Most common category:** {most_common} ({most_common_count} items)
    - **Unique categories:** {selected_items['category'].nunique()} of 10
    """)
    return


if __name__ == "__main__":
    app.run()
