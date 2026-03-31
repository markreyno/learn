# ============================================================
# SEABORN MAIN CONCEPTS
# ============================================================
# Seaborn is a statistical visualization library built on Matplotlib.
# It provides higher-level abstractions for common statistical plots
# and integrates natively with pandas DataFrames.
#
# Two APIs:
#   Figure-level  — returns a FacetGrid/PairGrid; manages its own Figure
#   Axes-level    — plots onto a Matplotlib Axes (ax= parameter)

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load built-in datasets for examples
tips    = sns.load_dataset("tips")       # restaurant tips
iris    = sns.load_dataset("iris")       # flower measurements
titanic = sns.load_dataset("titanic")    # passenger survival
flights = sns.load_dataset("flights")    # monthly passengers (wide/long)
penguins = sns.load_dataset("penguins") # penguin measurements

rng = np.random.default_rng(42)


# ============================================================
# 1. SEABORN THEMES & STYLING
# ============================================================

# --- Themes (set_theme) ---
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
# style options:  "white"  "whitegrid"  "dark"  "darkgrid"  "ticks"

# Apply temporarily with a context manager
with sns.axes_style("ticks"):
    fig, ax = plt.subplots()
    sns.lineplot(data=tips, x="total_bill", y="tip", ax=ax)
    plt.close()

# --- Context (scale for different outputs) ---
sns.set_context("notebook")   # "paper"  "notebook"  "talk"  "poster"

# --- Color palettes ---
print(sns.color_palette())                    # current palette

# Named palettes
sns.set_palette("deep")                       # qualitative
sns.set_palette("Blues")                      # sequential
sns.set_palette("RdBu_r")                     # diverging

# Palette types:
#   Qualitative:  deep, muted, bright, pastel, dark, colorblind, tab10
#   Sequential:   Blues, Greens, viridis, rocket, mako, flare, crest
#   Diverging:    RdBu_r, coolwarm, vlag, icefire

# Preview a palette
fig, ax = plt.subplots(figsize=(6, 1))
sns.palplot(sns.color_palette("tab10"))
plt.savefig("palette.png", bbox_inches="tight")
plt.close()

sns.set_theme()   # reset to defaults for remaining examples


# ============================================================
# 2. RELATIONAL PLOTS — scatterplot & lineplot
# ============================================================

# --- scatterplot (axes-level) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.scatterplot(
    data=tips, x="total_bill", y="tip",
    hue="sex",          # color by category
    size="size",        # point size by column
    style="smoker",     # marker style by category
    palette="Set1",
    alpha=0.8,
    ax=axes[0],
)
axes[0].set_title("Scatter: Tips dataset")

# --- lineplot (axes-level) ---
fmri = sns.load_dataset("fmri")
sns.lineplot(
    data=fmri, x="timepoint", y="signal",
    hue="region",       # separate lines by region
    style="event",      # line style by event
    errorbar="sd",      # show standard deviation band
    ax=axes[1],
)
axes[1].set_title("Line: FMRI signal over time")

fig.tight_layout()
fig.savefig("fig_relational.png", dpi=150, bbox_inches="tight")
plt.close()

# --- relplot (figure-level) — adds faceting ---
g = sns.relplot(
    data=tips, x="total_bill", y="tip",
    hue="sex", col="time", row="smoker",   # facet grid
    kind="scatter", height=3, aspect=1.2,
)
g.set_axis_labels("Total Bill ($)", "Tip ($)")
g.set_titles("{row_name} | {col_name}")
g.figure.savefig("fig_relplot.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 3. DISTRIBUTION PLOTS
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# --- histplot ---
sns.histplot(data=tips, x="total_bill", bins=20,
             kde=True, color="steelblue", ax=axes[0, 0])
axes[0, 0].set_title("histplot (with KDE)")

# Stacked / hue
sns.histplot(data=tips, x="total_bill", hue="sex",
             multiple="stack", bins=20, ax=axes[0, 1])
axes[0, 1].set_title("histplot (stacked by sex)")

# --- kdeplot — kernel density estimate ---
sns.kdeplot(data=tips, x="total_bill", hue="sex",
            fill=True, alpha=0.4, ax=axes[0, 2])
axes[0, 2].set_title("kdeplot")

# 2D KDE
sns.kdeplot(data=tips, x="total_bill", y="tip",
            fill=True, cmap="Blues", ax=axes[1, 0])
axes[1, 0].set_title("2D kdeplot")

# --- ecdfplot — empirical CDF ---
sns.ecdfplot(data=tips, x="total_bill", hue="sex", ax=axes[1, 1])
axes[1, 1].set_title("ecdfplot (CDF)")

# --- rugplot — tick marks on axis ---
sns.histplot(data=tips, x="total_bill", bins=20, ax=axes[1, 2])
sns.rugplot(data=tips, x="total_bill", color="tomato",
            height=0.05, ax=axes[1, 2])
axes[1, 2].set_title("histplot + rugplot")

fig.suptitle("Distribution Plots", fontsize=14)
fig.tight_layout()
fig.savefig("fig_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# --- displot (figure-level) ---
g = sns.displot(data=tips, x="total_bill", hue="sex",
                col="time", kind="kde", fill=True, height=4)
g.figure.savefig("fig_displot.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 4. CATEGORICAL PLOTS
# ============================================================

fig, axes = plt.subplots(2, 4, figsize=(20, 9))

# --- stripplot — all points, jittered ---
sns.stripplot(data=tips, x="day", y="total_bill", hue="sex",
              jitter=True, dodge=True, alpha=0.6, ax=axes[0, 0])
axes[0, 0].set_title("stripplot")

# --- swarmplot — non-overlapping points ---
sns.swarmplot(data=tips, x="day", y="total_bill", hue="sex",
              dodge=True, size=3, ax=axes[0, 1])
axes[0, 1].set_title("swarmplot")

# --- boxplot ---
sns.boxplot(data=tips, x="day", y="total_bill", hue="sex",
            palette="Set2", ax=axes[0, 2])
axes[0, 2].set_title("boxplot")

# --- violinplot ---
sns.violinplot(data=tips, x="day", y="total_bill", hue="sex",
               split=True, inner="quart", palette="muted", ax=axes[0, 3])
axes[0, 3].set_title("violinplot (split)")

# --- boxenplot — letter-value plot for large datasets ---
sns.boxenplot(data=tips, x="day", y="total_bill", palette="pastel",
              ax=axes[1, 0])
axes[1, 0].set_title("boxenplot")

# --- barplot — shows mean + confidence interval ---
sns.barplot(data=tips, x="day", y="total_bill", hue="sex",
            errorbar="ci", palette="Set1", ax=axes[1, 1])
axes[1, 1].set_title("barplot (mean + 95% CI)")

# --- countplot — counts of observations per category ---
sns.countplot(data=tips, x="day", hue="sex", palette="Set2",
              ax=axes[1, 2])
axes[1, 2].set_title("countplot")

# --- pointplot — mean + CI as points connected by lines ---
sns.pointplot(data=tips, x="day", y="total_bill", hue="sex",
              dodge=True, markers=["o", "s"], ax=axes[1, 3])
axes[1, 3].set_title("pointplot")

fig.suptitle("Categorical Plots", fontsize=14)
fig.tight_layout()
fig.savefig("fig_categorical.png", dpi=150, bbox_inches="tight")
plt.close()

# --- catplot (figure-level) ---
g = sns.catplot(data=tips, x="day", y="total_bill", hue="sex",
                col="time", kind="box", height=4, aspect=0.8)
g.figure.savefig("fig_catplot.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 5. REGRESSION PLOTS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# --- regplot — scatter + regression line ---
sns.regplot(data=tips, x="total_bill", y="tip",
            scatter_kws={"alpha": 0.5}, line_kws={"color": "tomato"},
            ax=axes[0])
axes[0].set_title("regplot (linear)")

# Polynomial regression
sns.regplot(data=tips, x="total_bill", y="tip",
            order=2, scatter_kws={"alpha": 0.4}, ax=axes[1])
axes[1].set_title("regplot (order=2, polynomial)")

# --- residplot — residuals of the regression ---
sns.residplot(data=tips, x="total_bill", y="tip",
              scatter_kws={"alpha": 0.5}, ax=axes[2])
axes[2].axhline(0, color="gray", linestyle="--")
axes[2].set_title("residplot (residuals)")

fig.tight_layout()
fig.savefig("fig_regression.png", dpi=150, bbox_inches="tight")
plt.close()

# --- lmplot (figure-level) — regplot with faceting ---
g = sns.lmplot(data=tips, x="total_bill", y="tip",
               hue="smoker", col="time", row="sex",
               height=3, aspect=1.1, scatter_kws={"alpha": 0.5})
g.figure.savefig("fig_lmplot.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 6. HEATMAPS
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Correlation matrix heatmap
corr = tips[["total_bill", "tip", "size"]].corr()
sns.heatmap(
    corr,
    annot=True,          # show values in cells
    fmt=".2f",           # format for annotations
    cmap="RdBu_r",       # diverging colormap
    vmin=-1, vmax=1,
    linewidths=0.5,
    square=True,
    ax=axes[0],
)
axes[0].set_title("Correlation Heatmap")

# Pivot table heatmap
flights_pivot = flights.pivot_table(index="month", columns="year",
                                    values="passengers")
sns.heatmap(
    flights_pivot,
    cmap="YlOrRd",
    annot=True,
    fmt="d",
    linewidths=0.3,
    cbar_kws={"label": "Passengers"},
    ax=axes[1],
)
axes[1].set_title("Flights Heatmap (pivot table)")

fig.tight_layout()
fig.savefig("fig_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 7. PAIRPLOT & PAIRGRID
# ============================================================

# --- pairplot — scatter matrix for all numeric columns ---
g = sns.pairplot(
    iris,
    hue="species",           # color by category
    diag_kind="kde",         # diagonal: kde or hist
    plot_kws={"alpha": 0.6},
    palette="Set1",
)
g.figure.suptitle("Pairplot — Iris dataset", y=1.02)
g.figure.savefig("fig_pairplot.png", dpi=150, bbox_inches="tight")
plt.close()

# --- PairGrid — fully customisable pair grid ---
g = sns.PairGrid(penguins.dropna(), hue="species", palette="Set2")
g.map_upper(sns.scatterplot, alpha=0.6)   # upper triangle
g.map_lower(sns.kdeplot, fill=True)       # lower triangle
g.map_diag(sns.histplot, kde=True)        # diagonal
g.add_legend()
g.figure.savefig("fig_pairgrid.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 8. CLUSTERMAP
# ============================================================
# Hierarchically-clustered heatmap — reorders rows and columns
# to group similar values together.

# Use the flights pivot as a numeric matrix
flights_pivot = flights.pivot_table(index="month", columns="year",
                                    values="passengers")

g = sns.clustermap(
    flights_pivot,
    cmap="YlOrRd",
    standard_scale=1,        # normalize each column to [0, 1]
    figsize=(10, 8),
    annot=False,
    linewidths=0.3,
    col_cluster=True,        # cluster columns
    row_cluster=True,        # cluster rows
    dendrogram_ratio=0.15,   # size of dendrogram
)
g.figure.suptitle("Clustermap — Flights", y=1.01)
g.figure.savefig("fig_clustermap.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 9. FACETGRID
# ============================================================
# FacetGrid splits data across a grid of subplots by category.
# Use it to build custom multi-panel figures.

# Map any plotting function across facets
g = sns.FacetGrid(tips, col="time", row="smoker",
                  hue="sex", height=3.5, aspect=1.1,
                  margin_titles=True)
g.map(sns.scatterplot, "total_bill", "tip", alpha=0.6)
g.add_legend()
g.set_axis_labels("Total Bill ($)", "Tip ($)")
g.set_titles(row_template="{row_name}", col_template="{col_name}")
g.figure.savefig("fig_facetgrid.png", dpi=150, bbox_inches="tight")
plt.close()

# Map a custom function
g = sns.FacetGrid(tips, col="day", height=3)
g.map_dataframe(sns.histplot, x="total_bill", bins=15, kde=True)
g.set_axis_labels("Total Bill ($)", "Count")
g.figure.savefig("fig_facetgrid_hist.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 10. JOINTPLOT
# ============================================================
# Shows bivariate distribution plus marginal univariate distributions.

# Scatter + KDE marginals
g = sns.jointplot(data=tips, x="total_bill", y="tip",
                  kind="scatter",         # "scatter" "kde" "hist" "hex" "reg" "resid"
                  hue="sex",
                  height=6,
                  marginal_kws={"fill": True, "alpha": 0.5})
g.set_axis_labels("Total Bill ($)", "Tip ($)")
g.figure.savefig("fig_joint_scatter.png", dpi=150, bbox_inches="tight")
plt.close()

# KDE jointplot
g = sns.jointplot(data=tips, x="total_bill", y="tip",
                  kind="kde", fill=True, cmap="Blues", height=5)
g.figure.savefig("fig_joint_kde.png", dpi=150, bbox_inches="tight")
plt.close()

# Hex jointplot
g = sns.jointplot(data=tips, x="total_bill", y="tip",
                  kind="hex", height=5, cmap="viridis")
g.figure.savefig("fig_joint_hex.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 11. WORKING WITH MATPLOTLIB (AXES-LEVEL INTEGRATION)
# ============================================================
# Axes-level functions return a Matplotlib Axes and accept ax=.
# This lets you embed Seaborn plots inside any Matplotlib layout.

fig = plt.figure(figsize=(14, 5))

# Mix Seaborn and raw Matplotlib in the same figure
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

sns.boxplot(data=tips, x="day", y="total_bill", palette="pastel", ax=ax1)
ax1.set_title("Seaborn boxplot")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30)

sns.kdeplot(data=tips, x="tip", hue="sex", fill=True, ax=ax2)
ax2.set_title("Seaborn kdeplot")

# Raw Matplotlib in the same figure
ax3.plot(np.sort(tips["total_bill"]), np.linspace(0, 1, len(tips)),
         color="steelblue")
ax3.set_title("Raw Matplotlib CDF")
ax3.set_xlabel("Total Bill ($)")
ax3.set_ylabel("Cumulative Probability")

fig.suptitle("Mixed Seaborn + Matplotlib", fontsize=13)
fig.tight_layout()
fig.savefig("fig_mixed.png", dpi=150, bbox_inches="tight")
plt.close()

# Post-processing a Seaborn figure-level plot with Matplotlib
g = sns.displot(data=tips, x="total_bill", hue="sex", kind="kde",
                fill=True, height=4, aspect=1.5)
g.axes[0, 0].axvline(tips["total_bill"].mean(), color="black",
                      linestyle="--", label="Mean")
g.axes[0, 0].legend()
g.figure.savefig("fig_postprocess.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 12. CUSTOMISING LABELS, TITLES & ANNOTATIONS
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

sns.barplot(data=tips, x="day", y="total_bill", hue="sex",
            palette="Set2", errorbar="ci", ax=ax)

# Labels and title
ax.set_title("Average Total Bill by Day and Sex", fontsize=14, pad=12)
ax.set_xlabel("Day of Week", fontsize=12)
ax.set_ylabel("Average Total Bill ($)", fontsize=12)

# Rotate tick labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# Add value labels on each bar
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f", padding=3, fontsize=9)

# Move legend outside the plot
ax.legend(title="Sex", bbox_to_anchor=(1.01, 1), loc="upper left")

# Remove top/right spines (Seaborn style override)
sns.despine(ax=ax)

fig.tight_layout()
fig.savefig("fig_custom_labels.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 13. HANDLING LONG vs WIDE DATA
# ============================================================
# Seaborn works best with tidy (long) format: one row per observation.

# Wide data
wide = pd.DataFrame({
    "day":   ["Mon", "Tue", "Wed", "Thu", "Fri"],
    "sales": [100, 120, 90, 140, 160],
    "costs": [60,  70,  55, 80,  95],
})

# Convert to long (tidy) format
long = wide.melt(id_vars="day", var_name="category", value_name="amount")
print(long)
#     day category  amount
#   0 Mon    sales     100
#   1 Tue    sales     120  ...

# Now Seaborn can use hue="category"
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Wide format — one series at a time
axes[0].plot(wide["day"], wide["sales"], label="Sales", marker="o")
axes[0].plot(wide["day"], wide["costs"], label="Costs", marker="o")
axes[0].legend()
axes[0].set_title("Matplotlib (wide)")

# Long format — hue handles grouping automatically
sns.lineplot(data=long, x="day", y="amount", hue="category",
             marker="o", ax=axes[1])
axes[1].set_title("Seaborn (long / tidy)")

fig.tight_layout()
fig.savefig("fig_long_vs_wide.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 14. STATISTICAL ANNOTATIONS & UNCERTAINTY
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# errorbar options: "ci" (95% CI), "pi" (prediction interval),
#                  "se" (std error), "sd" (std dev), None

sns.lineplot(data=tips, x="size", y="total_bill",
             errorbar="ci", err_style="band",
             marker="o", ax=axes[0])
axes[0].set_title("errorbar='ci', err_style='band'")

sns.lineplot(data=tips, x="size", y="total_bill",
             errorbar="sd", err_style="bars",
             marker="o", ax=axes[1])
axes[1].set_title("errorbar='sd', err_style='bars'")

# Bootstrap CI explicitly
sns.pointplot(data=tips, x="day", y="total_bill",
              errorbar=("ci", 99),     # 99% confidence interval
              capsize=0.15,
              ax=axes[2])
axes[2].set_title("pointplot with 99% CI")

fig.tight_layout()
fig.savefig("fig_uncertainty.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 15. OBJECT INTERFACE (Seaborn v0.12+)
# ============================================================
# The newer objects API uses method chaining and is more composable.

from seaborn import objects as so

fig = (
    so.Plot(tips, x="total_bill", y="tip", color="sex")
    .add(so.Dots(alpha=0.5))
    .add(so.Line(), so.PolyFit(order=1))   # add regression line
    .label(title="Objects API: Scatter + Regression",
           x="Total Bill ($)", y="Tip ($)")
    .layout(size=(8, 5))
    .on(plt.figure())   # attach to a Matplotlib figure
)
fig.save("fig_objects.png", bbox_inches="tight")
plt.close("all")


# ============================================================
# 16. SAVING & EXPORTING
# ============================================================

# --- Axes-level: use fig.savefig ---
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=tips, x="day", y="total_bill", palette="Set2", ax=ax)
fig.savefig("output.png", dpi=150, bbox_inches="tight")
fig.savefig("output.svg", bbox_inches="tight")
fig.savefig("output.pdf", bbox_inches="tight")
plt.close()

# --- Figure-level: use g.savefig ---
g = sns.displot(data=tips, x="total_bill", hue="sex", kind="kde", fill=True)
g.savefig("output_displot.png", dpi=150, bbox_inches="tight")
plt.close("all")

# --- Objects API: use fig.save ---
p = so.Plot(tips, x="total_bill", y="tip").add(so.Dots())
p.save("output_objects.png", bbox_inches="tight")
plt.close("all")

# Clean up all generated files
import os
for f in os.listdir("."):
    if f.startswith("fig_") or f.startswith("output") or f == "palette.png":
        os.remove(f)

print("All figures generated and cleaned up.")


# ============================================================
# QUICK REFERENCE CHEAT SHEET
# ============================================================
#
# SETUP
#   sns.set_theme(style=, palette=, font_scale=)  Global theme
#   sns.set_context("notebook")                   Scale for output
#   sns.set_palette("deep")                       Color palette
#   sns.despine(ax=)                              Remove spines
#
# RELATIONAL
#   sns.scatterplot(data, x, y, hue, size, style) Scatter
#   sns.lineplot(data, x, y, hue, errorbar)       Line + CI band
#   sns.relplot(..., kind="scatter"|"line")        Figure-level
#
# DISTRIBUTIONS
#   sns.histplot(data, x, bins, kde, hue)         Histogram
#   sns.kdeplot(data, x, hue, fill)               KDE curve
#   sns.ecdfplot(data, x, hue)                    CDF
#   sns.rugplot(data, x)                          Tick marks
#   sns.displot(..., kind="hist"|"kde"|"ecdf")     Figure-level
#
# CATEGORICAL
#   sns.stripplot / sns.swarmplot                 All points
#   sns.boxplot / sns.violinplot / sns.boxenplot  Summary stats
#   sns.barplot(errorbar="ci")                    Mean + CI
#   sns.countplot                                 Counts
#   sns.pointplot                                 Connected means
#   sns.catplot(..., kind=)                       Figure-level
#
# REGRESSION
#   sns.regplot(data, x, y, order)               Scatter + fit
#   sns.residplot(data, x, y)                    Residuals
#   sns.lmplot(..., col=, row=, hue=)            Figure-level
#
# MATRIX
#   sns.heatmap(df, annot, fmt, cmap, vmin, vmax) Heatmap
#   sns.clustermap(df, standard_scale, cmap)      Clustered heatmap
#
# MULTI-PLOT GRIDS
#   sns.pairplot(df, hue, diag_kind)             Scatter matrix
#   sns.PairGrid(df).map_upper/lower/diag        Custom pair grid
#   sns.FacetGrid(df, col, row, hue).map         Custom facet grid
#   sns.jointplot(data, x, y, kind)              Bivariate + marginals
#
# FIGURE-LEVEL NOTES
#   g = sns.relplot/displot/catplot/lmplot(...)  Returns FacetGrid
#   g.savefig("file.png")                        Save
#   g.set_axis_labels / g.set_titles             Label facets
#   g.axes[row, col]                             Access individual Axes
#
# AXES-LEVEL NOTES
#   All accept ax= parameter                     Embed in Matplotlib
#   Returns Matplotlib Axes                      Use fig.savefig()
#
# OBJECTS API (v0.12+)
#   so.Plot(data, x, y, color).add(so.Dots())   Build plot
#   .add(so.Line(), so.PolyFit())                Add layers
#   .label(...).layout(size=).save("file.png")   Configure + save
#
# DATA FORMAT
#   Prefer long/tidy format: one row per observation
#   df.melt(id_vars, var_name, value_name)       Wide -> long
