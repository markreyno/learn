# ============================================================
# MATPLOTLIB MAIN CONCEPTS
# ============================================================
# Matplotlib is Python's foundational plotting library.
# Most other visualization libraries (Seaborn, Pandas plots) build on it.
# Core idea: Figure (the canvas) contains Axes (individual plots).

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# 1. THE FIGURE / AXES ARCHITECTURE
# ============================================================
# Figure  — the top-level container (the whole window / image)
# Axes    — an individual plot area inside a Figure (has x/y axis, title, etc.)
# Axis    — the actual x-axis or y-axis object (ticks, labels)
#
# Two interfaces:
#   pyplot (implicit)  — plt.plot(), plt.title() — quick, stateful
#   OO (explicit)      — fig, ax = plt.subplots() — recommended for real code

# --- pyplot (implicit) ---
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("pyplot style")
plt.savefig("fig_pyplot.png")
plt.close()

# --- OO (explicit) --- recommended
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
ax.set_title("OO style")
fig.savefig("fig_oo.png")
plt.close()


# ============================================================
# 2. LINE PLOTS
# ============================================================

x = np.linspace(0, 2 * np.pi, 200)

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(x, np.sin(x), label="sin(x)", color="steelblue",
        linewidth=2, linestyle="-")
ax.plot(x, np.cos(x), label="cos(x)", color="tomato",
        linewidth=2, linestyle="--")
ax.plot(x, np.sin(2 * x), label="sin(2x)", color="seagreen",
        linewidth=1.5, linestyle=":", marker="o", markevery=20, markersize=5)

ax.set_title("Line Plot")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-1.5, 1.5)

fig.savefig("fig_line.png", dpi=150, bbox_inches="tight")
plt.close()

# Common linestyles: "-"  "--"  "-."  ":"
# Common markers:   "o"  "s"  "^"  "D"  "x"  "+"  "*"


# ============================================================
# 3. SCATTER PLOTS
# ============================================================

rng = np.random.default_rng(42)
n = 150
x = rng.normal(0, 1, n)
y = x * 0.8 + rng.normal(0, 0.5, n)
colors = rng.random(n)        # color-map values
sizes  = rng.uniform(20, 200, n)

fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(x, y, c=colors, s=sizes, cmap="viridis",
                alpha=0.7, edgecolors="white", linewidths=0.5)
fig.colorbar(sc, ax=ax, label="Random value")
ax.set_title("Scatter Plot")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.savefig("fig_scatter.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 4. BAR CHARTS
# ============================================================

categories = ["A", "B", "C", "D", "E"]
values     = [23, 45, 12, 67, 34]
errors     = [3, 5, 2, 6, 4]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Vertical bar
axes[0].bar(categories, values, yerr=errors, color="steelblue",
            edgecolor="white", capsize=5, alpha=0.85)
axes[0].set_title("Vertical Bar")

# Horizontal bar
axes[1].barh(categories, values, xerr=errors, color="tomato",
             edgecolor="white", capsize=5, alpha=0.85)
axes[1].set_title("Horizontal Bar")

for ax in axes:
    ax.grid(True, axis="both", linestyle="--", alpha=0.4)

fig.tight_layout()
fig.savefig("fig_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# Grouped bars
x_pos   = np.arange(len(categories))
width   = 0.35
group_a = [10, 20, 15, 30, 25]
group_b = [15, 25, 10, 20, 35]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x_pos - width/2, group_a, width, label="Group A", color="steelblue")
ax.bar(x_pos + width/2, group_b, width, label="Group B", color="tomato")
ax.set_xticks(x_pos)
ax.set_xticklabels(categories)
ax.legend()
ax.set_title("Grouped Bar Chart")
fig.savefig("fig_grouped_bar.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 5. HISTOGRAMS
# ============================================================

data1 = rng.normal(0, 1, 1000)
data2 = rng.normal(3, 1.5, 1000)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Basic histogram
axes[0].hist(data1, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
axes[0].set_title("Histogram")
axes[0].set_xlabel("Value")
axes[0].set_ylabel("Count")

# Overlapping histograms with density
axes[1].hist(data1, bins=30, density=True, alpha=0.6, label="Group 1", color="steelblue")
axes[1].hist(data2, bins=30, density=True, alpha=0.6, label="Group 2", color="tomato")
axes[1].set_title("Overlapping (density=True)")
axes[1].legend()

fig.tight_layout()
fig.savefig("fig_hist.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 6. BOX PLOTS & VIOLIN PLOTS
# ============================================================

groups = [rng.normal(loc, 1, 100) for loc in [0, 1, 2, 3]]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Box plot
bp = axes[0].boxplot(groups, labels=["A", "B", "C", "D"],
                     patch_artist=True, notch=False)
colors_box = ["steelblue", "tomato", "seagreen", "gold"]
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color)
axes[0].set_title("Box Plot")
axes[0].set_ylabel("Value")
axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)

# Violin plot
vp = axes[1].violinplot(groups, positions=[1, 2, 3, 4],
                         showmeans=True, showmedians=True)
axes[1].set_xticks([1, 2, 3, 4])
axes[1].set_xticklabels(["A", "B", "C", "D"])
axes[1].set_title("Violin Plot")
axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)

fig.tight_layout()
fig.savefig("fig_box_violin.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 7. PIE CHARTS
# ============================================================

labels  = ["Python", "JavaScript", "Java", "C++", "Other"]
sizes   = [35, 25, 20, 12, 8]
explode = [0.05, 0, 0, 0, 0]   # pull out Python slice

fig, ax = plt.subplots(figsize=(6, 6))
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, explode=explode,
    autopct="%1.1f%%", startangle=90,
    colors=["steelblue", "tomato", "seagreen", "gold", "plum"],
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
for at in autotexts:
    at.set_fontsize(9)
ax.set_title("Language Popularity")
fig.savefig("fig_pie.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 8. HEATMAPS & IMSHOW
# ============================================================

# Correlation-style heatmap
matrix = rng.uniform(-1, 1, (6, 6))
np.fill_diagonal(matrix, 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# imshow heatmap with annotations
im = axes[0].imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
fig.colorbar(im, ax=axes[0])
for i in range(6):
    for j in range(6):
        axes[0].text(j, i, f"{matrix[i, j]:.2f}", ha="center",
                     va="center", fontsize=7)
axes[0].set_title("Heatmap (imshow)")

# 2D image / grayscale
img = rng.random((28, 28))
axes[1].imshow(img, cmap="gray")
axes[1].axis("off")
axes[1].set_title("Grayscale Image")

fig.tight_layout()
fig.savefig("fig_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 9. SUBPLOTS & LAYOUTS
# ============================================================

# Basic grid
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.plot(np.random.randn(50).cumsum())
    ax.set_title(f"Plot {i+1}")
fig.suptitle("2×3 Grid", fontsize=14)
fig.tight_layout()
fig.savefig("fig_subplots.png", dpi=150, bbox_inches="tight")
plt.close()

# Shared axes
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
x = np.linspace(0, 10, 200)
axes[0].plot(x, np.sin(x), color="steelblue")
axes[0].set_ylabel("sin(x)")
axes[1].plot(x, np.cos(x), color="tomato")
axes[1].set_ylabel("cos(x)")
axes[1].set_xlabel("x")
fig.suptitle("Shared X Axis")
fig.tight_layout()
fig.savefig("fig_shared.png", dpi=150, bbox_inches="tight")
plt.close()

# subplot_mosaic — named layouts (Python 3.8+)
fig, axd = plt.subplot_mosaic(
    [["left", "right_top"],
     ["left", "right_bottom"]],
    figsize=(10, 5),
)
axd["left"].set_title("left (spans rows)")
axd["right_top"].set_title("right_top")
axd["right_bottom"].set_title("right_bottom")
fig.tight_layout()
fig.savefig("fig_mosaic.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 10. ANNOTATIONS & TEXT
# ============================================================

x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(x)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y)

# Simple text
ax.text(1.0, 0.9, "sin(x)", fontsize=12, color="steelblue",
        ha="center", va="bottom")

# Annotate with arrow
peak_x = np.pi / 2
ax.annotate(
    "Maximum",
    xy=(peak_x, 1.0),              # point to annotate
    xytext=(peak_x + 1, 0.7),      # text position
    arrowprops=dict(arrowstyle="->", color="black"),
    fontsize=10, color="darkred",
)

# Horizontal / vertical lines
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.axvline(np.pi, color="gray", linewidth=0.8, linestyle="--")

# Shaded region
ax.axvspan(0, np.pi, alpha=0.1, color="steelblue", label="Positive half")

ax.set_title("Annotations & Reference Lines")
ax.legend()
fig.savefig("fig_annotations.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 11. STYLING & THEMES
# ============================================================

print("Available styles:", plt.style.available[:8], "...")

# Apply a style
with plt.style.context("seaborn-v0_8-whitegrid"):
    fig, ax = plt.subplots()
    ax.plot(np.random.randn(50).cumsum())
    ax.set_title("seaborn-v0_8-whitegrid style")
    fig.savefig("fig_style.png", dpi=150, bbox_inches="tight")
    plt.close()

# Common styles:
# "seaborn-v0_8-whitegrid"  "seaborn-v0_8-darkgrid"  "ggplot"
# "fivethirtyeight"  "bmh"  "dark_background"  "grayscale"

# Manual RC params
plt.rcParams.update({
    "figure.figsize":   (8, 5),
    "font.size":        12,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.linestyle":   "--",
    "grid.alpha":       0.4,
})

# Reset to defaults
plt.rcParams.update(plt.rcdefaults())


# ============================================================
# 12. COLORMAPS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

data = np.random.randn(20, 20)

# Sequential
axes[0].imshow(data, cmap="viridis")
axes[0].set_title("viridis (sequential)")

# Diverging (good for data centered on zero)
axes[1].imshow(data, cmap="RdBu_r", vmin=-3, vmax=3)
axes[1].set_title("RdBu_r (diverging)")

# Qualitative (for discrete categories)
axes[2].imshow(np.random.randint(0, 8, (10, 10)), cmap="Set1")
axes[2].set_title("Set1 (qualitative)")

# Sequential:   viridis, plasma, inferno, magma, cividis, Blues, Greens
# Diverging:    RdBu, coolwarm, seismic, bwr, PiYG
# Qualitative:  Set1, Set2, tab10, tab20, Paired, Pastel1

fig.tight_layout()
fig.savefig("fig_colormaps.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 13. 3D PLOTS
# ============================================================

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (side-effect import)

fig = plt.figure(figsize=(14, 4))

# 3D line
ax1 = fig.add_subplot(131, projection="3d")
t = np.linspace(0, 4 * np.pi, 200)
ax1.plot(np.sin(t), np.cos(t), t / (4 * np.pi), color="steelblue")
ax1.set_title("3D Line")

# 3D scatter
ax2 = fig.add_subplot(132, projection="3d")
x3, y3, z3 = rng.normal(0, 1, (3, 100))
ax2.scatter(x3, y3, z3, c=z3, cmap="viridis", s=20)
ax2.set_title("3D Scatter")

# Surface plot
ax3 = fig.add_subplot(133, projection="3d")
gx = np.linspace(-3, 3, 50)
gy = np.linspace(-3, 3, 50)
GX, GY = np.meshgrid(gx, gy)
GZ = np.sin(np.sqrt(GX**2 + GY**2))
ax3.plot_surface(GX, GY, GZ, cmap="plasma", alpha=0.85)
ax3.set_title("Surface")

fig.tight_layout()
fig.savefig("fig_3d.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 14. TWIN AXES (DUAL Y-AXIS)
# ============================================================

x = np.arange(12)
revenue = rng.uniform(50, 150, 12)
growth  = rng.uniform(-5, 20, 12)

fig, ax1 = plt.subplots(figsize=(9, 4))

color1 = "steelblue"
ax1.bar(x, revenue, color=color1, alpha=0.7, label="Revenue ($k)")
ax1.set_xlabel("Month")
ax1.set_ylabel("Revenue ($k)", color=color1)
ax1.tick_params(axis="y", labelcolor=color1)

ax2 = ax1.twinx()                # share the x-axis
color2 = "tomato"
ax2.plot(x, growth, color=color2, marker="o", linewidth=2, label="Growth (%)")
ax2.set_ylabel("Growth (%)", color=color2)
ax2.tick_params(axis="y", labelcolor=color2)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
ax1.set_title("Dual Y-Axis: Revenue vs Growth")
fig.savefig("fig_twinx.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 15. FILL & AREA PLOTS
# ============================================================

x = np.linspace(0, 2 * np.pi, 200)
y1 = np.sin(x)
y2 = np.sin(2 * x)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# fill_between — shade between two curves
axes[0].plot(x, y1, label="sin(x)", color="steelblue")
axes[0].plot(x, y2, label="sin(2x)", color="tomato")
axes[0].fill_between(x, y1, y2, where=(y1 > y2), alpha=0.3,
                     color="steelblue", label="sin(x) > sin(2x)")
axes[0].fill_between(x, y1, y2, where=(y1 < y2), alpha=0.3,
                     color="tomato", label="sin(2x) > sin(x)")
axes[0].legend(fontsize=8)
axes[0].set_title("fill_between")

# Stacked area chart
x_s  = np.arange(10)
y_a  = rng.uniform(1, 5, 10)
y_b  = rng.uniform(1, 5, 10)
y_c  = rng.uniform(1, 5, 10)
axes[1].stackplot(x_s, y_a, y_b, y_c, labels=["A", "B", "C"],
                  colors=["steelblue", "tomato", "seagreen"], alpha=0.8)
axes[1].legend(loc="upper left")
axes[1].set_title("Stacked Area")

fig.tight_layout()
fig.savefig("fig_fill.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 16. SAVING FIGURES
# ============================================================

fig, ax = plt.subplots()
ax.plot([1, 2, 3])
ax.set_title("Saving Options")

fig.savefig("output.png",  dpi=150,  bbox_inches="tight")   # raster
fig.savefig("output.svg",  bbox_inches="tight")              # vector (scalable)
fig.savefig("output.pdf",  bbox_inches="tight")              # publication quality

# Key savefig params:
#   dpi          — resolution (72 screen, 150 presentation, 300 print)
#   bbox_inches  — "tight" removes extra whitespace
#   transparent  — True for transparent background (PNG/SVG)
#   facecolor    — background color override

plt.close("all")

# Clean up demo files
import os
for f in os.listdir("."):
    if f.startswith("fig_") or f in ("output.png", "output.svg", "output.pdf"):
        os.remove(f)

print("All figures generated and cleaned up.")


# ============================================================
# QUICK REFERENCE CHEAT SHEET
# ============================================================
#
# SETUP
#   fig, ax = plt.subplots(figsize=(w, h))       Single plot
#   fig, axes = plt.subplots(rows, cols)          Grid of plots
#   fig, axd = plt.subplot_mosaic([[...]])        Named layout
#
# PLOT TYPES
#   ax.plot(x, y)                                 Line plot
#   ax.scatter(x, y, c=, s=, cmap=)              Scatter
#   ax.bar(x, height) / ax.barh(y, width)        Bar chart
#   ax.hist(data, bins=, density=)               Histogram
#   ax.boxplot(data) / ax.violinplot(data)       Distribution
#   ax.pie(sizes, labels=, autopct=)             Pie chart
#   ax.imshow(matrix, cmap=)                     Heatmap/image
#   ax.fill_between(x, y1, y2)                  Shaded area
#   ax.stackplot(x, y1, y2, ...)                 Stacked area
#
# LABELS & LAYOUT
#   ax.set_title("...") / fig.suptitle("...")    Title
#   ax.set_xlabel("x") / ax.set_ylabel("y")     Axis labels
#   ax.legend() / ax.legend(loc="upper left")   Legend
#   ax.grid(True, linestyle="--", alpha=0.5)    Grid
#   ax.set_xlim(a, b) / ax.set_ylim(a, b)       Axis limits
#   ax.set_xticks([...]) / ax.set_xticklabels() Tick control
#   fig.tight_layout()                           Fix overlap
#
# STYLING
#   color=  linewidth=  linestyle=  marker=  alpha=  label=
#   plt.style.use("seaborn-v0_8-whitegrid")     Apply theme
#   plt.rcParams.update({...})                  Global defaults
#
# ANNOTATIONS
#   ax.text(x, y, "text")                       Plain text
#   ax.annotate("text", xy=, xytext=,           Arrow annotation
#               arrowprops=dict(arrowstyle="->"))
#   ax.axhline(y) / ax.axvline(x)              Reference lines
#   ax.axhspan / ax.axvspan                     Shaded bands
#
# COLORMAPS
#   Sequential:   viridis, plasma, Blues, Greens
#   Diverging:    RdBu_r, coolwarm, seismic
#   Qualitative:  Set1, tab10, tab20, Paired
#   fig.colorbar(im, ax=ax, label="...")        Add colorbar
#
# DUAL AXIS
#   ax2 = ax1.twinx()                           Shared x-axis
#   ax2 = ax1.twiny()                           Shared y-axis
#
# 3D
#   fig.add_subplot(111, projection="3d")       3D axes
#   ax.plot / scatter / plot_surface            3D plot types
#
# SAVING
#   fig.savefig("file.png", dpi=150, bbox_inches="tight")
#   fig.savefig("file.svg")  / fig.savefig("file.pdf")
#   plt.close() / plt.close("all")             Free memory
