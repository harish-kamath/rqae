from matplotlib.colors import LinearSegmentedColormap, to_rgb
import matplotlib.pyplot as plt
import seaborn as sns

# Custom colors to match blog theme
background_color = "#f2f2f2"  # Light gray background
grid_color = "#E0E0E0"  # Subtle gray for grid lines
text_color = "#030303"  # Dark gray for text
accent_color = "#d5a02c"  # Gold/yellow accent for lines or highlights

# Set seaborn theme
sns.set_theme(style="whitegrid")

# Update matplotlib rcParams to match the blog's theme
plt.rcParams.update(
    {
        # Background and grid
        "figure.facecolor": background_color,
        "axes.facecolor": background_color,
        "axes.edgecolor": grid_color,
        "axes.grid": True,
        "grid.color": grid_color,
        "grid.alpha": 0.4,  # Make the grid lines light and subtle
        # Line properties
        "lines.linewidth": 3,
        "lines.color": text_color,  # Main line color to be dark gray for contrast
        # Font and text properties
        "font.size": 12,
        "axes.labelsize": 24,
        "axes.titlesize": 32,
        "axes.labelcolor": text_color,
        "axes.titleweight": "bold",  # Bold for titles to match your blog's heading style
        "xtick.color": text_color,
        "xtick.labelsize": 20,
        "ytick.color": text_color,
        "ytick.labelsize": 20,
        # Font family for a clean look
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        # Legend
        "legend.frameon": False,
        "legend.fontsize": 12,
        # Accent colors for highlights (e.g., lines, borders)
        "axes.prop_cycle": plt.cycler(
            color=[text_color, accent_color, "#66CC99", "#FF6666"]
        ),
        # Figure size
        "figure.figsize": (12.8, 9.6),
        # Increase spacing between axis and labels
        "axes.labelpad": 20,  # Increase padding between axis and label
    }
)

# Optional: Create a custom color palette with the accent color
custom_palette = sns.color_palette([text_color, accent_color, "#66CC99", "#FF6666"])
sns.set_palette(custom_palette)


def create_blog_colormaps():
    """
    Creates custom colormaps matching the blog theme colors.
    Returns a dictionary of colormaps that can be used in matplotlib.
    """
    # Define the colors from your theme
    text_color = "#030303"  # Dark gray
    accent_color = "#d5a02c"  # Gold/yellow
    tertiary_1 = "#66CC99"  # Green
    tertiary_2 = "#FF6666"  # Red
    background_color = "#f2f2f2"  # Light gray
    grid_color = "#E0E0E0"  # Medium gray

    # Create different colormaps for different use cases

    # 1. Primary-Accent colormap (good for sequential data)
    primary_accent = LinearSegmentedColormap.from_list(
        "primary_accent", [text_color, accent_color], N=256
    )

    # 2. Full palette colormap (good for categorical data)
    full_palette = LinearSegmentedColormap.from_list(
        "full_palette", [text_color, accent_color, tertiary_1, tertiary_2], N=256
    )

    # 3. Monochrome colormap (good for grayscale/neutral visualizations)
    monochrome = LinearSegmentedColormap.from_list(
        "monochrome", [background_color, grid_color, text_color], N=256
    )

    # 4. Accent gradient (good for highlighting)
    # Create varying opacities of the accent color
    accent_rgb = to_rgb(accent_color)
    accent_gradient = LinearSegmentedColormap.from_list(
        "accent_gradient",
        [
            (accent_rgb[0], accent_rgb[1], accent_rgb[2], a)
            for a in [0.2, 0.4, 0.6, 0.8, 1.0]
        ],
        N=256,
    )

    return {
        "primary_accent": primary_accent,
        "full_palette": full_palette,
        "monochrome": monochrome,
        "accent_gradient": accent_gradient,
    }
