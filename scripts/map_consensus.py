import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import cm


def extract_data(filepath):
    """
    Extracts the emission data from a text file.

    Args:
        filepath: Path to the data file.

    Returns:
        A list of lists containing the numerical data, or None if an error occurs.
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

    # Regular expression to extract data rows.  Matches strings like "[0.51, 1.00, 1.00, ...]"
    # Explanation:
    #   \[         Matches the opening square bracket.
    #   (?: ... )  Non-capturing group.
    #   [0-9.]+   Matches one or more digits or decimal points.
    #   ,\s*      Matches a comma followed by zero or more whitespace characters.
    #   +         Matches the previous group (number and comma) one or more times.
    #   [0-9.]+   Matches the last number in the list.
    #   \]         Matches the closing square bracket.
    
    list_pattern = r'\[(?:[0-9.]+,\s*)+[0-9.]+\]'  # Regular expression to match data rows
    matches = re.findall(list_pattern, content)

    if not matches:
        print("Error: No matching data found in the file.")
        return None

    data = []
    for match in matches:
        try:
            # Extract numerical values from the matched string.
            # 1. match[1:-1]: Removes the square brackets from the beginning and end.
            # 2. .split(','): Splits the string into a list of strings at each comma.
            # 3. [float(x.strip()) for x in ...]: Converts each string to a float 
            #       after removing leading/trailing whitespace.
            
            row = [float(x.strip()) for x in match[1:-1].split(',')]
            data.append(row)
        except ValueError:
            print(f"Warning: Skipping invalid data row: {match}")

    return data


def visualize_data(emission_data, output_filename="consensus_plot.svg"):
    """
    Generates and saves a contour plot of the retention map.

    Args:
        emission_data: The extracted emission data.
        output_filename: The name of the output SVG file.
    """
    major_ratios = {}
    avg_weight_devs = {}

    # Process the data to organize it by major stake
    for major_stake, major_weight, minor_weight, avg_weight_dev, major_ratio in emission_data:
        major_stake_str = f'{major_stake:.2f}'
        maj_idx, min_idx = int(round(50 * major_weight)), int(round(50 * minor_weight))

        avg_weight_devs.setdefault(major_stake_str, np.zeros((51, 51)))
        avg_weight_devs[major_stake_str][maj_idx][min_idx] = avg_weight_dev

        major_ratios.setdefault(major_stake_str, np.zeros((51, 51)))
        major_ratios[major_stake_str][maj_idx][min_idx] = major_ratio


    # Create the meshgrid for the contour plot
    x = np.linspace(0, 1, 51)
    y = np.linspace(0, 1, 51)
    x, y = np.meshgrid(x, y, indexing='ij')

    # Set up the plot
    fig = plt.figure(figsize=(6, 6), dpi=70)
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 1, 0.05))
    ax.set_yticks(np.arange(0, 1., 0.05))
    ax.set_xticklabels([f'{_:.2f}'[1:] for _ in np.arange(0, 1., 0.05)])
    plt.grid(linestyle="dotted", color=[0.85, 0.85, 0.85])


    # Define stakes and colors for contour lines
    isolate = ['0.60']  # Stakes to highlight
    stakes = [0.51, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    colors = cm.viridis(np.linspace(0, 1, len(stakes) + 1))

    # Create contour lines for each stake
    for i, stake in enumerate(stakes):
        contours = plt.contour(x, y, major_ratios[f'{stake:.2f}'], levels=[0., stake], colors=[colors[i + 1]])
        if f'{stake:.2f}' in isolate:
            contours.collections[1].set_linewidth(3) # Highlight isolated stake
        plt.clabel(contours, inline=True, fontsize=10)

    # points_to_plot = [(0.78, 1, '1')]
    # to_highlight = 0
    # hyperparam = 0.0

    # points_to_plot = [(0.77, 1, '2')]
    # to_highlight = 0
    # hyperparam = 0.25

    points_to_plot = [(0.78, 1, '1'), (0.77, 1, '2'), (0.73, 0.8, '3')]
    to_highlight = 2
    hyperparam = 0.5
    
    annotation_text = [
        f"60% stake + {100*points_to_plot[to_highlight][0]:.0f}% utility",  # Bold by default (first line)
        "receives 60% emission",
        "retains 60% stake",
        f"[optimal cabal weight = {100*points_to_plot[to_highlight][1]:.0f}%]",
        "",  # Blank line
        f"bond consensus level = {100*hyperparam:.0f}%"
    ]
    for i, line in enumerate(annotation_text):
        ax.text(0.1, 0.9 - 0.04 * i, line, 
                fontweight='bold' if i == 0 else 'normal', # Conditional font weight
                ha='left', va='center', color='black', alpha=0.8, fontsize=12)
    
    # rotated_text_x = 0.4
    # rotated_text_y = 0.24
    # ax.text(rotated_text_x, rotated_text_y, "stake=emission=60%", ha='center', va='center', 
    #         rotation=41, color='#38588C', alpha=0.8, fontsize=12)

    # Vertical lines, dots, circles, and numbers
    for i, (x, y, nr) in enumerate(points_to_plot):
        alpha = 0.75 if i == to_highlight else 0.35  # Main line has higher alpha
        ax.axvline(x, linestyle='--', color='red', alpha=alpha)
        if i == to_highlight:
            ax.plot(x, y, 'ro', markersize=8, alpha=alpha)
        
        circle = patches.Circle((x, 0.502 - 0.1 * i), radius=0.027, facecolor='black', 
                                edgecolor='white', alpha=alpha, linewidth=2, zorder=100)  # Consistent alpha
        ax.add_patch(circle)

        font_props = {'family': 'monospace', 'weight': 'bold', 'size': 12,
                       'ha': 'center', 'va': 'center', 'color': 'white', 'zorder': 101}
        ax.text(x,  0.5 - 0.1 * i, nr, **font_props)  # Numbered labels
    
    # Add title and labels
    plt.title(f'Major emission [$S_{{maj}}=E_{{maj}}$] [$\\lambda={hyperparam:.2f}$]')
    plt.ylabel('Minor self-weight')
    plt.xlabel('Major self-weight')

    # Save the plot
    plt.savefig(output_filename, format='svg', bbox_inches='tight')
    print(f"Plot saved to {output_filename}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/map_consensus.py <consensus.txt> [optional_output_filename]")
        sys.exit(1)

    filepath = sys.argv[1]
    output_filename = "consensus_plot.svg"  # Default output filename
    if len(sys.argv) >= 3:
        output_filename = sys.argv[2]  # Optional output filename

    extracted_data = extract_data(filepath)
    if extracted_data:
        visualize_data(extracted_data, output_filename)
