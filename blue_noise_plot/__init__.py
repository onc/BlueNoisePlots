# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from blue_noise_plot.lloyd_relax import blue_noise_single_class, blue_noise_multi_class, \
    apply_jitter


def __split_array_into_class_arrays(data, points_per_class):
    start_index = 0
    points_for_class = []
    for i, _ in enumerate(points_per_class):
        num_elements = points_per_class[i]
        end_index = start_index + num_elements
        subset = data[start_index:end_index]

        points_for_class.append(subset)
        start_index += num_elements

    return points_for_class


def __calculate_plot_width(multi_data, points_per_class=None):
    def single_class_width(data):
        pixel_width = 740
        marker_size = 10

        hist = np.histogram(data, bins=int(pixel_width / marker_size),
                            density=False)
        alpha = 1 / 35
        k_max = max(hist[0]) * alpha
        return k_max

    if not points_per_class:
        return single_class_width(multi_data)

    points_for_class = __split_array_into_class_arrays(multi_data, points_per_class)
    return [single_class_width(points) for points in points_for_class]


def __compute_fig_size(plot_width, orient, scaling):
    if orient == 'v':
        return (plot_width * scaling, scaling)
    return (scaling, plot_width * scaling)


def __prepare_data(x=None, hue=None, data=None):
    col = data[x]

    if not hue:
        return {
            'points_per_class': [col.shape[0]],
            'points': np.interp(col, (col.min(), col.max()), (0, +1))
        }

    points_per_class = []
    points = np.array([])
    for _, group in data.groupby(hue):
        points_per_class.append(group.shape[0])
        points = np.concatenate((points, group[x]))

    return {
        'points_per_class': points_per_class,
        'points': np.interp(points, (points.min(), points.max()), (0, +1))
    }


def __plot(x=None, hue=None, data=None, dodge=False, orient=None, color='black', palette='tab10',
           size=3, centralized=False, plot_width=None, filename='', scaling=10, method=''):
    """ Renders a plot from the given data.

    Args:
        x (str in data): Variables that specify positions on the data-encoding axes.
        hue (str in data): Optional. Grouping variable that will produce points with different
                           colors.
        data (pandas.DataFrame): Input data structure. Long-form collection of vectors that can be
                                 assigned to named variables.
        dodge (boolean): Optional. Wether to dodge the categorical classes of the plot.
                         Defaults to False.
        orient ("v" | "h"): Optional. Orientation of the plot (vertical or horizontal).
                            Defaults to 'v'.
        color (str): Color to use for markers, in case there is only one class (hue not given).
                     Defaults to 'black'.
        palette (str): Method for choosing the colors to use when mapping the hue semantic.
                       String values are passed to color_palette(). List or dict values imply
                       categorical mapping, while a colormap object implies numeric mapping.
                       Defaults to 'tab10'.
        size (float): The marker size in points**2.
        centralized (boolean): Optional. Where the plot should be centralized or not.
                               Defaults to False.
        filename (str): Filename of the plot.
        scaling (int): Optional. Scaling for the size of plot.
                       Defaults to 10 for a 740 pixel lot (long side).
        method (str): Type of the plot to draw. Either 'jitter' or 'blue_noise'.
    """
    dodge_margin = 0.1

    prepared_data = __prepare_data(x=x, hue=hue, data=data)
    if dodge:
        if not plot_width:
            single_class_plot_width = max(__calculate_plot_width(
                prepared_data['points'], prepared_data['points_per_class']))
        else:
            single_class_plot_width = plot_width
        # - dodge_margin in the end, because on margin after the last one
        plot_limits = (len(prepared_data['points_per_class']) *
                       (single_class_plot_width + dodge_margin)) - dodge_margin
    else:
        # check, if a width was given, or a automatic width should be calculated
        if not plot_width:
            single_class_plot_width = __calculate_plot_width(prepared_data['points'])
        else:
            single_class_plot_width = plot_width
        plot_limits = single_class_plot_width

    if method == 'blue_noise':
        if len(prepared_data['points_per_class']) <= 1:
            # single class BN
            spread_points = blue_noise_single_class(prepared_data['points'],
                                                    single_class_plot_width, centralized)
        else:
            # multi class BN
            if dodge:
                points_for_class = __split_array_into_class_arrays(prepared_data['points'],
                                                                   prepared_data['points_per_class'])

                single_class_plot_widths = __calculate_plot_width(prepared_data['points'],
                                                                  prepared_data['points_per_class'])
                spread_points = []
                for i, points in enumerate(points_for_class):
                    spread_points += blue_noise_single_class(points,
                                                             single_class_plot_widths[i],
                                                             centralized).tolist()
                spread_points = np.array(spread_points)
            else:
                spread_points = blue_noise_multi_class(prepared_data, single_class_plot_width,
                                                       centralized, max_iterations=100)
    else:
        spread_points = apply_jitter(prepared_data['points'], single_class_plot_width)

    # interpolate the points in their data dimension from 0-1 back to the orignal values
    min_x = min(data[x])
    max_x = max(data[x])
    spread_points[:, 1] = np.interp(spread_points[:, 1], (0, 1), (min_x, max_x))

    if orient == 'h':
        # If plot is horiontal, swap x and y
        spread_points[:, [0, 1]] = spread_points[:, [1, 0]]

    # Split spread points from a single array, back into sub-arrays
    points_for_class = __split_array_into_class_arrays(spread_points,
                                                       prepared_data['points_per_class'])

    if not dodge:
        # classes with more points get more prominent colors.
        points_for_class.sort(key=len, reverse=True)

    # Prepare figure
    my_dpi = 96
    padding = 0.02
    fig_size = __compute_fig_size(plot_limits, orient, scaling)

    if not filename:
        return points_for_class

    plt.figure(figsize=fig_size, dpi=my_dpi)

    plt.xticks([])
    plt.yticks([])
    plt.grid()
    if orient == 'v':
        plt.xlim(0.0 - padding, plot_limits + padding)
        plt.ylim(min_x - (max_x * padding), max_x + (max_x * padding))
    else:
        plt.xlim(min_x - (max_x * padding), max_x + (max_x * padding))
        plt.ylim(0.0 - padding, plot_limits + padding)


    # Draw
    for i, points in enumerate(points_for_class):
        if len(points_for_class) <= 1:
            plt.scatter(points[:, 0], points[:, 1],
                        color=color, s=size)
        else:
            if dodge:
                offset = i * (single_class_plot_width + dodge_margin)
                if orient == 'v':
                    points[:, 0] = points[:, 0] + offset
                else:
                    points[:, 1] = points[:, 1] + offset

            colors = matplotlib.cm.get_cmap(palette)
            plt.scatter(points[:, 0], points[:, 1], color=colors(i), s=size)

    # Save
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, transparent=True, dpi=my_dpi)

    plt.clf()
    plt.close()

    return points_for_class


def jitter(x=None, hue=None, data=None, dodge=False, orient='v', plot_width=None,
           color='black', palette='tab10', size=3, filename='', scaling=10):
    """ Renders a Jitter Plot from the given data.

    Args:
        x (str in data): Variables that specify positions on the data-encoding axes.
        hue (str in data): Optional. Grouping variable that will produce points with different
                           colors.
        data (pandas.DataFrame): Input data structure. Long-form collection of vectors that can be
                                 assigned to named variables.
        dodge (boolean): Optional. Wether to dodge the categorical classes of the plot.
                         Defaults to False.
        orient ("v" | "h"): Optional. Orientation of the plot (vertical or horizontal).
                            Defaults to 'v'.
        color (str): Color to use for markers, in case there is only one class (hue not given).
                     Defaults to 'black'.
        palette (str): Method for choosing the colors to use when mapping the hue semantic.
                       String values are passed to color_palette(). List or dict values imply
                       categorical mapping, while a colormap object implies numeric mapping.
                       Defaults to 'tab10'.
        size (float): The marker size in points**2.
        centralized (boolean): Optional. Where the plot should be centralized or not.
                               Defaults to False.
        filename (str): Filename of the plot.
        scaling (int): Optional. Scaling for the size of plot.
                       Defaults to 10 for a 740 pixel lot (long side).
    """
    return __plot(x=x, hue=hue, data=data, dodge=dodge, orient=orient, plot_width=plot_width,
                  color=color, palette=palette, size=size, filename=filename,
                  scaling=scaling, method='jitter')


def blue_noise(x=None, hue=None, data=None, dodge=False, orient='v', plot_width=None,
               color='black', palette='tab10', size=3, centralized=False,
               filename='', scaling=10):
    """ Renders a *Blue Noise Plot* from the given data.

    Args:
        x (str in data): Variables that specify positions on the data-encoding axes.
        hue (str in data): Optional. Grouping variable that will produce points with different
                           colors.
        data (pandas.DataFrame): Input data structure. Long-form collection of vectors that can be
                                 assigned to named variables.
        dodge (boolean): Optional. Wether to dodge the categorical classes of the plot.
                         Defaults to False.
        orient ("v" | "h"): Optional. Orientation of the plot (vertical or horizontal).
                            Defaults to 'v'.
        color (str): Color to use for markers, in case there is only one class (hue not given).
                     Defaults to 'black'.
        palette (str): Method for choosing the colors to use when mapping the hue semantic.
                       String values are passed to color_palette(). List or dict values imply
                       categorical mapping, while a colormap object implies numeric mapping.
                       Defaults to 'tab10'.
        size (float): The marker size in points**2.
        centralized (boolean): Optional. Where the plot should be centralized or not.
                               Defaults to False.
        filename (str): Filename of the plot.
        scaling (int): Optional. Scaling for the size of plot.
                       Defaults to 10 for a 740 pixel lot (long side).
    """
    return __plot(x=x, hue=hue, data=data, dodge=dodge, orient=orient, plot_width=plot_width,
                  color=color, palette=palette, size=size, centralized=centralized,
                  filename=filename, scaling=scaling, method='blue_noise')
