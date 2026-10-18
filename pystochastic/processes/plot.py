import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
import pystochastic

# Plotly colorscale used for the paths and the histogram, restricted to avoid the too light colors
_SCALE = sample_colorscale("RdPu", list(np.linspace(0.3, 0.95, 8)))
_COLORSCALE = [[k / (len(_SCALE) - 1), c] for k, c in enumerate(_SCALE)]
_EXP_COLOR = "#6A66A3"
_MEAN_COLOR = "#542E71"


def colors(colorscale = "RdPu", expectation=None, mean=None):
    global _SCALE, _COLORSCALE, _EXP_COLOR, _MEAN_COLOR

    _SCALE = sample_colorscale(colorscale, list(np.linspace(0.3, 0.95, 8)))
    _COLORSCALE = [[k / (len(_SCALE) - 1), c] for k, c in enumerate(_SCALE)]

    hex_char = ["1","2","3","4","5","6","7","8","9","0","a","b","c","d","e","f","A","B","C","D","E","F"]

    if expectation is not None:

        if len(expectation) != 6:
            raise ValueError("The color must be a hex code of length 6")

        for chr in expectation:
            if chr not in hex_char:
                raise ValueError("The color must be a hex code of length 6")

        _EXP_COLOR = "#" + expectation

    if mean is not None:

        if len(mean) != 6:
            raise ValueError("The color must be a hex code of length 6")

        for chr in mean:
            if chr not in hex_char:
                raise ValueError("The color must be a hex code of length 6")

        _MEAN_COLOR = "#" + mean

def _path_colors(values):

    """
    Path Colors function

    Returns one color of the colorscale per path, ranked by the given values
    (the higher the value, the darker the color).

    Parameters
    ----------
    values : np.ndarray
        Values used to rank the paths, e.g. the final values of the paths.

    Returns
    -------
    list of str
        Colors of the paths.
    """

    values = np.asarray(values, dtype=float)
    n = values.size
    positions = np.full(n, 0.5) if n <= 1 else np.argsort(np.argsort(values)) / (n - 1)
    return sample_colorscale(_COLORSCALE, list(positions))

def _make_histogram(process, t, n_bins=30):

    """
    Make Histogram function

    Returns the empirical histogram of paths values pool at time t, with a fixed
    number of histogram bins.

    Parameters
    ----------
    process : pystochastic.processes.Process
        Process to be simulated.
    t : float
        Time at which the histogram is computed.
    n_bins : int, optional
        Number of histogram bins, by default 30.

    Returns
    -------
    np.ndarray
        Returns the empirical histogram of paths values pool at time t.
    """

    t_index = np.argmin(np.abs(process.t - t))

    data = process.path[:, t_index, :]
    return np.histogram(data, bins=n_bins, density=True)

def _compute_density(process, t, points=1000):

    """
    Compute Density function

    Returns the computed density of the process at time t on several values x,
    depending of the minimum and maximum values of the paths values pool at time t.

    Parameters
    ----------
    process : pystochastic.processes.Process
        Process to be simulated.
    t : float
        Time at which the histogram is computed.
    points : int, optional
        Number of evaluation points in the interval [min(pool), max(pool)], by default 1000.

    Returns
    -------
    tuple (np.ndarray, np.ndarray)
        Returns the choosen evalutation points and the corresponding density values.
    """

    t_index = np.argmin(np.abs(process.t - t))
    max_path = process.path.max()
    min_path = process.path.min()
    data = process.path[:, t_index, :]
    data_space = np.linspace(min_path, max_path, points)

    # We check if the density is available for the process
    try:
        # If the density is available for the effective process, we compute the values all over the data space.
        with np.errstate(all="ignore"):
            return data_space, np.asarray(process.density(process.t[t_index], data_space), dtype=float).reshape(points)

    # If the density is not available for the effective process, we return None.
    except (AttributeError, NotImplementedError, ValueError, TypeError,
            ZeroDivisionError, FloatingPointError):
        return None

def _distribution_traces(process, t, n_bins=30, points=1000, density_available=True):

    """
    Distribution Traces function

    Returns the traces of the theoretical density (if available) and of the empirical histogram
    of the paths values pool at time t, along with the maximal density value reached.

    Parameters
    ----------
    process : pystochastic.processes.Process
        Process to be simulated.
    t : float
        Time at which the distribution is computed.
    n_bins : int, optional
        Number of bins of the histogram, by default 30.
    points : int, optional
        Number of evaluation points in the interval [min(pool), max(pool)], by default 1000.
    density_available : bool, optional
        Specify if the theoretical density trace should be returned, by default True.

    Returns
    -------
    tuple (list, float)
        Returns the traces and the maximal density value.
    """

    traces = []
    x_max = 0.

    # If available, we compute the theoretical density
    if density_available:
        density = _compute_density(process, t, points)
        data_space, values = density if density is not None else (np.array([]), np.array([]))
        values = np.where(np.isfinite(values), values, np.nan)
        if np.any(np.isfinite(values)):
            x_max = np.nanmax(values)
        traces.append(go.Scatter(x=values, y=data_space, mode="lines", name="Theoretical density"))

    # We compute the empirical histogram of the paths values pool and the centers of each bin
    hist, bin_edges = _make_histogram(process, t, n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    x_max = max(x_max, hist.max())

    # Each bar of the histogram is colored according to its height
    traces.append(go.Bar(x=hist, y=bin_centers, width=np.diff(bin_edges), orientation="h",
                         marker=dict(color=hist, colorscale=_COLORSCALE, showscale=False,
                                     line=dict(width=0)),
                         name="Empirical density", opacity=0.8))

    return traces, x_max

def _plot_1d(process, n_bins=30, points=1000, slider_steps=100):

    """
    Plot 1D function

    Plot the simulated process paths and the empirical histogram of the paths values pool for 1D process.

    Parameters
    ----------
    process : pystochastic.processes.Process
        Process to be simulated.
    n_bins : int, optional
        Number of bins of the histogram, by default 30.
    points : int, optional
        Number of evaluation points in the interval [min(pool), max(pool)], by default 1000.
    slider_steps : int, optional
        Maximal number of times selectable with the slider, by default 100.
    """

    if process.path is None:
        raise ValueError("The path has not been simulated yet.")

    t = process.t[-1]
    t_nb = process.t.size
    ##################################################################################################################

    # The right panel shares the y axis with the paths, so the distribution is drawn horizontally
    process_figure = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.75, 0.25],
                                   horizontal_spacing=0.02)

    # We display the paths for each simulation, colored according to their final value
    colors = _path_colors(process.path[:, -1, 0])

    for i in range(process.n_simulations):
        process_figure.add_trace(go.Scatter(x=process.t, y=process.path[i, :, 0], mode="lines",
                                            line=dict(color=colors[i]),
                                            name=f"Path {i + 1}", showlegend=False), row=1, col=1)

    ##################################################################################################################

    # Computation and display of the mean path

    mean_path = np.mean(process.path, axis=0)
    process_figure.add_trace(
        go.Scatter(x=process.t, y=mean_path[:, 0], mode="lines", line=dict(width=3, color=_MEAN_COLOR),
                   name="Mean path"), row=1, col=1)

    # Computation of the expectation path
    try:
        # If the expectation is available for the effective process, we compute the values all over the data space.
        with np.errstate(all="ignore"):
            exp_path = np.asarray([process.expectation(i) for i in process.t],dtype=float).reshape(t_nb)
            process_figure.add_trace(
                go.Scatter(x=process.t, y=exp_path, mode="lines", line=dict(width=3, color=_EXP_COLOR, dash="dash"),
                           name=r"$\mathbb{E}[X_t]$"), row=1, col=1)

    # If the density is not available for the effective process, we return None.
    except (AttributeError, NotImplementedError, ValueError, TypeError,
            ZeroDivisionError, FloatingPointError):
        print("No expectation available for the effective process")


    ##################################################################################################################

    # Vertical cursor showing on the paths the time at which the distribution is displayed

    y_min, y_max = process.path[:, :, 0].min(), process.path[:, :, 0].max()

    def cursor_trace(t_cursor):
        return go.Scatter(x=[t_cursor, t_cursor], y=[y_min, y_max], mode="lines",
                          line=dict(width=1.5, dash="dot", color="#6b7280"),
                          hoverinfo="skip", showlegend=False, name="Time cursor")

    process_figure.add_trace(cursor_trace(t), row=1, col=1)
    cursor_index = len(process_figure.data) - 1

    ##################################################################################################################

    # We display the theoretical density (if available) and the empirical histogram at the final time

    density_available = _compute_density(process, t, points) is not None

    traces, x_max = _distribution_traces(process, t, n_bins, points, density_available)
    for trace in traces:
        process_figure.add_trace(trace, row=1, col=2)
    distribution_indices = list(range(cursor_index + 1, len(process_figure.data)))

    ##################################################################################################################

    # Slider choosing the time at which the distribution is displayed : one frame per selectable time

    time_indices = np.unique(np.linspace(0, process.t.size - 1, min(process.t.size, slider_steps)).astype(int))

    frames, steps = [], []
    for k in time_indices:
        t_k = process.t[k]
        traces, x_max_k = _distribution_traces(process, t_k, n_bins, points, density_available)
        name = f"{t_k:.4g}"
        frames.append(go.Frame(name=name, data=[cursor_trace(t_k)] + traces,
                               traces=[cursor_index] + distribution_indices,
                               layout=dict(xaxis2=dict(range=[0, 1.1 * x_max_k]))))
        steps.append(dict(method="animate", label=name,
                          args=[[name], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                             transition=dict(duration=0))]))

    process_figure.frames = frames

    # The slider spans the paths plot only. Plotly insets the first and last steps by 10px from the slider
    # edges, so negative paddings of 10px align the slider circle with the time cursor.
    paths_domain = process_figure.layout.xaxis.domain
    process_figure.update_layout(sliders=[dict(active=len(steps) - 1, steps=steps,
                                               x=paths_domain[0], len=paths_domain[1] - paths_domain[0],
                                               xanchor="left", pad=dict(t=50, l=-10, r=-10),
                                               currentvalue=dict(prefix="t = "))])

    # The legend is moved to the free space under the distribution plot
    process_figure.update_layout(legend=dict(x=process_figure.layout.xaxis2.domain[0], xanchor="left",
                                             y=-0.15, yanchor="top"))

    ##################################################################################################################

    # We add several details to the plot

    process_figure.update_layout(bargap=0)
    process_figure.update_xaxes(range=[0, 1.1 * x_max], row=1, col=2)

    process_figure.update_xaxes(title_text="t", range=[process.t[0], process.t[-1]], row=1, col=1)
    process_figure.update_xaxes(title_text="Density", row=1, col=2)
    process_figure.update_yaxes(title_text=process.name, row=1, col=1)
    process_figure.update_annotations(font=dict(size=13))

    process_figure.update_layout(title=f"Simulation : {process.name}")

    ##################################################################################################################

    process_figure.show()