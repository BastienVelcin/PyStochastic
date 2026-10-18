import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

# Plot style constants.
_FONT = dict(
    family="Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
    size=13,
    color="#1f2937",
)
_ACCENT = "#3b82f6"
_SCALE = sample_colorscale("RdPu", list(np.linspace(0.3, 0.95, 8)))
_COLORSCALE = [[k / (len(_SCALE) - 1), c] for k, c in enumerate(_SCALE)]
_DENSITY = "#ff4b7d"
_MEAN = "#111827"
_MUTED = "#6b7280"
_GRID = "#eef0f3"
_MARGIN_L, _MARGIN_R = 60, 30
_GRIP_INSET = 10 # Slider circle diameter

__all__ = ["plot_graph"]

def _histogram_frame(values, n_bins):

    """
    Returns a empirical probability distribution histogram.
    Returns (centers, densities, width).
    """

    lo, hi = float(np.min(values)), float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-12:
        return np.array([]), np.array([]), None

    dens, edges = np.histogram(values, bins=n_bins, range=(lo, hi), density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return centers, dens, float(edges[1] - edges[0])

def _safe_density(process, t, x):

    """
    Returns the theoretical density at time t at point x, or None if it is not defined.
    """

    try:
        with np.errstate(all="ignore"):
            y = np.asarray(process.density(t, x), dtype=float)
        y = np.broadcast_to(y, np.shape(x)).astype(float)
    except (AttributeError, NotImplementedError, ValueError, TypeError,
            ZeroDivisionError, FloatingPointError):
        return None

    y = np.where(np.isfinite(y), y, np.nan)
    return None if np.all(np.isnan(y)) else y

def _path_colors(scores):
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    pos = np.full(n, 0.5) if n <= 1 else np.argsort(np.argsort(scores)) / (n - 1)
    return sample_colorscale(_SCALE, list(pos))

def _base_layout(process, bottom=40, width=None):
    return dict(
        title=dict(text=f"Simulations of {process.name}", x=0.02, xanchor="left",
                   font=dict(size=20)),
        template="plotly_white",
        hovermode="closest",
        margin=dict(l=_MARGIN_L, r=_MARGIN_R, t=90, b=bottom),
        width=width,
    )

# --------------------------------------------------------------------- 1D
def _plot_1d_density(process, paths, n_bins, width):

    """
    Plotting function for 1D paths.
    """

    n_t = process.t.size
    final_values = paths[:, -1]
    mean_path = np.mean(paths, axis=0)
    colors = _path_colors(final_values)   # higher final value = darker path color

    # y-axis limits
    y_min, y_max = float(np.min(paths)), float(np.max(paths))

    pad = 0.05 * (y_max - y_min) or 1.0
    y_lo, y_hi = y_min - pad, y_max + pad

    # Check if the theroretical density exists
    probe = np.linspace(final_values.min(), final_values.max(), 300)
    density_available = _safe_density(process, process.t[-1], probe) is not None

    # Nombre de décimales adapté au pas de temps (affichage de "t = ...")
    dt = float(np.min(np.diff(process.t))) if n_t > 1 else 1.0
    decimals = int(np.clip(np.ceil(np.round(-np.log10(dt), 6)) + 1, 1, 6)) if dt > 0 else 3

    # ---- Traces qui évoluent avec le temps (curseur, histogramme, densité)
    def frame_traces(i):
        t_i = process.t[i]
        values = paths[:, i]
        centers, dens, width = _histogram_frame(values, n_bins)

        traces = [
            go.Scatter(x=[t_i, t_i], y=[y_lo, y_hi], mode="lines",
                       line=dict(width=1.5, dash="dot", color=_MUTED),
                       hoverinfo="skip", showlegend=False, name="Time cursor"),
            go.Bar(x=dens, y=centers, orientation="h",
                   width=None if width is None else 0.94 * width,
                   marker=dict(color=dens / np.max(dens) if dens.size else dens,
                               colorscale=_COLORSCALE, cmin=0, cmax=1,
                               showscale=False, line=dict(width=0)),
                   hovertemplate="density: %{x:.3f}<br>value: %{y:.3f}<extra></extra>",
                   showlegend=False, name="Histogram"),
        ]
        x_max = float(np.max(dens)) if dens.size else 0.0

        if density_available:
            lo, hi = float(np.min(values)), float(np.max(values))
            if hi - lo > 1e-12:
                m = 0.1 * (hi - lo)
                grid = np.linspace(lo - m, hi + m, 300)
            else:
                grid = np.linspace(y_lo, y_hi, 300)

            dens_values = _safe_density(process, t_i, grid)
            if dens_values is None:      # non définie à cet instant (p. ex. t = 0)
                gx, gy = [], []
            else:
                gx, gy = dens_values, grid
                x_max = max(x_max, float(np.nanmax(dens_values)))

            traces.append(
                go.Scatter(x=gx, y=gy, mode="lines",
                           line=dict(width=3, color=_DENSITY),
                           name="Density", showlegend=True,
                           hovertemplate="density: %{x:.3f}<br>value: %{y:.3f}<extra></extra>"))
        return traces, x_max

    # ---- Figure
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        horizontal_spacing=0.02, column_widths=[0.8, 0.2],
                        subplot_titles=("Simulated paths", "Distribution"))

    # Trajectoires (ajoutées en une seule fois : beaucoup plus rapide)
    fig.add_traces(
        [go.Scatter(x=process.t, y=paths[sim], mode="lines",
                    line=dict(width=1, color=colors[sim]), opacity=0.6,
                    name=f"Path {sim + 1}", showlegend=False,
                    hovertemplate="t = %{x:.3f}<br>value = %{y:.3f}<extra>" + f"Path {sim + 1}" + "</extra>")
         for sim in range(process.n_simulations)],
        rows=[1] * process.n_simulations,
        cols=[1] * process.n_simulations,
    )

    # Trajectoire moyenne
    fig.add_trace(
        go.Scatter(x=process.t, y=mean_path, mode="lines",
                   line=dict(width=3, color=_MEAN), name="Mean path",
                   hovertemplate="t = %{x:.3f}<br>mean = %{y:.3f}<extra></extra>"),
        row=1, col=1)

    # État initial = dernier instant (comme avant)
    first_id = len(fig.data)
    init_traces, init_xmax = frame_traces(n_t - 1)
    for tr, col in zip(init_traces, [1, 2, 2]):
        fig.add_trace(tr, row=1, col=col)
    trace_ids = list(range(first_id, first_id + len(init_traces)))

    # ---- Frames (une par instant) : on met aussi à jour l'axe de densité
    frames = []
    for i in range(n_t):
        traces, x_max = frame_traces(i)
        frames.append(
            go.Frame(name=str(i), data=traces, traces=trace_ids,
                     layout=dict(xaxis2=dict(range=[0, 1.1 * x_max if x_max > 0 else 1])))
        )
    fig.frames = frames

    # ---- Alignement slider / axe des temps
    # Dans plotly.js, le centre de la boule parcourt [x + 10 px, x + len - 10 px].
    # On étend donc le slider de 10 px de chaque côté de l'axe des temps du panneau
    # de gauche (dont la plage est fixée à [t_0, t_N]) : la boule est alors
    # exactement sous le curseur vertical (grille de temps uniforme).
    x0, x1 = fig.layout.xaxis.domain
    if width is not None:
        plot_w = width - _MARGIN_L - _MARGIN_R
        s_x = x0 - _GRIP_INSET / plot_w
        s_len = (x1 - x0) + 2 * _GRIP_INSET / plot_w
    else:
        s_x, s_len = x0, x1 - x0

    # ---- Slider minimaliste : rail fin + valeur courante, sans labels qui se chevauchent
    sliders = [dict(
        active=n_t - 1,
        x=s_x, len=s_len, y=0, yanchor="top",
        pad=dict(t=90, b=10),
        bgcolor="#e5e7eb",
        activebgcolor=_ACCENT,
        bordercolor="#cbd5e1",
        borderwidth=1,
        tickcolor="rgba(0,0,0,0)", ticklen=0, minorticklen=0,
        font=dict(size=1, color="rgba(0,0,0,0)"),           # masque les labels des graduations
        currentvalue=dict(prefix="t = ", visible=True, xanchor="left", offset=12,
                          font=dict(size=15, color=_MEAN)),
        transition=dict(duration=0),
        steps=[dict(method="animate",
                    label=f"{t:.{decimals}f}",
                    args=[[str(i)],
                          dict(mode="immediate",
                               frame=dict(duration=0, redraw=True),
                               transition=dict(duration=0))])
               for i, t in enumerate(process.t)],
    )]

    fig.update_layout(
        **_base_layout(process, bottom=150, width=width),
        sliders=sliders,
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.75)"),
    )
    fig.update_xaxes(showgrid=True, gridcolor=_GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID, zeroline=False)
    t_lo, t_hi = float(process.t[0]), float(process.t[-1])
    if t_hi <= t_lo:
        t_lo, t_hi = t_lo - 0.5, t_lo + 0.5
    fig.update_xaxes(title_text="t", range=[t_lo, t_hi], row=1, col=1)
    fig.update_xaxes(title_text="Density",
                     range=[0, 1.1 * init_xmax or 1], row=1, col=2)
    fig.update_yaxes(title_text=process.name, range=[y_lo, y_hi], row=1, col=1)
    fig.update_annotations(font=dict(size=13, color=_MUTED))
    fig.show()

# --------------------------------------------------------------------- plot
def plot_graph(process, density=True, n_bins=30, width=1100):
    """Affiche les trajectoires simulées d'un processus.

    process : objet processus (attributs : path, t, n_simulations, dim, name et,
              optionnellement, une méthode density(t, x))

    density : (1D) histogramme + densité théorique pilotés par un slider
    n_bins  : nombre de classes de l'histogramme
    width   : largeur de la figure (px) en 1D avec densité. Elle permet d'aligner
              exactement la boule du slider sur le curseur vertical ; avec
              width=None la figure est adaptative, mais l'alignement n'est plus
              exact (écart d'au plus ~10 px aux extrémités).
    """
    if process.path is None:
        raise ValueError(
            "The path has not been simulated yet. Please run the simulate method first."
        )

    if process.dim > 3:
        raise ValueError("The path can be plotted only for 1D, 2D and 3D.")

    show_legend = process.n_simulations <= 10   # évite une légende illisible

    if process.dim == 1:
        paths = process.path[:, :, 0]

        if density:
            _plot_1d_density(process, paths, n_bins, width)
        else:
            colors = _path_colors(paths[:, -1])
            fig = go.Figure()
            for sim in range(process.n_simulations):
                fig.add_trace(
                    go.Scatter(x=process.t, y=paths[sim], mode="lines",
                               line=dict(width=2, color=colors[sim]),
                               name=f"Path {sim + 1}", showlegend=show_legend))
            fig.update_layout(**_base_layout(process), xaxis_title="t",
                              yaxis_title=process.name)
            fig.update_xaxes(showgrid=True, gridcolor=_GRID, zeroline=False)
            fig.update_yaxes(showgrid=True, gridcolor=_GRID, zeroline=False)
            fig.show()

    elif process.dim == 2:
        colors = _path_colors(
            np.linalg.norm(process.path[:, -1, :] - process.path[:, 0, :], axis=1))
        fig = go.Figure()
        for sim in range(process.n_simulations):
            fig.add_trace(
                go.Scatter(x=process.path[sim, :, 0], y=process.path[sim, :, 1],
                           mode="lines", line=dict(width=2, color=colors[sim]),
                           name=f"Path {sim + 1}", showlegend=show_legend))
        fig.update_layout(**_base_layout(process), xaxis_title="X", yaxis_title="Y")
        fig.update_xaxes(showgrid=True, gridcolor=_GRID, zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor=_GRID, zeroline=False)
        fig.show()

    else:
        colors = _path_colors(
            np.linalg.norm(process.path[:, -1, :] - process.path[:, 0, :], axis=1))
        fig = go.Figure()
        for sim in range(process.n_simulations):
            fig.add_trace(
                go.Scatter3d(x=process.path[sim, :, 0], y=process.path[sim, :, 1],
                             z=process.path[sim, :, 2], mode="lines",
                             line=dict(width=2, color=colors[sim]),
                             name=f"Path {sim + 1}", showlegend=show_legend))
        fig.update_layout(**_base_layout(process),
                          scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
        fig.show()