import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_timeseries(timestamps, *series, start=None, end=None,
                    labels=None, title="Time Series", ylabel="Value", stacked=False):

    num_series = len(series)
    if num_series == 0:
        raise ValueError("At least one values series must be provided.")

    if labels and len(labels) != num_series:
        raise ValueError("Length of labels must match number of series.")

    # If stacked, create subplots; otherwise, overlay on one axis
    if stacked:
        fig, axes = plt.subplots(num_series, 1, figsize=(10, 3*num_series), sharex=True)
        if num_series == 1:
            axes = [axes]
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        axes = [ax] * num_series

    # Plot each series
    for i, (vals, ax) in enumerate(zip(series, axes)):
        label = labels[i] if labels else None
        ax.plot(timestamps, vals, marker='o', markersize=3, linewidth=1.5, label=label)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.grid(True, linestyle='--', alpha=0.6)
        if label:
            ax.legend(loc='upper right')

        if stacked:
            ax.set_ylabel(labels[i] if labels else ylabel)

    # Apply shared formatting
    if not stacked:
        axes[0].set_ylabel(ylabel)
    axes[-1].set_xlabel("Time")

    if start:
        axes[-1].set_xlim(left=start)
    if end:
        axes[-1].set_xlim(right=end)

    fig.suptitle(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_surface_currents(values, total_coordinates, step=2, scale=1000, width=0.002):
    """
    Plot surface currents with refined coastlines and borders (no external tiles).
    """
    W, H = 93, 74
    assert values.shape == (2, H, W)

    lon = total_coordinates[:, 0].reshape(H, W)
    lat = total_coordinates[:, 1].reshape(H, W)

    U_sub = values[0][::step, ::step]
    V_sub = values[1][::step, ::step]
    Lon_sub = lon[::step, ::step]
    Lat_sub = lat[::step, ::step]

    mag = np.sqrt(U_sub**2 + V_sub**2)

    # --- High-res map setup ---
    fig = plt.figure(figsize=(9, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Strait of Georgia bounding box
    min_lon = np.min(total_coordinates[:, 0])
    max_lon = np.max(total_coordinates[:, 0])
    min_lat = np.min(total_coordinates[:, 1])
    max_lat = np.max(total_coordinates[:, 1])

    # Strait of Georgia region (rough bounds)
    ax.set_extent([min_lon, max_lon, min_lat, max_lat])

    # Add refined map features
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='white')
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6, edgecolor='black')
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.4, linestyle=':')
    ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='white', edgecolor='gray', linewidth=0.3)
    ax.add_feature(cfeature.RIVERS.with_scale('10m'), edgecolor='skyblue', linewidth=0.3, alpha=0.7)

    # Optional: add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False

    # --- Plot the quiver arrows ---
    q = ax.quiver(
        Lon_sub, Lat_sub, U_sub, V_sub, mag,
        transform=ccrs.PlateCarree(),
        angles='xy', scale_units='xy', scale=scale,
        cmap='turbo', width=width
    )

    cbar = plt.colorbar(q, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label("Velocity Magnitude (m/s)", rotation=270, labelpad=15)

    ax.set_title("Surface Currents — Strait of Georgia", fontsize=13)
    plt.tight_layout()
    plt.show()







