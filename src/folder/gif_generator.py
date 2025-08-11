import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection
from PIL import Image
from logzero import logger

# dataframe_path = 'MPcombined.csv'
# geo_path = 'madhya-pradesh.geojson'
# STATE = 'madhya-pradesh'
dataframe_path = 'MHcombined.csv'
geo_path = 'maharashtra.geojson'
STATE = 'MAHARASHTRA'

if not dataframe_path or not geo_path:
    raise FileNotFoundError("❌ Missing environment variables: df_PATH and/or geo_PATH.")

# Read CSV
df = pd.read_csv(dataframe_path)
logger.info(f"Loaded data from {dataframe_path} with {len(df)} rows.")

df['District'] = df['District'].str.upper()
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M')
df = df[(df['date'].dt.year >= 2019) & (df['date'].dt.year <= 2023)]

# Load GeoJSON
with open(geo_path, "r", encoding="utf-8") as f:
    geojson = json.load(f)
logger.info(f"Loaded GeoJSON from {geo_path} with {len(geojson['features'])} features.")

for feat in geojson["features"]:
    feat["properties"]["district"] = feat["properties"]["district"].upper()

# Prepare district polygons and centroids
district_poly_map = {}
district_centroids = {}
all_x, all_y = [], []

for feature in geojson["features"]:
    district = feature["properties"]["district"]
    geom_type = feature["geometry"]["type"]
    coords = feature["geometry"]["coordinates"]

    polygons = []
    if geom_type == "Polygon":
        polygons.append(coords)
    elif geom_type == "MultiPolygon":
        polygons.extend(coords)
    else:
        continue

    district_poly_map[district] = polygons

    try:
        largest = max(polygons[0], key=len)
        xs, ys = zip(*largest)
        centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
        district_centroids[district] = centroid

        # Add to bounding box calculation
        all_x.extend(xs)
        all_y.extend(ys)
    except:
        pass

# Dynamic bounds
min_x, max_x = min(all_x), max(all_x)
min_y, max_y = min(all_y), max(all_y)
padding_x = (max_x - min_x) * 0.05
padding_y = (max_y - min_y) * 0.05

# Prepare output directory
frame_dir = os.path.join("output", "frames")
os.makedirs(frame_dir, exist_ok=True)
image_paths = []

# Rainfall and temperature normalization
all_rain = df.groupby("District")["rainfall_mm"].mean()
rain_min, rain_max = all_rain.min(), all_rain.max()

all_temp = df.groupby(["District", "month"])["mean"].mean()
temp_min, temp_max = all_temp.min(), all_temp.max()
norm = plt.Normalize(temp_min, temp_max)
cmap = plt.cm.plasma

# Generate frames
for month in sorted(df['month'].unique()):
    monthly = df[df["month"] == month]
    temp_by_dist = monthly.groupby("District")["mean"].mean()
    rain_by_dist = monthly.groupby("District")["rainfall_mm"].mean()

    fig, ax = plt.subplots(figsize=(11, 11))
    patches, colors = [], []

    # Draw polygons
    for district, polys in district_poly_map.items():
        value = temp_by_dist.get(district, None)
        color = "#cccccc" if pd.isna(value) else cmap(norm(value))

        for poly_group in polys:
            for ring in poly_group:
                try:
                    poly = Polygon(ring, closed=True)
                    patches.append(poly)
                    colors.append(color)
                except:
                    pass

    pc = PatchCollection(patches, facecolor=colors, edgecolor="black", linewidth=0.3)
    ax.add_collection(pc)

    # Add rainfall circles
    for district, centroid in district_centroids.items():
        rain_val = rain_by_dist.get(district, None)
        if pd.isna(rain_val):
            continue

        min_r, max_r = 0.08, 0.25
        norm_r = (rain_val - rain_min) / (rain_max - rain_min + 1e-6)
        radius = min_r + norm_r * (max_r - min_r)

        circ = Circle(centroid, radius, facecolor="cyan", alpha=0.6,
                      edgecolor='white', linewidth=0.3)
        ax.add_patch(circ)

    month_str = month.strftime("%b-%Y")
    ax.set_title(f"🌡️ Temp (color) + Rainfall (circle) — {month_str}", fontsize=14)

    # Apply dynamic bounds
    ax.set_xlim(min_x - padding_x, max_x + padding_x)
    ax.set_ylim(min_y - padding_y, max_y + padding_y)
    ax.set_aspect("equal")
    ax.axis("off")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.03, pad=0.01)
    cbar.set_label("Temperature (°C)", fontsize=10)

    # Save frame
    filename = os.path.join(frame_dir, f"temp_rain_{month_str}.png")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.1, dpi=150)
    plt.close()
    image_paths.append(filename)
    logger.info(f"Saved: {filename}")

# Create GIF
gif_path = os.path.join("output", f"{STATE}_district_temp_rain_monthly.gif")
frames = [Image.open(p) for p in image_paths]
frames[0].save(
    gif_path,
    format="GIF",
    append_images=frames[1:],
    save_all=True,
    duration=1000,
    loop=0
)
logger.info(f"GIF created: {gif_path}")

# Cleanup
for path in image_paths:
    os.remove(path)
logger.info("Temporary PNGs deleted.")