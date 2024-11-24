from flask import Flask, render_template, request
import pandas as pd
import folium
import json

app = Flask(__name__)

# Load dataset
data_path = "data/egg_clustering_data_english.csv"
coordinates_path = "data/province_coordinates.json"
data = pd.read_csv(data_path)

# Load forecast data
forecast_data_path = "data/ABDFORECAST.csv"
forecast_data = pd.read_csv(forecast_data_path)

# Load coordinates
with open(coordinates_path, "r") as f:
    coordinates = json.load(f)

# Convert coordinates to a dictionary for easy access
coordinates_dict = {item["name"]: (item["latitude"], item["longitude"]) for item in coordinates}

# Define colors for clusters
cluster_colors = {"Rendah": "lightgreen", "Sedang": "yellow", "Tinggi": "red"}

@app.route("/", methods=["GET", "POST"])
def index():
    # List of years available in the dataset
    years = [col.split("_")[0] for col in data.columns if "_cluster" in col]
    selected_year = request.args.get("year", years[-1])  # Default to the latest year

    # Generate the map if a year is selected
    map_html = None
    if selected_year:
        production_col = f"{selected_year}_production"
        cluster_col = f"{selected_year}_cluster"

        # Create an interactive map
        m = folium.Map(location=[-2.5, 118.0], zoom_start=5, tiles="cartodbpositron")
        for _, row in data.iterrows():
            province = row["Provinsi"]
            production = row[production_col]
            cluster = row[cluster_col]
            coord = coordinates_dict.get(province)

            if coord:
                # Tooltip text
                tooltip_text = f"{province}: {production} ton ({cluster})"

                # Add marker
                folium.CircleMarker(
                    location=coord,
                    radius=8,
                    color=cluster_colors[cluster],
                    fill=True,
                    fill_color=cluster_colors[cluster],
                    fill_opacity=0.7,
                    tooltip=tooltip_text
                ).add_to(m)

        # Render map as HTML
        map_html = m._repr_html_()

    return render_template("index.html", years=years, selected_year=selected_year, map_html=map_html)


if __name__ == "__main__":
    app.run(debug=True)
