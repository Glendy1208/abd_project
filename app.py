from flask import Flask, render_template, request, url_for, Response
import pandas as pd
import folium
import json
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Muat model klasifikasi gambar
model = load_model('models/egg_classification_model.h5')

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def classify_frame(frame):
    preprocessed_image = preprocess_image(frame)
    prediction = model.predict(preprocessed_image)
    result = 'Damaged' if prediction[0][0] > 0.5 else 'Not Damaged'
    return result

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            result = classify_frame(frame)
            height, width, _ = frame.shape
            start_point = (int(width * 0.3), int(height * 0.3))
            end_point = (int(width * 0.7), int(height * 0.7))
            color = (0, 255, 0) if result == 'Not Damaged' else (0, 0, 255)
            cv2.rectangle(frame, start_point, end_point, color, 2)
            cv2.putText(frame, result, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Load dataset
data_path = "data/egg_clustering_data_english.csv"
coordinates_path = "data/province_coordinates.json"
data = pd.read_csv(data_path)

# Load coordinates
with open(coordinates_path, "r") as f:
    coordinates = json.load(f)

# Convert coordinates to a dictionary for easy access
coordinates_dict = {item["name"]: (item["latitude"], item["longitude"]) for item in coordinates}

# Define colors for clusters
cluster_colors = {"Rendah": "lightgreen", "Sedang": "yellow", "Tinggi": "red"}

# Load forecast data
forecast_data_path = "data/Formatted_Forecasting_Data.csv"
forecast_data = pd.read_csv(forecast_data_path)

# Restructure data for forecasting
forecast_data_long = forecast_data.melt(
    id_vars=["No", "Komoditas (Rp)"],
    var_name="Date",
    value_name="Price"
)

# Clean up the data
forecast_data_long["Date"] = forecast_data_long["Date"].astype(str).str.strip()  # Ensure 'Date' column is of type string before using .str accessor
forecast_data_long["Price"] = (
    forecast_data_long["Price"].astype(str).str.replace(",", "").replace("-", None).astype(float)  # Ensure 'Price' column is of type string before using .str accessor
)  # Convert price to numeric
forecast_data_long["Date"] = pd.to_datetime(forecast_data_long["Date"], format="%d/ %m/ %Y", errors="coerce")  # Convert date to datetime

# Drop invalid rows and extract year and month
forecast_data_long = forecast_data_long.dropna(subset=["Date", "Price"])
forecast_data_long["Year"] = forecast_data_long["Date"].dt.year
forecast_data_long["Month"] = forecast_data_long["Date"].dt.month

# Filter data untuk tahun 2023, 2024, dan 2025
forecast_data_filtered = forecast_data_long[forecast_data_long["Year"].isin([2023, 2024, 2025])]

def generate_forecast_graph(data):
    """Generate the forecast graph and return it as a base64 image."""
    # Aggregate data by month and year (average price per month)
    monthly_avg_prices = data.groupby(["Year", "Month"])["Price"].mean().reset_index()
    prov = data["Komoditas (Rp)"].iloc[0]
    # Create a line plot for 2023, 2024, and 2025
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=monthly_avg_prices,
        x="Month",
        y="Price",
        hue="Year",
        style="Year",
        markers=True,
        dashes=False
    )

    # Customize the plot
    plt.title(f"Harga Telur {prov} (2023-2025)", fontsize=16)
    plt.xlabel("Bulan", fontsize=12)
    plt.ylabel("Harga (Rp)", fontsize=12)
    plt.xticks(ticks=range(1, 13), labels=["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"])
    plt.legend(title="Tahun", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close()

    return img_base64


@app.route("/", methods=["GET", "POST"])
def index():
    # List of years available in the dataset
    years = [col.split("_")[0] for col in data.columns if "_cluster" in col]
    selected_year = request.args.get("year", years[-1])  # Default to the latest year

    # List of provinces available in the dataset
    provinces = forecast_data["Komoditas (Rp)"].unique().tolist()
    selected_province = request.args.get("province", "")

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

                # Create HTML content for popup with a button
                popup_html = f"""
                    <div>
                        <strong>{province}</strong><br>
                        Produksi: {production} ton<br>
                        Cluster: {cluster}<br>
                    </div>
                """
                iframe = folium.IFrame(popup_html, width=300, height=100)
                popup = folium.Popup(iframe, max_width=300)

                # Add marker with popup
                folium.CircleMarker(
                    location=coord,
                    radius=8,
                    color=cluster_colors[cluster],
                    fill=True,
                    fill_color=cluster_colors[cluster],
                    fill_opacity=0.7,
                    tooltip=tooltip_text,
                    popup=popup
                ).add_to(m)

        # Render map as HTML
        map_html = m._repr_html_()

    # Generate forecast graph based on selected province and year
    forecast_graph = None
    if selected_year == "2023":
        if selected_province:
            province_data = forecast_data_filtered[forecast_data_filtered["Komoditas (Rp)"] == selected_province]
            if not province_data.empty:
                forecast_graph = generate_forecast_graph(province_data)
        else:
            forecast_graph = generate_forecast_graph(forecast_data_filtered)

    return render_template(
        "index.html",
        years=years,
        selected_year=selected_year,
        provinces=provinces,
        selected_province=selected_province,
        map_html=map_html,
        forecast_graph=forecast_graph
    )

if __name__ == "__main__":
    app.run(debug=True)