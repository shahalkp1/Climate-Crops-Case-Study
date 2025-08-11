# Climate-Crops-Case-Study

This repository contains a comprehensive case study analyzing climate resiliency for agricultural production in the Indian states of **Maharashtra (MH)** and **Madhya Pradesh (MP)**. Using historical climate and remote sensing data, this project evaluates crop performance, assesses economic impacts, and proposes evidence-based recommendations to enhance agricultural resilience.

A detailed project presentation is available [here](https://docs.google.com/presentation/d/19iMewfU2JwAaToTk8Sa-QxGCWAXy3M05LJs7vHJ9qC8/edit?slide=id.g37616b1d3cb_0_163#slide=id.g37616b1d3cb_0_163).



## Methodology and Key Analysis

This project follows a systematic analytical pipeline using Python scripts and notebooks:

- **Data Cleaning & Preprocessing**  
  The `climate_cleaning_pipeline.py` script handles missing data imputation and removes outliers from raw climate datasets, ensuring data quality.

- **Satellite Data Integration & Machine Learning**  
  The `yield_climate.py` script fetches satellite NDVI data (a proxy for crop health) via the Google Earth Engine API. It integrates this with climate data and trains a RandomForestRegressor model to predict crop yield and assess resilience.

- **Financial Analysis**  
  The `crop_climate.ipynb` notebook analyzes the financial impact of climate variability by combining yield and climate data with market prices, calculating profit or loss per hectare for various crops over multiple years.

- **Data Visualization**  
  The `climate_data_analysis.ipynb` notebook produces detailed static plots for trend analysis. The `gif_generator.py` script creates an animated GIF visualizing monthly temperature and rainfall changes to highlight dynamic climate patterns.

---

## File Descriptions

| File/Folder                 | Description                                                      |
|----------------------------|------------------------------------------------------------------|
| `src/climate_cleaning_pipeline.py` | Cleans and preprocesses raw climate data (outlier removal, missing data imputation). |
| `src/yield_climate.py`             | Fetches NDVI satellite data, calculates anomalies, and trains machine learning models for yield prediction. |
| `src/gif_generator.py`             | Generates an animated GIF to visualize monthly climate changes on a map. |
| `notebooks/climate_data_analysis.ipynb` | Exploratory data analysis with static plots for climate data.    |
| `notebooks/crop_climate.ipynb`    | Financial impact analysis notebook, mapping crop profits/losses to seasons and years. |
| `data/`                 | Contains raw CSV and GeoJSON data files used in analysis.        |

---

## How to Run the Project

### Prerequisites

- **Python 3.x** installed  
- Required Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `logzero`, `earthengine-api`, `Pillow`  
  (Use a virtual environment recommended)  
- **Google Earth Engine Python API** installed and authenticated  

### Execution Steps

1. **Clean and preprocess raw climate data:**  
   ```bash
   python src/climate_cleaning_pipeline.py
   ```

2. **Run yield prediction and modeling:**  
   ```bash
   python src/yield_climate.py
   ```

3. **Generate animated GIF for climate visualization:**  
   ```bash
   python src/gif_generator.py
   ```

4. **Open Jupyter Notebooks for exploratory and financial analysis:**  
   ```bash
   jupyter notebook
   ```  
   Then run the notebooks:  
   - `notebooks/climate_data_analysis.ipynb`  
   - `notebooks/crop_climate.ipynb`

---

## Technologies Used

- **Python:** Main programming language  
- **Pandas & NumPy:** Data manipulation and numerical operations  
- **Matplotlib & Seaborn:** Data visualization  
- **Google Earth Engine (GEE):** Satellite and geospatial data access  
- **Scikit-learn:** Machine learning (RandomForestRegressor)  
- **Pillow:** Image processing and GIF generation  

---
