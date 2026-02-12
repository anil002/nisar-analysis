# NISAR L3 Soil Moisture Comprehensive Analysis Suite

## 🎯 Overview

Complete analysis toolkit for NISAR L3 Soil Moisture (SME2) products with **15+ analysis types**, **10+ export formats**, and **ML-ready pipelines**.

---

## 📦 Installation

### Required Dependencies
```bash
pip install h5py numpy matplotlib scipy pandas seaborn scikit-learn
```

### Optional Dependencies (for advanced features)
```bash
# For GeoTIFF export
pip install rasterio

# For NetCDF export
pip install netCDF4

# For geospatial analysis
pip install geopandas shapely

# For machine learning
pip install tensorflow torch
```

---

## 🚀 Quick Start

### Basic Usage
```python
from nisar_comprehensive_analysis import NISARSoilMoistureAnalyzer

# Initialize analyzer
analyzer = NISARSoilMoistureAnalyzer("your_file.h5")

# Load data
analyzer.load_data()

# Run analyses
stats = analyzer.basic_statistics()
spatial = analyzer.spatial_analysis()
hydro = analyzer.hydrological_indices()
drought = analyzer.drought_monitoring()

# Generate visualizations
fig = analyzer.create_visualizations()
fig.savefig('results.png', dpi=300)

# Export data
analyzer.export_geotiff("output.tif")
analyzer.generate_report("report.txt")
```

### Run Complete Suite
```bash
python nisar_comprehensive_analysis.py
```

---

## 📊 Analysis Types

### 1. **Basic Statistical Analysis**
- Descriptive statistics (mean, median, std, CV, skewness, kurtosis)
- Distribution testing (normality tests)
- Quartile analysis
- Outlier detection

**Output:** Statistical summary with interpretation

---

### 2. **Spatial Analysis**
- Regional statistics by quadrants
- Gradient analysis (spatial variability)
- Hotspot/coldspot detection
- Spatial clustering patterns

**Output:** Spatial pattern maps and metrics

---

### 3. **Hydrological Indices**
- Soil moisture classification (dry/optimal/saturated)
- Soil Water Deficit Index (SWDI)
- Plant Available Water (PAW)
- Water Stress Index (WSI)

**Use Cases:**
- Drought monitoring
- Irrigation scheduling
- Crop water stress assessment

---

### 4. **Agricultural Applications**
- Crop-specific suitability analysis
  - Wheat, Rice, Cotton, Soybean, Maize
- Irrigation requirement zones
- Growing season indicators
- Germination suitability

**Output:** Actionable agricultural recommendations

---

### 5. **Drought Monitoring**
- Multi-level drought classification
  - None / Moderate / Severe / Extreme
- Percentile-based thresholds (P5, P10, P20)
- Overall drought index
- Severity assessment

**Use Cases:**
- Early warning systems
- Disaster management
- Policy planning

---

### 6. **Anomaly Detection**
- Z-score based statistical anomalies
- Local spatial outliers
- Extreme value identification
- Pattern deviation analysis

---

### 7. **Moisture Zone Clustering**
- K-means clustering (5 zones by default)
- Zone characterization
- Spatial segmentation
- Moisture regime mapping

---

### 8. **Parametric Insurance Triggers**
- Multi-threshold trigger analysis
- Area-based payout calculations
- Risk level assessment
- Activation status monitoring

**Thresholds:**
- Critical: <0.10 m³/m³
- Severe: <0.12 m³/m³
- Moderate: <0.15 m³/m³
- Mild: <0.18 m³/m³

---
