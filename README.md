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

## 🔬 Advanced Analysis Module

### 9. **Change Detection**
```python
from nisar_advanced_analysis import NISARAdvancedAnalysis

analyzer = NISARAdvancedAnalysis(sm_data)
changes = analyzer.change_detection(previous_sm)
```

- Temporal change quantification
- Significant change zones
- Drying/wetting trend detection

---

### 10. **Texture Analysis**
- Local variability metrics
- Homogeneity assessment
- Edge detection (boundaries)
- Spatial pattern characterization

---

### 11. **Frequency Domain Analysis**
- 2D Fourier transform
- Spatial frequency spectra
- Dominant pattern scales
- Radial profile analysis

---

### 12. **Connectivity Analysis**
- Patch identification
- Fragmentation indices
- Size distribution
- Spatial connectivity metrics

---

### 13. **Geostatistical Analysis**
- Variogram computation
- Spatial autocorrelation (Moran's I)
- Range estimation
- Sill/nugget analysis

---

### 14. **Extreme Value Analysis**
- Percentile-based extremes
- IQR outlier detection
- Spatial clustering of extremes
- Return period estimation

---

### 15. **Multi-Factor Risk Assessment**
```python
risk = analyzer.risk_assessment()
```

**Risk Factors:**
1. Absolute moisture level
2. Spatial variability
3. Proximity to extremes

**Output:**
- Risk score (0-100)
- Risk category distribution
- Actionable recommendations

---

## 📤 Export Formats

### Geospatial Formats

#### 1. **GeoTIFF** (Most Common)
```python
from nisar_data_export import NISARDataExporter

exporter = NISARDataExporter(sm_data)
exporter.export_geotiff_advanced("output.tif")
```

**Features:**
- LZW compression
- Tiled structure (256x256)
- Full metadata tags
- EPSG:4326 projection

**Use With:**
- QGIS, ArcGIS
- Python (rasterio, GDAL)
- R (raster, terra)

---

#### 2. **NetCDF** (Climate Models)
```python
exporter.export_netcdf("output.nc")
```

**Features:**
- CF-compliant metadata
- Time dimension support
- Standard variable names
- Coordinate reference system

**Use With:**
- Climate models (WRF, RegCM)
- CDO, NCO tools
- Python (xarray, netCDF4)

---

#### 3. **GeoJSON** (Web Mapping)
```python
exporter.export_geojson("output.geojson", grid_size=10)
```

**Use With:**
- Leaflet, Mapbox
- D3.js visualizations
- Web GIS applications

---

### Tabular Formats

#### 4. **CSV** (Universal)
```python
df = exporter.export_csv("output.csv", sample_rate=1)
```

**Columns:**
- latitude, longitude
- row, col (pixel indices)
- soil_moisture

---

#### 5. **ML Features CSV**
```python
df = exporter.export_ml_features("ml_features.csv")
```

**Features Included:**
- soil_moisture (original)
- local_mean, local_std
- gradient_magnitude
- texture metrics
- distance_to_dry
- moisture_anomaly
- coefficient_of_variation
- moisture_class (categorical)

**Ready for:**
- Random Forest, XGBoost
- Neural networks
- Statistical modeling

---

### Machine Learning Formats

#### 6. **TensorFlow Dataset**
```python
from nisar_data_export import NISARMLPreparation

ml_prep = NISARMLPreparation(sm_data)
ml_prep.export_tensorflow_dataset("tf_dataset/")
```

**Output Files:**
- `patches.npy` - Image patches (32x32)
- `labels.npy` - Class labels (0=dry, 1=optimal, 2=saturated)
- `dataset_info.json` - Metadata
- `load_dataset.py` - Loading script

**Use For:**
- CNN training
- Deep learning pipelines
- Transfer learning

---

### Cloud Platform Formats

#### 7. **Google Earth Engine**
```python
exporter.export_gee_asset("my_nisar_product")
```

**Creates:**
- GeoTIFF for upload
- Manifest JSON
- Upload instructions
- GCS transfer commands

---

## 📈 Visualization Outputs

### Main Visualization Suite
Generated by: `analyzer.create_visualizations()`

**6 plots:**
1. Soil Moisture Map (color-coded)
2. Distribution Histogram
3. Drought Severity Zones
4. Quartile Box Plots
5. Spatial Variability (gradient)
6. Agricultural Classification

**Saved as:** `nisar_analysis_comprehensive.png` (300 DPI)

---

### Advanced Visualization Suite
Generated by: `create_advanced_visualizations()`

**9 plots:**
1. Spatial Texture
2. Moisture Boundaries
3. Dry Zone Connectivity
4. Wet Zone Connectivity
5. Extreme Value Zones
6. Multi-Factor Risk Map
7. Risk Distribution (pie chart)
8. Spatial Homogeneity
9. Summary Statistics Panel

---

## 🌾 Agricultural Applications

### Crop Suitability Analysis
```python
agri = analyzer.agricultural_analysis()
```

**Crops Analyzed:**
- Wheat (optimal: 0.20-0.30 m³/m³)
- Rice (optimal: 0.30-0.45 m³/m³)
- Cotton (optimal: 0.15-0.25 m³/m³)
- Soybean (optimal: 0.20-0.35 m³/m³)
- Maize (optimal: 0.22-0.32 m³/m³)

**Output:**
- Percentage of area suitable per crop
- Status: Optimal / Suboptimal / Poor

---

### Irrigation Recommendations
**4 Zones:**
1. No irrigation (SM > 0.30)
2. Light irrigation (0.20-0.30)
3. Moderate irrigation (0.15-0.20)
4. Heavy irrigation (< 0.15)

---

## 💰 Parametric Insurance Implementation

### Trigger Structure
```python
insurance = analyzer.insurance_triggers()
```

**5-Level Trigger System:**

| Level | Threshold | Action |
|-------|-----------|--------|
| Critical Drought | < 0.10 m³/m³ | Full payout |
| Severe Drought | < 0.12 m³/m³ | 75% payout |
| Moderate Drought | < 0.15 m³/m³ | 50% payout |
| Mild Stress | < 0.18 m³/m³ | 25% payout |
| Low Moisture | < 0.20 m³/m³ | Monitor only |

**Example Payout:**
- If >50% area below 0.10 → ₹10,000/hectare
- If >30% area below 0.12 → ₹5,000/hectare

---

## 🔍 Metadata Structure

### From Your NISAR File

```
Product: NISAR_L3_SME2_BETA_V1
Processing Level: L3
Product Type: Soil Moisture EASE Grid 2.0

Temporal Coverage:
- Start: 2025-12-16 16:42:23 UTC
- End: 2025-12-16 16:43:00 UTC

Spatial Coverage:
- Bounds: [29.12°E, 29.38°N] to [32.41°E, 32.25°N]
- Region: Egypt (Nile Delta area)

Orbit: 2009
Track: 7
Frame: 73
Direction: Descending

Instrument: L-SAR
Polarization: HH+HV (dual-pol)
Frequencies: A, B

Processing:
- Center: JPL
- Version: R05.00.1
- Date: 2026-01-20
```

---

## 🎓 Research Applications

### 1. **Drought Studies**
- Early warning indicators
- Historical drought reconstruction
- Impact assessment
- Recovery monitoring

### 2. **Agricultural Research**
- Crop stress detection
- Yield prediction modeling
- Irrigation optimization
- Precision agriculture

### 3. **Hydrological Modeling**
- Soil moisture assimilation
- Flood forecasting
- Groundwater recharge
- Watershed management

### 4. **Climate Studies**
- Land-atmosphere coupling
- Evapotranspiration estimation
- Heat wave analysis
- Climate change impacts

---

## 📝 Citation

If you use this analysis suite, please cite:

```
NISAR Mission: NASA-ISRO SAR Mission
Product: L3 Soil Moisture EASE Grid 2.0 (SME2)
Processing: JPL (Jet Propulsion Laboratory)
```

---

## 🤝 Support & Contribution

### Common Issues

**Issue 1: "File not found"**
- Check file path is correct
- Ensure .h5 file is accessible

**Issue 2: "Cannot find dataset"**
- HDF5 structure varies by version
- Use `view` tool to explore structure
- Adjust path in `load_data()` method

**Issue 3: "Memory error"**
- File too large for RAM
- Use sampling in export functions
- Process by tiles

---

## 📚 Additional Resources

### NISAR Documentation
- [NISAR Mission Website](https://nisar.jpl.nasa.gov/)
- [Product Specifications](https://nisar.jpl.nasa.gov/data/products/)
- [Algorithm Theoretical Basis Documents (ATBDs)](https://nisar.jpl.nasa.gov/documents/26/)

### Soil Moisture Resources
- SMAP Mission
- ESA CCI Soil Moisture
- SMOS Mission

---

## 🚀 Performance Tips

### For Large Files
```python
# Process by chunks
chunk_size = 1000
for i in range(0, h, chunk_size):
    chunk = sm_data[i:i+chunk_size, :]
    # Process chunk
```

### Parallel Processing
```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=4)(
    delayed(process_tile)(tile) for tile in tiles
)
```

---

## 📊 Expected Output Summary

Running the complete suite generates:

**Files (10+):**
1. `nisar_analysis_comprehensive.png` - Main plots
2. `nisar_soil_moisture.tif` - GeoTIFF
3. `nisar_analysis_report.txt` - Text report
4. `nisar_soil_moisture.nc` - NetCDF
5. `nisar_soil_moisture.csv` - Tabular data
6. `nisar_soil_moisture.geojson` - Web mapping
7. `nisar_ml_features.csv` - ML features
8. `nisar_tf_dataset/` - TensorFlow dataset
9. Advanced visualizations
10. Export instructions

**Analysis Time:** ~2-5 minutes (depending on file size)

---

## 🎯 Next Steps

### For Operational Use
1. Integrate with real-time data pipelines
2. Automate daily processing
3. Set up alerting systems
4. Deploy as web service

### For Research
1. Multi-temporal analysis
2. Fusion with other datasets (Sentinel, MODIS)
3. Machine learning model development
4. Validation with ground measurements

---

## ⚡ Quick Reference

### Essential Commands
```python
# Load and analyze
analyzer = NISARSoilMoistureAnalyzer("file.h5")
analyzer.load_data()

# Run all analyses
analyzer.basic_statistics()
analyzer.spatial_analysis()
analyzer.hydrological_indices()
analyzer.agricultural_analysis()
analyzer.drought_monitoring()
analyzer.insurance_triggers()

# Export
analyzer.create_visualizations()
analyzer.export_geotiff("output.tif")
analyzer.generate_report("report.txt")
```

---

## 📞 Contact

For questions or issues:
- Open an issue on GitHub
- Consult NISAR documentation
- Check JPL support resources

---

**Version:** 1.0  
**Last Updated:** February 2026  
**Tested With:** NISAR L3 SME2 BETA V1 products
