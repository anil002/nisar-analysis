# NISAR Soil Moisture Analysis - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install h5py numpy matplotlib scipy pandas seaborn scikit-learn rasterio
```

### Step 2: Run Analysis
```bash
python run_complete_analysis.py your_nisar_file.h5
```

### Step 3: View Results
Check the `nisar_outputs/` directory for all results!

---

## 📦 What You Get

### ✅ 15 Types of Analysis
1. Basic Statistics
2. Spatial Analysis  
3. Hydrological Indices
4. Agricultural Applications
5. Drought Monitoring
6. Anomaly Detection
7. Moisture Clustering
8. Insurance Triggers
9. Change Detection
10. Texture Analysis
11. Frequency Analysis
12. Connectivity Analysis
13. Geostatistics
14. Extreme Values
15. Risk Assessment

### ✅ 10+ Export Formats
- GeoTIFF (for QGIS/ArcGIS)
- NetCDF (for climate models)
- CSV (for Excel/R/Python)
- GeoJSON (for web maps)
- ML-ready datasets (for AI models)
- Google Earth Engine (for cloud analysis)

### ✅ Beautiful Visualizations
- 6-panel comprehensive analysis
- 9-panel advanced analysis
- All at 300 DPI publication quality

---

## 💡 Quick Examples

### Example 1: Just Load and View
```python
from nisar_comprehensive_analysis import NISARSoilMoistureAnalyzer

analyzer = NISARSoilMoistureAnalyzer("file.h5")
analyzer.load_data()
analyzer.basic_statistics()
```

### Example 2: Drought Assessment
```python
analyzer.load_data()
drought_info = analyzer.drought_monitoring()
# Automatically prints drought severity and recommendations
```

### Example 3: Export for GIS
```python
analyzer.load_data()
analyzer.export_geotiff("my_soil_moisture.tif")
# Open in QGIS/ArcGIS immediately!
```

### Example 4: Machine Learning
```python
from nisar_data_export import NISARDataExporter

exporter = NISARDataExporter(sm_data)
ml_features = exporter.export_ml_features("features.csv")
# Ready for Random Forest, XGBoost, Neural Networks!
```

---

## 🎯 Use Cases

### For Researchers
- ✅ Drought studies
- ✅ Climate change analysis
- ✅ Hydrological modeling
- ✅ Agricultural research

### For Agronomists
- ✅ Crop suitability maps
- ✅ Irrigation planning
- ✅ Growing season monitoring
- ✅ Yield prediction

### For Insurance Companies
- ✅ Parametric trigger analysis
- ✅ Risk assessment
- ✅ Payout calculations
- ✅ Portfolio monitoring

### For Government/NGOs
- ✅ Drought early warning
- ✅ Disaster management
- ✅ Policy planning
- ✅ Food security monitoring

---

## 📊 Sample Output

```
NISAR L3 SOIL MOISTURE ANALYSIS
================================

BASIC STATISTICS
Mean:     0.2345 m³/m³
Median:   0.2298 m³/m³
Std Dev:  0.0456 m³/m³

DROUGHT ASSESSMENT
Drought Index: 0.34
Status: ✅ No significant drought

AGRICULTURAL SUITABILITY
Wheat:    78.4% suitable  ✅ Optimal
Rice:     45.2% suitable  ⚠️  Suboptimal
Cotton:   82.1% suitable  ✅ Optimal

INSURANCE TRIGGERS
Critical Drought:  NOT ACTIVATED
Area Below 0.12:   8.3%

RISK ASSESSMENT
Overall Risk:      32.4/100  ⚡ Moderate Risk
High Risk Areas:   1,234 pixels

RECOMMENDATIONS
✅ Continue normal monitoring
⚡ Review crop water requirements
📊 Monitor for changes
```

---

## 🔥 Pro Tips

### Tip 1: Large Files
If file is too large for memory:
```python
# Process by tiles
analyzer.export_csv("output.csv", sample_rate=10)  # Sample every 10 pixels
```

### Tip 2: Custom Analysis
```python
# Access raw data
sm_data = analyzer.sm_data  # numpy array
lat = analyzer.lat
lon = analyzer.lon

# Do your own analysis!
custom_metric = np.mean(sm_data[sm_data > 0.3])
```

### Tip 3: Automation
```bash
# Batch process multiple files
for file in *.h5; do
    python run_complete_analysis.py "$file"
done
```

---

## ❓ Troubleshooting

### Problem: "File not found"
**Solution:** Check the file path is correct
```python
import os
print(os.path.exists("your_file.h5"))  # Should be True
```

### Problem: "Cannot find dataset"
**Solution:** Check HDF5 structure
```python
import h5py
with h5py.File("file.h5", 'r') as f:
    f.visititems(print)  # Shows all paths
```

### Problem: "Memory error"
**Solution:** Use sampling
```python
analyzer.export_csv("output.csv", sample_rate=5)
```

---

## 📚 Documentation

**Full Documentation:** See `README.md`

**Key Modules:**
- `nisar_comprehensive_analysis.py` - Main analysis (15 types)
- `nisar_advanced_analysis.py` - Advanced techniques
- `nisar_data_export.py` - Export to 10+ formats
- `run_complete_analysis.py` - One-click pipeline

---

## 🎓 Learn More

### Your NISAR File Contains:
- **Product:** L3 Soil Moisture EASE Grid 2.0
- **Sensor:** L-SAR (L-band radar)
- **Coverage:** 37 seconds (2025-12-16)
- **Region:** Egypt (Nile Delta)
- **Resolution:** ~3 km
- **Polarization:** HH+HV (dual-pol)

### Why L-band SAR is Great:
✅ Penetrates vegetation  
✅ Works through clouds  
✅ Day/night operation  
✅ Direct moisture sensing  
✅ Superior to optical sensors  

---

## 🚀 Next Steps

1. **Run the complete analysis** (5 minutes)
   ```bash
   python run_complete_analysis.py file.h5
   ```

2. **Open the visualizations** (publication quality)
   - `01_comprehensive_analysis.png`
   - `02_advanced_analysis.png`

3. **Read the executive summary**
   - `executive_summary.txt`

4. **Import GeoTIFF to QGIS/ArcGIS**
   - `soil_moisture.tif`

5. **Build ML models with features**
   - `ml_features.csv`

---

## 💬 Support

Need help? Check:
1. `README.md` - Full documentation
2. NISAR website: https://nisar.jpl.nasa.gov/
3. Sample outputs in `nisar_outputs/`

---

## ⭐ Features Highlight

```
✅ 15 analysis types        ✅ Publication-quality plots
✅ 10+ export formats       ✅ ML-ready datasets  
✅ Parametric insurance     ✅ Drought monitoring
✅ Agricultural insights    ✅ Risk assessment
✅ GIS integration         ✅ Cloud-ready exports
```

---

**Ready to analyze?**

```bash
python run_complete_analysis.py your_file.h5
```

**That's it! 🎉**

---

*NISAR Analysis Suite v1.0 | February 2026*
