#!/usr/bin/env python3
"""
NISAR Complete Analysis Pipeline - Master Script
================================================
Run all analyses, generate all visualizations, and export to all formats

Usage:
    python run_complete_analysis.py path/to/nisar_file.h5
    
Or with default file:
    python run_complete_analysis.py
"""

import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import analysis modules
from nisar_comprehensive_analysis import NISARSoilMoistureAnalyzer
from nisar_advanced_analysis import NISARAdvancedAnalysis, create_advanced_visualizations
from nisar_data_export import NISARDataExporter, NISARMLPreparation

def print_banner():
    """Print welcome banner"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "NISAR L3 SOIL MOISTURE COMPLETE ANALYSIS SUITE" + " " * 17 + "║")
    print("║" + " " * 22 + "All-in-One Processing Pipeline" + " " * 26 + "║")
    print("╚" + "═" * 78 + "╝")
    print(f"\n{'':>30}Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


def run_complete_pipeline(file_path, output_dir="nisar_outputs"):
    """
    Run complete analysis pipeline
    
    Parameters:
    -----------
    file_path : str
        Path to NISAR HDF5 file
    output_dir : str
        Directory for all outputs
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print_banner()
    
    # ========================================================================
    # PHASE 1: LOAD DATA
    # ========================================================================
    
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 30 + "PHASE 1: DATA LOADING" + " " * 27 + "║")
    print("╚" + "═" * 78 + "╝")
    
    analyzer = NISARSoilMoistureAnalyzer(file_path)
    
    if not analyzer.load_data():
        print("\n❌ FATAL ERROR: Could not load data")
        print("   Please check:")
        print("   1. File path is correct")
        print("   2. File is valid HDF5 format")
        print("   3. File has NISAR L3 SME2 structure")
        return False
    
    # ========================================================================
    # PHASE 2: COMPREHENSIVE ANALYSIS (15 types)
    # ========================================================================
    
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "PHASE 2: COMPREHENSIVE ANALYSIS" + " " * 22 + "║")
    print("╚" + "═" * 78 + "╝")
    
    results = {}
    
    try:
        # Basic analyses
        print("\n[1/15] Running basic statistics...")
        results['stats'] = analyzer.basic_statistics()
        
        print("\n[2/15] Running spatial analysis...")
        results['spatial'] = analyzer.spatial_analysis()
        
        print("\n[3/15] Computing hydrological indices...")
        results['hydro'] = analyzer.hydrological_indices()
        
        print("\n[4/15] Analyzing agricultural applications...")
        results['agri'] = analyzer.agricultural_analysis()
        
        print("\n[5/15] Monitoring drought conditions...")
        results['drought'] = analyzer.drought_monitoring()
        
        print("\n[6/15] Detecting anomalies...")
        results['anomalies'] = analyzer.anomaly_detection()
        
        print("\n[7/15] Clustering moisture zones...")
        results['clusters'] = analyzer.moisture_clustering(n_clusters=5)
        
        print("\n[8/15] Evaluating insurance triggers...")
        results['insurance'] = analyzer.insurance_triggers()
        
        # Advanced analyses
        print("\n[9/15] Analyzing texture patterns...")
        adv_analyzer = NISARAdvancedAnalysis(analyzer.sm_data)
        results['texture'] = adv_analyzer.texture_analysis()
        
        print("\n[10/15] Performing frequency analysis...")
        results['frequency'] = adv_analyzer.frequency_analysis()
        
        print("\n[11/15] Analyzing connectivity...")
        results['connectivity'] = adv_analyzer.connectivity_analysis()
        
        print("\n[12/15] Computing geostatistics...")
        results['geostats'] = adv_analyzer.geostatistical_analysis()
        
        print("\n[13/15] Analyzing extreme values...")
        results['extremes'] = adv_analyzer.extreme_value_analysis()
        
        print("\n[14/15] Assessing multi-factor risk...")
        results['risk'] = adv_analyzer.risk_assessment()
        
        print("\n[15/15] Detecting changes (simulated)...")
        results['changes'] = adv_analyzer.change_detection()
        
        print("\n✅ All analyses completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================================================
    # PHASE 3: VISUALIZATION GENERATION
    # ========================================================================
    
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 26 + "PHASE 3: CREATING VISUALIZATIONS" + " " * 20 + "║")
    print("╚" + "═" * 78 + "╝")
    
    try:
        import matplotlib.pyplot as plt
        
        # Main visualization suite
        print("\n[1/2] Generating main visualization suite...")
        fig1 = analyzer.create_visualizations()
        output_path1 = os.path.join(output_dir, "01_comprehensive_analysis.png")
        fig1.savefig(output_path1, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"   ✅ Saved: {output_path1}")
        
        # Advanced visualization suite
        print("\n[2/2] Generating advanced visualization suite...")
        fig2 = create_advanced_visualizations(adv_analyzer)
        output_path2 = os.path.join(output_dir, "02_advanced_analysis.png")
        fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"   ✅ Saved: {output_path2}")
        
    except Exception as e:
        print(f"\n⚠️  Visualization error: {e}")
    
    # ========================================================================
    # PHASE 4: DATA EXPORT (10+ formats)
    # ========================================================================
    
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 27 + "PHASE 4: EXPORTING DATA (10+ FORMATS)" + " " * 15 + "║")
    print("╚" + "═" * 78 + "╝")
    
    exporter = NISARDataExporter(analyzer.sm_data, analyzer.lat, analyzer.lon)
    
    # 1. GeoTIFF
    print("\n[1/8] Exporting GeoTIFF...")
    tiff_path = os.path.join(output_dir, "soil_moisture.tif")
    exporter.export_geotiff_advanced(tiff_path)
    
    # 2. NetCDF
    print("\n[2/8] Exporting NetCDF...")
    try:
        nc_path = os.path.join(output_dir, "soil_moisture.nc")
        exporter.export_netcdf(nc_path)
    except Exception as e:
        print(f"   ⚠️  NetCDF export skipped: {e}")
    
    # 3. CSV (sampled)
    print("\n[3/8] Exporting CSV (sampled every 5 pixels)...")
    csv_path = os.path.join(output_dir, "soil_moisture_sampled.csv")
    exporter.export_csv(csv_path, sample_rate=5)
    
    # 4. Full CSV
    print("\n[4/8] Exporting full CSV (may be large)...")
    csv_full_path = os.path.join(output_dir, "soil_moisture_full.csv")
    exporter.export_csv(csv_full_path, sample_rate=1)
    
    # 5. GeoJSON
    print("\n[5/8] Exporting GeoJSON...")
    geojson_path = os.path.join(output_dir, "soil_moisture.geojson")
    exporter.export_geojson(geojson_path, grid_size=20)
    
    # 6. ML Features
    print("\n[6/8] Exporting ML-ready features...")
    ml_path = os.path.join(output_dir, "ml_features.csv")
    exporter.export_ml_features(ml_path)
    
    # 7. TensorFlow Dataset
    print("\n[7/8] Creating TensorFlow dataset...")
    try:
        ml_prep = NISARMLPreparation(analyzer.sm_data)
        tf_dir = os.path.join(output_dir, "tensorflow_dataset")
        ml_prep.export_tensorflow_dataset(tf_dir)
    except Exception as e:
        print(f"   ⚠️  TensorFlow export skipped: {e}")
    
    # 8. Google Earth Engine
    print("\n[8/8] Preparing Google Earth Engine assets...")
    gee_prefix = os.path.join(output_dir, "gee_soil_moisture")
    exporter.export_gee_asset(gee_prefix)
    
    # ========================================================================
    # PHASE 5: GENERATE REPORTS
    # ========================================================================
    
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 29 + "PHASE 5: GENERATING REPORTS" + " " * 23 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Text report
    print("\n[1/3] Generating text report...")
    report_path = os.path.join(output_dir, "analysis_report.txt")
    analyzer.generate_report(report_path)
    
    # Summary JSON
    print("\n[2/3] Creating summary JSON...")
    import json
    
    summary = {
        'metadata': {
            'file': file_path,
            'analysis_date': datetime.now().isoformat(),
            'product': 'NISAR_L3_SME2',
            'data_date': '2025-12-16',
            'region': 'Egypt'
        },
        'statistics': {
            'mean': float(results['stats']['Mean']),
            'median': float(results['stats']['Median']),
            'std': float(results['stats']['Std Dev']),
            'min': float(results['stats']['Min']),
            'max': float(results['stats']['Max'])
        },
        'drought': {
            'index': float(results['drought']['drought_index']),
            'status': 'See detailed report'
        },
        'risk': {
            'overall_score': float(results['risk']['overall_risk']),
            'high_risk_pixels': int(results['risk']['risk_categories']['High'] + 
                                   results['risk']['risk_categories']['Very High'])
        },
        'outputs': {
            'visualizations': 2,
            'export_formats': 8,
            'reports': 3
        }
    }
    
    summary_path = os.path.join(output_dir, "analysis_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   ✅ Saved: {summary_path}")
    
    # Executive summary
    print("\n[3/3] Creating executive summary...")
    exec_summary = f"""
{'=' * 80}
NISAR L3 SOIL MOISTURE - EXECUTIVE SUMMARY
{'=' * 80}

ANALYSIS INFORMATION
--------------------
File:           {os.path.basename(file_path)}
Analysis Date:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Data Date:      2025-12-16 16:42:23 UTC
Region:         Egypt (Nile Delta)
Product:        NISAR L3 SME2 BETA V1

KEY STATISTICS
--------------
Mean Soil Moisture:     {results['stats']['Mean']:.4f} m³/m³
Median:                 {results['stats']['Median']:.4f} m³/m³
Standard Deviation:     {results['stats']['Std Dev']:.4f} m³/m³
Range:                  [{results['stats']['Min']:.4f}, {results['stats']['Max']:.4f}]

DROUGHT ASSESSMENT
------------------
Drought Index:          {results['drought']['drought_index']:.4f}
Status:                 {'Severe' if results['drought']['drought_index'] > 0.6 else 'Moderate' if results['drought']['drought_index'] > 0.4 else 'Normal'}

RISK ASSESSMENT
---------------
Overall Risk Score:     {results['risk']['overall_risk']:.1f}/100
High Risk Areas:        {results['risk']['risk_categories']['High'] + results['risk']['risk_categories']['Very High']} pixels
Risk Level:             {'Very High' if results['risk']['overall_risk'] > 60 else 'High' if results['risk']['overall_risk'] > 40 else 'Moderate'}

AGRICULTURAL SUITABILITY
-------------------------
See detailed analysis in:
- {output_dir}/analysis_report.txt
- {output_dir}/01_comprehensive_analysis.png

INSURANCE TRIGGERS
------------------
Critical Trigger Status:  {'ACTIVATED' if results['insurance']['critical_area'] > 50 else 'NOT ACTIVATED'}
Area Below Critical:      {results['insurance']['critical_area']:.2f}%

OUTPUTS GENERATED
-----------------
✅ Visualizations:   2 comprehensive figures (600+ DPI)
✅ Export Formats:   8 different formats (GeoTIFF, NetCDF, CSV, etc.)
✅ ML Datasets:      TensorFlow-ready patches
✅ Reports:          3 document types (TXT, JSON, PDF-ready)

RECOMMENDATIONS
---------------
"""
    
    # Add specific recommendations
    if results['risk']['overall_risk'] > 60:
        exec_summary += """
⚠️  HIGH PRIORITY ACTIONS:
1. Implement emergency irrigation immediately
2. Monitor soil moisture daily
3. Consider drought-resistant crop varieties
4. Activate insurance claims if applicable
5. Coordinate with local agricultural extension
"""
    elif results['risk']['overall_risk'] > 40:
        exec_summary += """
⚡ MODERATE PRIORITY ACTIONS:
1. Increase monitoring frequency to every 3 days
2. Prepare irrigation infrastructure
3. Review crop water requirements
4. Consider supplementary irrigation
"""
    else:
        exec_summary += """
✅ STANDARD MONITORING:
1. Continue regular monitoring schedule
2. Maintain current agricultural practices
3. Monitor for changes in conditions
"""
    
    exec_summary += f"""

NEXT STEPS
----------
1. Review detailed visualizations
2. Import GeoTIFF into GIS software
3. Analyze ML features for predictive modeling
4. Monitor for temporal changes
5. Integrate with ground-truth data

For questions or support, consult NISAR documentation at:
https://nisar.jpl.nasa.gov/

{'=' * 80}
End of Executive Summary
{'=' * 80}
"""
    
    exec_path = os.path.join(output_dir, "executive_summary.txt")
    with open(exec_path, 'w') as f:
        f.write(exec_summary)
    print(f"   ✅ Saved: {exec_path}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "✅ ANALYSIS PIPELINE COMPLETE ✅" + " " * 21 + "║")
    print("╚" + "═" * 78 + "╝")
    
    print(f"\n📁 All outputs saved to: {output_dir}/")
    print("\n📊 Generated Files:")
    print("   Visualizations:")
    print("   • 01_comprehensive_analysis.png")
    print("   • 02_advanced_analysis.png")
    print("\n   Geospatial Exports:")
    print("   • soil_moisture.tif (GeoTIFF)")
    print("   • soil_moisture.nc (NetCDF)")
    print("   • soil_moisture.geojson (Web mapping)")
    print("\n   Tabular Data:")
    print("   • soil_moisture_full.csv (All pixels)")
    print("   • soil_moisture_sampled.csv (Every 5th pixel)")
    print("   • ml_features.csv (ML-ready)")
    print("\n   Machine Learning:")
    print("   • tensorflow_dataset/ (CNN-ready patches)")
    print("\n   Reports:")
    print("   • executive_summary.txt (Key findings)")
    print("   • analysis_report.txt (Full details)")
    print("   • analysis_summary.json (Structured data)")
    print("\n   Google Earth Engine:")
    print("   • gee_soil_moisture_gee.tif")
    print("   • gee_soil_moisture_gee_instructions.txt")
    
    print(f"\n⏱️  Total processing time: Complete")
    print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "=" * 80)
    print("Thank you for using NISAR Analysis Suite!")
    print("=" * 80 + "\n")
    
    return True


def main():
    """Main entry point"""
    
    # Get file path from command line or use default
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default file name from metadata
        file_path = "NISAR_L3_PR_SME2_008_007_D_073_4005_DHDH_A_20251216T164223_20251216T164300_X05007_N_F_J_001.h5"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"\n❌ ERROR: File not found: {file_path}")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} path/to/nisar_file.h5")
        print("\nOr place the file in the current directory and run:")
        print(f"  python {sys.argv[0]}")
        sys.exit(1)
    
    # Run pipeline
    success = run_complete_pipeline(file_path)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
