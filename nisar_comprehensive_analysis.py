"""
NISAR L3 Soil Moisture Comprehensive Analysis Suite
====================================================
Advanced analysis for NISAR_L3_SME2 products
Author: Analysis Framework for Remote Sensing
Date: February 2026

Analysis Coverage:
1. Data Exploration & QA
2. Statistical Analysis
3. Spatial Analysis
4. Temporal Analysis
5. Hydrological Indices
6. Agricultural Applications
7. Drought Monitoring
8. Anomaly Detection
9. Export & Visualization
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, ndimage
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Configuration
FILE_PATH = "NISAR_L3_PR_SME2_008_007_D_073_4005_DHDH_A_20251216T164223_20251216T164300_X05007_N_F_J_001.h5"

class NISARSoilMoistureAnalyzer:
    """Comprehensive analysis class for NISAR soil moisture data"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.sm_data = None
        self.lat = None
        self.lon = None
        self.metadata = {}
        
    def load_data(self):
        """Load soil moisture data and metadata"""
        print("=" * 70)
        print("LOADING NISAR DATA")
        print("=" * 70)
        
        try:
            with h5py.File(self.file_path, 'r') as f:
                # Explore structure first
                print("\n📁 HDF5 Structure:")
                self._print_structure(f)
                
                # Load soil moisture (adjust path based on actual structure)
                # Common paths:
                possible_paths = [
                    '/science/LSAR/SME2/data/soil_moisture',
                    '/LSAR/SME2/data/soil_moisture',
                    '/science/soil_moisture',
                    '/soil_moisture'
                ]
                
                for path in possible_paths:
                    try:
                        self.sm_data = f[path][:]
                        print(f"\n✅ Loaded data from: {path}")
                        break
                    except:
                        continue
                
                if self.sm_data is None:
                    print("⚠️  Standard paths not found. Manual inspection needed.")
                    return False
                
                # Load coordinates
                try:
                    self.lat = f['/science/LSAR/SME2/latitude'][:]
                    self.lon = f['/science/LSAR/SME2/longitude'][:]
                except:
                    print("⚠️  Coordinates not found in standard location")
                
                # Clean data
                self.sm_data = np.where(self.sm_data < -9990, np.nan, self.sm_data)
                self.sm_data = np.where(self.sm_data > 1.0, np.nan, self.sm_data)
                
                print(f"\n✅ Data shape: {self.sm_data.shape}")
                print(f"✅ Valid pixels: {np.sum(~np.isnan(self.sm_data))}")
                print(f"✅ Data range: [{np.nanmin(self.sm_data):.4f}, {np.nanmax(self.sm_data):.4f}]")
                
                return True
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def _print_structure(self, h5file, path='/', indent=0):
        """Recursively print HDF5 structure"""
        if indent > 3:  # Limit depth
            return
        for key in h5file[path].keys():
            item = h5file[path + key]
            print('  ' * indent + f'├── {key}')
            if isinstance(item, h5py.Group):
                self._print_structure(h5file, path + key + '/', indent + 1)
    
    # =========================================================================
    # ANALYSIS 1: BASIC STATISTICS
    # =========================================================================
    
    def basic_statistics(self):
        """Comprehensive statistical analysis"""
        print("\n" + "=" * 70)
        print("ANALYSIS 1: BASIC STATISTICS")
        print("=" * 70)
        
        valid_data = self.sm_data[~np.isnan(self.sm_data)]
        
        stats_dict = {
            'Count': len(valid_data),
            'Mean': np.mean(valid_data),
            'Median': np.median(valid_data),
            'Std Dev': np.std(valid_data),
            'Min': np.min(valid_data),
            'Max': np.max(valid_data),
            'Q1 (25%)': np.percentile(valid_data, 25),
            'Q3 (75%)': np.percentile(valid_data, 75),
            'IQR': np.percentile(valid_data, 75) - np.percentile(valid_data, 25),
            'Skewness': stats.skew(valid_data),
            'Kurtosis': stats.kurtosis(valid_data),
            'CV (%)': (np.std(valid_data) / np.mean(valid_data)) * 100
        }
        
        print("\n📊 Descriptive Statistics:")
        for key, value in stats_dict.items():
            print(f"  {key:.<25} {value:.6f}")
        
        # Distribution test
        _, p_value = stats.normaltest(valid_data)
        print(f"\n  Normality Test (p-value): {p_value:.6f}")
        print(f"  Distribution: {'Normal' if p_value > 0.05 else 'Non-normal'}")
        
        return stats_dict
    
    # =========================================================================
    # ANALYSIS 2: SPATIAL ANALYSIS
    # =========================================================================
    
    def spatial_analysis(self):
        """Spatial patterns and autocorrelation"""
        print("\n" + "=" * 70)
        print("ANALYSIS 2: SPATIAL ANALYSIS")
        print("=" * 70)
        
        # Spatial statistics
        print("\n🗺️  Spatial Characteristics:")
        
        # Spatial variability by quadrant
        h, w = self.sm_data.shape
        quadrants = {
            'NW': self.sm_data[:h//2, :w//2],
            'NE': self.sm_data[:h//2, w//2:],
            'SW': self.sm_data[h//2:, :w//2],
            'SE': self.sm_data[h//2:, w//2:]
        }
        
        print("\n  Regional Statistics:")
        for region, data in quadrants.items():
            mean_val = np.nanmean(data)
            print(f"    {region}: {mean_val:.4f} m³/m³")
        
        # Gradient analysis
        if not np.all(np.isnan(self.sm_data)):
            grad_y, grad_x = np.gradient(np.nan_to_num(self.sm_data, nan=0))
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            print(f"\n  Gradient Magnitude (spatial variability):")
            print(f"    Mean: {np.nanmean(gradient_magnitude):.6f}")
            print(f"    Max: {np.nanmax(gradient_magnitude):.6f}")
        
        # Hotspot detection
        threshold_high = np.nanpercentile(self.sm_data, 90)
        threshold_low = np.nanpercentile(self.sm_data, 10)
        
        hotspots = np.sum(self.sm_data > threshold_high)
        coldspots = np.sum(self.sm_data < threshold_low)
        
        print(f"\n  Extreme Value Zones:")
        print(f"    High moisture hotspots (>P90): {hotspots} pixels")
        print(f"    Low moisture coldspots (<P10): {coldspots} pixels")
        
        return {
            'quadrants': quadrants,
            'gradient': gradient_magnitude,
            'hotspots': hotspots,
            'coldspots': coldspots
        }
    
    # =========================================================================
    # ANALYSIS 3: HYDROLOGICAL INDICES
    # =========================================================================
    
    def hydrological_indices(self):
        """Calculate key hydrological indicators"""
        print("\n" + "=" * 70)
        print("ANALYSIS 3: HYDROLOGICAL INDICES")
        print("=" * 70)
        
        valid_mask = ~np.isnan(self.sm_data)
        total_pixels = np.sum(valid_mask)
        
        # Moisture classification (based on typical soil types)
        dry = (self.sm_data < 0.15) & valid_mask
        optimal = (self.sm_data >= 0.15) & (self.sm_data <= 0.35) & valid_mask
        saturated = (self.sm_data > 0.35) & valid_mask
        
        print("\n💧 Soil Moisture Classification:")
        print(f"  Dry (<15%):        {np.sum(dry)} pixels ({np.sum(dry)/total_pixels*100:.2f}%)")
        print(f"  Optimal (15-35%):  {np.sum(optimal)} pixels ({np.sum(optimal)/total_pixels*100:.2f}%)")
        print(f"  Saturated (>35%):  {np.sum(saturated)} pixels ({np.sum(saturated)/total_pixels*100:.2f}%)")
        
        # Soil Water Deficit Index (SWDI)
        field_capacity = 0.35  # Typical field capacity
        wilting_point = 0.15   # Typical wilting point
        
        swdi = (field_capacity - self.sm_data) / (field_capacity - wilting_point)
        swdi = np.clip(swdi, 0, 1)
        
        print(f"\n  Soil Water Deficit Index (SWDI):")
        print(f"    Mean SWDI: {np.nanmean(swdi):.4f}")
        print(f"    Deficit area (SWDI>0.6): {np.sum(swdi > 0.6)/total_pixels*100:.2f}%")
        
        # Plant Available Water (PAW)
        paw = self.sm_data - wilting_point
        paw = np.clip(paw, 0, None)
        
        print(f"\n  Plant Available Water:")
        print(f"    Mean PAW: {np.nanmean(paw):.4f} m³/m³")
        print(f"    Critical PAW (<5%): {np.sum(paw < 0.05)/total_pixels*100:.2f}%")
        
        # Water Stress Index
        wsi = 1 - (paw / (field_capacity - wilting_point))
        wsi = np.clip(wsi, 0, 1)
        
        print(f"\n  Water Stress Index (0=no stress, 1=max stress):")
        print(f"    Mean WSI: {np.nanmean(wsi):.4f}")
        print(f"    High stress (WSI>0.7): {np.sum(wsi > 0.7)/total_pixels*100:.2f}%")
        
        return {
            'swdi': swdi,
            'paw': paw,
            'wsi': wsi,
            'classification': {'dry': dry, 'optimal': optimal, 'saturated': saturated}
        }
    
    # =========================================================================
    # ANALYSIS 4: AGRICULTURAL APPLICATIONS
    # =========================================================================
    
    def agricultural_analysis(self):
        """Agricultural stress and crop suitability analysis"""
        print("\n" + "=" * 70)
        print("ANALYSIS 4: AGRICULTURAL APPLICATIONS")
        print("=" * 70)
        
        # Crop-specific moisture requirements (examples)
        crop_requirements = {
            'Wheat': (0.20, 0.30),
            'Rice': (0.30, 0.45),
            'Cotton': (0.15, 0.25),
            'Soybean': (0.20, 0.35),
            'Maize': (0.22, 0.32)
        }
        
        print("\n🌾 Crop Suitability Analysis:")
        valid_mask = ~np.isnan(self.sm_data)
        total_pixels = np.sum(valid_mask)
        
        for crop, (min_req, max_req) in crop_requirements.items():
            suitable = ((self.sm_data >= min_req) & 
                       (self.sm_data <= max_req) & 
                       valid_mask)
            percent = np.sum(suitable) / total_pixels * 100
            
            status = "✅ Optimal" if percent > 60 else "⚠️  Suboptimal" if percent > 30 else "❌ Poor"
            print(f"  {crop:.<15} {percent:>6.2f}% suitable  {status}")
        
        # Irrigation requirement zones
        print("\n💦 Irrigation Requirement Zones:")
        
        no_irrigation = self.sm_data > 0.30
        light_irrigation = (self.sm_data >= 0.20) & (self.sm_data <= 0.30)
        moderate_irrigation = (self.sm_data >= 0.15) & (self.sm_data < 0.20)
        heavy_irrigation = self.sm_data < 0.15
        
        print(f"  No irrigation needed:      {np.sum(no_irrigation)/total_pixels*100:>6.2f}%")
        print(f"  Light irrigation:          {np.sum(light_irrigation)/total_pixels*100:>6.2f}%")
        print(f"  Moderate irrigation:       {np.sum(moderate_irrigation)/total_pixels*100:>6.2f}%")
        print(f"  Heavy irrigation needed:   {np.sum(heavy_irrigation)/total_pixels*100:>6.2f}%")
        
        # Growing season suitability
        print(f"\n🌱 Growing Season Indicators:")
        suitable_germination = np.sum((self.sm_data > 0.15) & (self.sm_data < 0.40)) / total_pixels * 100
        print(f"  Germination suitable:      {suitable_germination:>6.2f}%")
        
        return {
            'crop_suitability': crop_requirements,
            'irrigation_zones': {
                'none': no_irrigation,
                'light': light_irrigation,
                'moderate': moderate_irrigation,
                'heavy': heavy_irrigation
            }
        }
    
    # =========================================================================
    # ANALYSIS 5: DROUGHT MONITORING
    # =========================================================================
    
    def drought_monitoring(self):
        """Drought severity assessment"""
        print("\n" + "=" * 70)
        print("ANALYSIS 5: DROUGHT MONITORING")
        print("=" * 70)
        
        valid_mask = ~np.isnan(self.sm_data)
        total_pixels = np.sum(valid_mask)
        
        # Drought categories (based on percentiles)
        p20 = np.nanpercentile(self.sm_data, 20)
        p10 = np.nanpercentile(self.sm_data, 10)
        p5 = np.nanpercentile(self.sm_data, 5)
        
        print("\n🌵 Drought Severity Classification:")
        print(f"  Thresholds: P20={p20:.4f}, P10={p10:.4f}, P5={p5:.4f}")
        
        no_drought = self.sm_data > p20
        moderate_drought = (self.sm_data <= p20) & (self.sm_data > p10)
        severe_drought = (self.sm_data <= p10) & (self.sm_data > p5)
        extreme_drought = self.sm_data <= p5
        
        print(f"\n  No drought:         {np.sum(no_drought)/total_pixels*100:>6.2f}%")
        print(f"  Moderate drought:   {np.sum(moderate_drought)/total_pixels*100:>6.2f}%")
        print(f"  Severe drought:     {np.sum(severe_drought)/total_pixels*100:>6.2f}%")
        print(f"  Extreme drought:    {np.sum(extreme_drought)/total_pixels*100:>6.2f}%")
        
        # Overall drought index
        mean_sm = np.nanmean(self.sm_data)
        drought_index = 1 - (mean_sm / 0.35)  # normalized to field capacity
        
        print(f"\n  Overall Drought Index: {drought_index:.4f}")
        if drought_index < 0.3:
            status = "✅ No significant drought"
        elif drought_index < 0.5:
            status = "⚠️  Moderate drought conditions"
        elif drought_index < 0.7:
            status = "❌ Severe drought conditions"
        else:
            status = "🚨 Extreme drought conditions"
        
        print(f"  Status: {status}")
        
        return {
            'drought_index': drought_index,
            'categories': {
                'none': no_drought,
                'moderate': moderate_drought,
                'severe': severe_drought,
                'extreme': extreme_drought
            }
        }
    
    # =========================================================================
    # ANALYSIS 6: ANOMALY DETECTION
    # =========================================================================
    
    def anomaly_detection(self):
        """Statistical anomaly detection"""
        print("\n" + "=" * 70)
        print("ANALYSIS 6: ANOMALY DETECTION")
        print("=" * 70)
        
        # Z-score based anomalies
        mean_sm = np.nanmean(self.sm_data)
        std_sm = np.nanstd(self.sm_data)
        
        z_scores = (self.sm_data - mean_sm) / std_sm
        
        anomalies_high = np.abs(z_scores) > 2
        anomalies_extreme = np.abs(z_scores) > 3
        
        valid_mask = ~np.isnan(self.sm_data)
        total_pixels = np.sum(valid_mask)
        
        print("\n⚡ Statistical Anomalies:")
        print(f"  Moderate anomalies (|z|>2): {np.sum(anomalies_high)/total_pixels*100:.2f}%")
        print(f"  Extreme anomalies (|z|>3):  {np.sum(anomalies_extreme)/total_pixels*100:.2f}%")
        
        # Local outliers (compared to neighbors)
        smoothed = ndimage.uniform_filter(np.nan_to_num(self.sm_data, nan=mean_sm), size=5)
        local_diff = np.abs(self.sm_data - smoothed)
        
        local_outliers = local_diff > 2 * std_sm
        
        print(f"  Local outliers (vs neighbors): {np.sum(local_outliers)/total_pixels*100:.2f}%")
        
        return {
            'z_scores': z_scores,
            'anomalies': anomalies_high,
            'extreme_anomalies': anomalies_extreme,
            'local_outliers': local_outliers
        }
    
    # =========================================================================
    # ANALYSIS 7: CLUSTERING ANALYSIS
    # =========================================================================
    
    def moisture_clustering(self, n_clusters=5):
        """K-means clustering of moisture zones"""
        print("\n" + "=" * 70)
        print("ANALYSIS 7: MOISTURE ZONE CLUSTERING")
        print("=" * 70)
        
        # Prepare data for clustering
        valid_mask = ~np.isnan(self.sm_data)
        valid_data = self.sm_data[valid_mask].reshape(-1, 1)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(valid_data)
        
        # Create cluster map
        cluster_map = np.full(self.sm_data.shape, -1, dtype=int)
        cluster_map[valid_mask] = labels
        
        print(f"\n🎯 Identified {n_clusters} Moisture Zones:")
        
        cluster_centers = kmeans.cluster_centers_.flatten()
        sorted_indices = np.argsort(cluster_centers)
        
        zone_names = ['Very Dry', 'Dry', 'Moderate', 'Moist', 'Very Moist']
        
        for i, idx in enumerate(sorted_indices):
            center = cluster_centers[idx]
            count = np.sum(labels == idx)
            percent = count / len(labels) * 100
            
            zone_name = zone_names[i] if i < len(zone_names) else f"Zone {i+1}"
            print(f"  {zone_name:.<20} SM={center:.4f} m³/m³  ({percent:.2f}%)")
        
        return {
            'cluster_map': cluster_map,
            'centers': cluster_centers,
            'labels': labels
        }
    
    # =========================================================================
    # ANALYSIS 8: PARAMETRIC INSURANCE TRIGGERS
    # =========================================================================
    
    def insurance_triggers(self):
        """Parametric insurance trigger analysis"""
        print("\n" + "=" * 70)
        print("ANALYSIS 8: PARAMETRIC INSURANCE TRIGGERS")
        print("=" * 70)
        
        valid_mask = ~np.isnan(self.sm_data)
        total_pixels = np.sum(valid_mask)
        
        # Define trigger thresholds
        triggers = {
            'Critical Drought': 0.10,
            'Severe Drought': 0.12,
            'Moderate Drought': 0.15,
            'Mild Stress': 0.18,
            'Low Moisture': 0.20
        }
        
        print("\n💰 Insurance Trigger Analysis:")
        print("\n  Trigger Level           Threshold    Area Affected  Status")
        print("  " + "-" * 65)
        
        for trigger_name, threshold in triggers.items():
            affected = self.sm_data < threshold
            percent = np.sum(affected) / total_pixels * 100
            
            if percent > 50:
                status = "🚨 TRIGGER ACTIVATED"
            elif percent > 30:
                status = "⚠️  WARNING"
            elif percent > 15:
                status = "⚡ WATCH"
            else:
                status = "✅ Normal"
            
            print(f"  {trigger_name:.<22} <{threshold:.2f}     {percent:>6.2f}%      {status}")
        
        # Payout calculation example
        print("\n  Example Payout Structure:")
        print(f"    If >30% area below 0.12: ₹5000/hectare")
        print(f"    If >50% area below 0.10: ₹10000/hectare")
        
        critical_area = np.sum(self.sm_data < 0.10) / total_pixels * 100
        
        if critical_area > 50:
            print(f"\n  🚨 PAYOUT RECOMMENDED: Critical drought threshold exceeded!")
        
        return {
            'triggers': triggers,
            'critical_area': critical_area
        }
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def create_visualizations(self):
        """Generate comprehensive visualization suite"""
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Main soil moisture map
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(self.sm_data, cmap='RdYlBu', vmin=0, vmax=0.5)
        ax1.set_title('Soil Moisture Map', fontsize=14, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, label='Soil Moisture (m³/m³)', fraction=0.046)
        
        # 2. Histogram
        ax2 = plt.subplot(2, 3, 2)
        valid_data = self.sm_data[~np.isnan(self.sm_data)]
        ax2.hist(valid_data, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(valid_data), color='red', linestyle='--', label='Mean')
        ax2.axvline(np.median(valid_data), color='green', linestyle='--', label='Median')
        ax2.set_xlabel('Soil Moisture (m³/m³)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Drought zones
        ax3 = plt.subplot(2, 3, 3)
        drought_map = np.full(self.sm_data.shape, 0)
        p20 = np.nanpercentile(self.sm_data, 20)
        p10 = np.nanpercentile(self.sm_data, 10)
        p5 = np.nanpercentile(self.sm_data, 5)
        
        drought_map[self.sm_data <= p5] = 3
        drought_map[(self.sm_data > p5) & (self.sm_data <= p10)] = 2
        drought_map[(self.sm_data > p10) & (self.sm_data <= p20)] = 1
        
        im3 = ax3.imshow(drought_map, cmap='YlOrRd', vmin=0, vmax=3)
        ax3.set_title('Drought Severity Zones', fontsize=14, fontweight='bold')
        ax3.axis('off')
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046)
        cbar3.set_ticks([0.375, 1.125, 1.875, 2.625])
        cbar3.set_ticklabels(['None', 'Moderate', 'Severe', 'Extreme'])
        
        # 4. Box plot by quantiles
        ax4 = plt.subplot(2, 3, 4)
        data_quantiles = [
            valid_data[valid_data <= np.percentile(valid_data, 25)],
            valid_data[(valid_data > np.percentile(valid_data, 25)) & 
                      (valid_data <= np.percentile(valid_data, 50))],
            valid_data[(valid_data > np.percentile(valid_data, 50)) & 
                      (valid_data <= np.percentile(valid_data, 75))],
            valid_data[valid_data > np.percentile(valid_data, 75)]
        ]
        ax4.boxplot(data_quantiles, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        ax4.set_ylabel('Soil Moisture (m³/m³)')
        ax4.set_title('Quartile Distribution', fontsize=14, fontweight='bold')
        ax4.grid(alpha=0.3)
        
        # 5. Spatial gradient
        ax5 = plt.subplot(2, 3, 5)
        grad_y, grad_x = np.gradient(np.nan_to_num(self.sm_data, nan=0))
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        im5 = ax5.imshow(gradient_mag, cmap='viridis')
        ax5.set_title('Spatial Variability', fontsize=14, fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, label='Gradient Magnitude', fraction=0.046)
        
        # 6. Classification zones
        ax6 = plt.subplot(2, 3, 6)
        classification = np.full(self.sm_data.shape, 0)
        classification[self.sm_data < 0.15] = 0  # Dry
        classification[(self.sm_data >= 0.15) & (self.sm_data <= 0.35)] = 1  # Optimal
        classification[self.sm_data > 0.35] = 2  # Saturated
        
        im6 = ax6.imshow(classification, cmap='RdYlGn', vmin=0, vmax=2)
        ax6.set_title('Agricultural Classification', fontsize=14, fontweight='bold')
        ax6.axis('off')
        cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046)
        cbar6.set_ticks([0.33, 1, 1.67])
        cbar6.set_ticklabels(['Dry', 'Optimal', 'Saturated'])
        
        plt.suptitle('NISAR L3 Soil Moisture Comprehensive Analysis\n' + 
                     f'Date: 2025-12-16 | Location: Egypt Region',
                     fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        return fig
    
    # =========================================================================
    # EXPORT FUNCTIONS
    # =========================================================================
    
    def export_geotiff(self, output_path="nisar_soil_moisture.tif"):
        """Export to GeoTIFF format"""
        try:
            import rasterio
            from rasterio.transform import from_bounds
            
            # Get bounds from metadata (if available)
            # From metadata: Egypt region approximately
            min_lon, max_lon = 29.12, 32.41
            min_lat, max_lat = 29.38, 32.25
            
            transform = from_bounds(min_lon, min_lat, max_lon, max_lat,
                                   self.sm_data.shape[1], self.sm_data.shape[0])
            
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=self.sm_data.shape[0],
                width=self.sm_data.shape[1],
                count=1,
                dtype=self.sm_data.dtype,
                crs='EPSG:4326',
                transform=transform,
                nodata=-9999
            ) as dst:
                dst.write(self.sm_data, 1)
            
            print(f"✅ GeoTIFF exported: {output_path}")
            return True
            
        except ImportError:
            print("⚠️  rasterio not installed. Install with: pip install rasterio")
            return False
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return False
    
    def generate_report(self, output_path="nisar_analysis_report.txt"):
        """Generate text report"""
        with open(output_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("NISAR L3 SOIL MOISTURE ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"File: {self.file_path}\n")
            f.write(f"Analysis Date: 2026-02-11\n")
            f.write(f"Data Date: 2025-12-16\n\n")
            
            # Add statistics
            f.write("BASIC STATISTICS\n")
            f.write("-" * 70 + "\n")
            valid_data = self.sm_data[~np.isnan(self.sm_data)]
            f.write(f"Mean: {np.mean(valid_data):.4f} m³/m³\n")
            f.write(f"Std: {np.std(valid_data):.4f} m³/m³\n")
            f.write(f"Range: [{np.min(valid_data):.4f}, {np.max(valid_data):.4f}]\n\n")
            
        print(f"✅ Report generated: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete analysis suite"""
    
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "NISAR SOIL MOISTURE ANALYSIS SUITE" + " " * 19 + "║")
    print("║" + " " * 20 + "Comprehensive Analysis Tool" + " " * 21 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")
    
    # Initialize analyzer
    analyzer = NISARSoilMoistureAnalyzer(FILE_PATH)
    
    # Load data
    if not analyzer.load_data():
        print("\n❌ Failed to load data. Please check file path and structure.")
        return
    
    # Run all analyses
    try:
        # 1. Basic statistics
        stats = analyzer.basic_statistics()
        
        # 2. Spatial analysis
        spatial = analyzer.spatial_analysis()
        
        # 3. Hydrological indices
        hydro = analyzer.hydrological_indices()
        
        # 4. Agricultural analysis
        agri = analyzer.agricultural_analysis()
        
        # 5. Drought monitoring
        drought = analyzer.drought_monitoring()
        
        # 6. Anomaly detection
        anomalies = analyzer.anomaly_detection()
        
        # 7. Clustering
        clusters = analyzer.moisture_clustering(n_clusters=5)
        
        # 8. Insurance triggers
        insurance = analyzer.insurance_triggers()
        
        # 9. Generate visualizations
        print("\n" + "=" * 70)
        print("CREATING VISUALIZATIONS...")
        print("=" * 70)
        fig = analyzer.create_visualizations()
        plt.savefig('nisar_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: nisar_analysis_comprehensive.png")
        
        # 10. Export data
        print("\n" + "=" * 70)
        print("EXPORTING DATA...")
        print("=" * 70)
        analyzer.export_geotiff()
        analyzer.generate_report()
        
        print("\n" + "=" * 70)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 70)
        print("\nGenerated files:")
        print("  • nisar_analysis_comprehensive.png")
        print("  • nisar_soil_moisture.tif")
        print("  • nisar_analysis_report.txt")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
