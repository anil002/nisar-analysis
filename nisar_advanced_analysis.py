"""
NISAR Time Series & Advanced Analysis Module
============================================
Multi-temporal analysis, change detection, and advanced metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate
from datetime import datetime, timedelta

class NISARAdvancedAnalysis:
    """Advanced analysis for NISAR soil moisture"""
    
    def __init__(self, sm_data):
        self.sm_data = sm_data
        
    # =========================================================================
    # ANALYSIS 9: CHANGE DETECTION (for multi-temporal data)
    # =========================================================================
    
    def change_detection(self, previous_sm=None):
        """Detect changes between two time periods"""
        print("\n" + "=" * 70)
        print("ANALYSIS 9: CHANGE DETECTION")
        print("=" * 70)
        
        if previous_sm is None:
            # Simulate previous observation for demo
            print("\n⚠️  Single timestamp - simulating previous observation")
            previous_sm = self.sm_data + np.random.normal(0, 0.05, self.sm_data.shape)
        
        # Calculate change
        change = self.sm_data - previous_sm
        
        valid_mask = ~np.isnan(change)
        total_pixels = np.sum(valid_mask)
        
        print("\n📊 Moisture Change Statistics:")
        print(f"  Mean change: {np.nanmean(change):.6f} m³/m³")
        print(f"  Std change: {np.nanstd(change):.6f} m³/m³")
        print(f"  Max increase: {np.nanmax(change):.6f} m³/m³")
        print(f"  Max decrease: {np.nanmin(change):.6f} m³/m³")
        
        # Classify changes
        significant_threshold = np.nanstd(change)
        
        sig_increase = change > significant_threshold
        sig_decrease = change < -significant_threshold
        no_change = np.abs(change) <= significant_threshold
        
        print(f"\n  Significant increase:  {np.sum(sig_increase)/total_pixels*100:.2f}%")
        print(f"  Significant decrease:  {np.sum(sig_decrease)/total_pixels*100:.2f}%")
        print(f"  No significant change: {np.sum(no_change)/total_pixels*100:.2f}%")
        
        # Hotspot analysis
        extreme_drying = change < -2 * significant_threshold
        extreme_wetting = change > 2 * significant_threshold
        
        print(f"\n  Extreme drying zones:  {np.sum(extreme_drying)}")
        print(f"  Extreme wetting zones: {np.sum(extreme_wetting)}")
        
        return {
            'change_map': change,
            'significant_increase': sig_increase,
            'significant_decrease': sig_decrease
        }
    
    # =========================================================================
    # ANALYSIS 10: TEXTURE ANALYSIS
    # =========================================================================
    
    def texture_analysis(self):
        """Analyze spatial texture and patterns"""
        print("\n" + "=" * 70)
        print("ANALYSIS 10: TEXTURE ANALYSIS")
        print("=" * 70)
        
        # Gray Level Co-occurrence Matrix (GLCM) features
        from scipy.ndimage import uniform_filter
        
        # Normalize to 0-255 for texture analysis
        sm_normalized = (self.sm_data - np.nanmin(self.sm_data)) / (np.nanmax(self.sm_data) - np.nanmin(self.sm_data))
        sm_normalized = np.nan_to_num(sm_normalized, nan=0) * 255
        sm_normalized = sm_normalized.astype(np.uint8)
        
        # Calculate local statistics
        window_size = 9
        local_mean = uniform_filter(sm_normalized, size=window_size)
        local_std = np.sqrt(uniform_filter(sm_normalized**2, size=window_size) - local_mean**2)
        
        print("\n🎨 Texture Metrics:")
        print(f"  Mean local variability: {np.mean(local_std):.2f}")
        print(f"  Max local variability: {np.max(local_std):.2f}")
        
        # Homogeneity assessment
        homogeneous = local_std < np.percentile(local_std, 25)
        heterogeneous = local_std > np.percentile(local_std, 75)
        
        print(f"\n  Homogeneous areas: {np.sum(homogeneous)/homogeneous.size*100:.2f}%")
        print(f"  Heterogeneous areas: {np.sum(heterogeneous)/heterogeneous.size*100:.2f}%")
        
        # Edge detection
        from scipy.ndimage import sobel
        edges_x = sobel(np.nan_to_num(self.sm_data, nan=0), axis=0)
        edges_y = sobel(np.nan_to_num(self.sm_data, nan=0), axis=1)
        edges = np.hypot(edges_x, edges_y)
        
        strong_edges = edges > np.percentile(edges, 90)
        
        print(f"  Strong boundaries: {np.sum(strong_edges)} pixels")
        
        return {
            'local_std': local_std,
            'edges': edges,
            'homogeneous': homogeneous,
            'heterogeneous': heterogeneous
        }
    
    # =========================================================================
    # ANALYSIS 11: FREQUENCY DOMAIN ANALYSIS
    # =========================================================================
    
    def frequency_analysis(self):
        """Spectral analysis of spatial patterns"""
        print("\n" + "=" * 70)
        print("ANALYSIS 11: FREQUENCY DOMAIN ANALYSIS")
        print("=" * 70)
        
        # Fill NaN for FFT
        sm_filled = np.nan_to_num(self.sm_data, nan=np.nanmean(self.sm_data))
        
        # 2D FFT
        fft2d = np.fft.fft2(sm_filled)
        fft2d_shifted = np.fft.fftshift(fft2d)
        magnitude_spectrum = np.abs(fft2d_shifted)
        
        # Radial profile
        h, w = magnitude_spectrum.shape
        center = (h // 2, w // 2)
        
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(int)
        
        # Calculate radial average
        tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
        nr = np.bincount(r.ravel())
        radial_prof = tbin / nr
        
        print("\n🌊 Spatial Frequency Characteristics:")
        print(f"  Dominant frequencies detected: {len(radial_prof)} scales")
        print(f"  Peak frequency index: {np.argmax(radial_prof[1:]) + 1}")
        
        # Interpret dominant patterns
        peak_freq = np.argmax(radial_prof[1:]) + 1
        if peak_freq < len(radial_prof) / 4:
            pattern = "Large-scale patterns dominant"
        elif peak_freq < len(radial_prof) / 2:
            pattern = "Medium-scale patterns"
        else:
            pattern = "Fine-scale textures dominant"
        
        print(f"  Interpretation: {pattern}")
        
        return {
            'magnitude_spectrum': magnitude_spectrum,
            'radial_profile': radial_prof
        }
    
    # =========================================================================
    # ANALYSIS 12: CONNECTIVITY ANALYSIS
    # =========================================================================
    
    def connectivity_analysis(self):
        """Analyze connectivity of moisture zones"""
        print("\n" + "=" * 70)
        print("ANALYSIS 12: CONNECTIVITY ANALYSIS")
        print("=" * 70)
        
        from scipy import ndimage
        
        # Define dry and wet zones
        threshold_dry = np.nanpercentile(self.sm_data, 25)
        threshold_wet = np.nanpercentile(self.sm_data, 75)
        
        dry_zones = self.sm_data < threshold_dry
        wet_zones = self.sm_data > threshold_wet
        
        # Label connected components
        labeled_dry, num_dry = ndimage.label(dry_zones)
        labeled_wet, num_wet = ndimage.label(wet_zones)
        
        print("\n🔗 Connectivity Metrics:")
        print(f"  Number of distinct dry patches: {num_dry}")
        print(f"  Number of distinct wet patches: {num_wet}")
        
        # Size distribution
        dry_sizes = ndimage.sum(dry_zones, labeled_dry, range(1, num_dry + 1))
        wet_sizes = ndimage.sum(wet_zones, labeled_wet, range(1, num_wet + 1))
        
        if len(dry_sizes) > 0:
            print(f"\n  Dry patch sizes:")
            print(f"    Largest: {np.max(dry_sizes)} pixels")
            print(f"    Mean: {np.mean(dry_sizes):.1f} pixels")
            print(f"    Fragmentation index: {num_dry / np.sum(dry_zones):.6f}")
        
        if len(wet_sizes) > 0:
            print(f"\n  Wet patch sizes:")
            print(f"    Largest: {np.max(wet_sizes)} pixels")
            print(f"    Mean: {np.mean(wet_sizes):.1f} pixels")
            print(f"    Fragmentation index: {num_wet / np.sum(wet_zones):.6f}")
        
        return {
            'labeled_dry': labeled_dry,
            'labeled_wet': labeled_wet,
            'num_patches': {'dry': num_dry, 'wet': num_wet}
        }
    
    # =========================================================================
    # ANALYSIS 13: GEOSTATISTICAL ANALYSIS
    # =========================================================================
    
    def geostatistical_analysis(self, max_lag=50):
        """Variogram and spatial autocorrelation"""
        print("\n" + "=" * 70)
        print("ANALYSIS 13: GEOSTATISTICAL ANALYSIS")
        print("=" * 70)
        
        # Sample data for variogram (for efficiency)
        valid_mask = ~np.isnan(self.sm_data)
        indices = np.where(valid_mask)
        
        if len(indices[0]) > 5000:
            sample_idx = np.random.choice(len(indices[0]), 5000, replace=False)
            y_coords = indices[0][sample_idx]
            x_coords = indices[1][sample_idx]
        else:
            y_coords = indices[0]
            x_coords = indices[1]
        
        values = self.sm_data[y_coords, x_coords]
        
        # Calculate experimental variogram
        lags = []
        semivariances = []
        
        for lag in range(1, min(max_lag, len(values)//10), 5):
            pairs = []
            for i in range(len(values)):
                for j in range(i+1, len(values)):
                    dist = np.sqrt((x_coords[i]-x_coords[j])**2 + (y_coords[i]-y_coords[j])**2)
                    if lag-2.5 <= dist < lag+2.5:
                        pairs.append((values[i] - values[j])**2)
            
            if len(pairs) > 0:
                lags.append(lag)
                semivariances.append(np.mean(pairs) / 2)
        
        if len(lags) > 3:
            print("\n📊 Variogram Analysis:")
            print(f"  Number of lag classes: {len(lags)}")
            print(f"  Max lag distance: {max(lags):.1f} pixels")
            
            # Estimate range (where semivariance plateaus)
            if len(semivariances) > 5:
                sill = np.max(semivariances)
                range_idx = np.argmax(np.array(semivariances) > 0.95 * sill)
                if range_idx > 0:
                    range_val = lags[range_idx]
                    print(f"  Estimated range: ~{range_val:.1f} pixels")
                    print(f"  Sill: {sill:.6f}")
        
        # Moran's I (simplified)
        from scipy.spatial.distance import pdist, squareform
        
        sample_size = min(1000, len(values))
        sample_indices = np.random.choice(len(values), sample_size, replace=False)
        
        sample_values = values[sample_indices]
        sample_x = x_coords[sample_indices]
        sample_y = y_coords[sample_indices]
        
        # Distance matrix
        coords = np.column_stack((sample_x, sample_y))
        dist_matrix = squareform(pdist(coords))
        
        # Weight matrix (inverse distance)
        weight_matrix = np.zeros_like(dist_matrix)
        weight_matrix[dist_matrix > 0] = 1 / dist_matrix[dist_matrix > 0]
        np.fill_diagonal(weight_matrix, 0)
        
        # Moran's I calculation
        n = len(sample_values)
        mean_val = np.mean(sample_values)
        
        numerator = 0
        denominator = 0
        W = np.sum(weight_matrix)
        
        for i in range(n):
            for j in range(n):
                numerator += weight_matrix[i,j] * (sample_values[i] - mean_val) * (sample_values[j] - mean_val)
            denominator += (sample_values[i] - mean_val)**2
        
        morans_i = (n / W) * (numerator / denominator)
        
        print(f"\n  Moran's I: {morans_i:.4f}")
        if morans_i > 0.3:
            print(f"  Interpretation: Strong positive spatial autocorrelation")
        elif morans_i > 0:
            print(f"  Interpretation: Weak positive spatial autocorrelation")
        else:
            print(f"  Interpretation: Negative or no spatial autocorrelation")
        
        return {
            'lags': lags,
            'semivariances': semivariances,
            'morans_i': morans_i
        }
    
    # =========================================================================
    # ANALYSIS 14: EXTREME VALUE ANALYSIS
    # =========================================================================
    
    def extreme_value_analysis(self):
        """Analyze extreme values and their spatial distribution"""
        print("\n" + "=" * 70)
        print("ANALYSIS 14: EXTREME VALUE ANALYSIS")
        print("=" * 70)
        
        valid_data = self.sm_data[~np.isnan(self.sm_data)]
        
        # Identify extremes using different methods
        # Method 1: Percentile-based
        p1 = np.percentile(valid_data, 1)
        p5 = np.percentile(valid_data, 5)
        p95 = np.percentile(valid_data, 95)
        p99 = np.percentile(valid_data, 99)
        
        # Method 2: IQR-based outliers
        q1 = np.percentile(valid_data, 25)
        q3 = np.percentile(valid_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        print("\n🎯 Extreme Value Thresholds:")
        print(f"  P1:  {p1:.4f} m³/m³")
        print(f"  P5:  {p5:.4f} m³/m³")
        print(f"  P95: {p95:.4f} m³/m³")
        print(f"  P99: {p99:.4f} m³/m³")
        print(f"\n  IQR-based bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
        
        # Count extremes
        extreme_low = np.sum(self.sm_data < p5)
        extreme_high = np.sum(self.sm_data > p95)
        outliers_low = np.sum(self.sm_data < lower_bound)
        outliers_high = np.sum(self.sm_data > upper_bound)
        
        valid_total = np.sum(~np.isnan(self.sm_data))
        
        print(f"\n  Extreme low values (<P5): {extreme_low} ({extreme_low/valid_total*100:.2f}%)")
        print(f"  Extreme high values (>P95): {extreme_high} ({extreme_high/valid_total*100:.2f}%)")
        print(f"  Statistical outliers (IQR): {outliers_low + outliers_high}")
        
        # Spatial clustering of extremes
        from scipy import ndimage
        
        extreme_map = np.zeros(self.sm_data.shape)
        extreme_map[self.sm_data < p5] = -1
        extreme_map[self.sm_data > p95] = 1
        
        # Cluster extremes
        low_clusters, num_low = ndimage.label(extreme_map == -1)
        high_clusters, num_high = ndimage.label(extreme_map == 1)
        
        print(f"\n  Spatial clustering:")
        print(f"    Low-value clusters: {num_low}")
        print(f"    High-value clusters: {num_high}")
        
        return {
            'thresholds': {
                'p1': p1, 'p5': p5, 'p95': p95, 'p99': p99,
                'lower_bound': lower_bound, 'upper_bound': upper_bound
            },
            'extreme_map': extreme_map,
            'clusters': {'low': num_low, 'high': num_high}
        }
    
    # =========================================================================
    # ANALYSIS 15: RISK ASSESSMENT
    # =========================================================================
    
    def risk_assessment(self):
        """Multi-factor risk assessment for agriculture"""
        print("\n" + "=" * 70)
        print("ANALYSIS 15: AGRICULTURAL RISK ASSESSMENT")
        print("=" * 70)
        
        valid_mask = ~np.isnan(self.sm_data)
        total_pixels = np.sum(valid_mask)
        
        # Multi-factor risk scoring
        risk_map = np.zeros(self.sm_data.shape)
        
        # Factor 1: Absolute moisture level
        risk_map[self.sm_data < 0.10] += 5  # Critical
        risk_map[(self.sm_data >= 0.10) & (self.sm_data < 0.15)] += 4  # Severe
        risk_map[(self.sm_data >= 0.15) & (self.sm_data < 0.20)] += 3  # High
        risk_map[(self.sm_data >= 0.20) & (self.sm_data < 0.25)] += 2  # Moderate
        risk_map[(self.sm_data >= 0.25) & (self.sm_data < 0.30)] += 1  # Low
        
        # Factor 2: Spatial variability
        from scipy.ndimage import uniform_filter
        local_std = uniform_filter(np.nan_to_num(self.sm_data, nan=0)**2, size=7) - \
                    uniform_filter(np.nan_to_num(self.sm_data, nan=0), size=7)**2
        local_std = np.sqrt(np.maximum(local_std, 0))
        
        high_variability = local_std > np.percentile(local_std, 75)
        risk_map[high_variability] += 2
        
        # Factor 3: Proximity to extremes
        extreme_dry = self.sm_data < np.nanpercentile(self.sm_data, 10)
        from scipy.ndimage import distance_transform_edt
        dist_to_extreme = distance_transform_edt(~extreme_dry)
        
        near_extreme = dist_to_extreme < 10
        risk_map[near_extreme] += 1
        
        # Classify risk levels
        risk_map[~valid_mask] = np.nan
        
        risk_categories = {
            'Very Low': np.sum((risk_map >= 0) & (risk_map <= 2)),
            'Low': np.sum((risk_map > 2) & (risk_map <= 4)),
            'Moderate': np.sum((risk_map > 4) & (risk_map <= 6)),
            'High': np.sum((risk_map > 6) & (risk_map <= 8)),
            'Very High': np.sum(risk_map > 8)
        }
        
        print("\n⚠️  Risk Distribution:")
        for category, count in risk_categories.items():
            percent = count / total_pixels * 100
            print(f"  {category:.<15} {count:>7} pixels ({percent:>5.2f}%)")
        
        # Overall risk score
        overall_risk = np.nanmean(risk_map) / 10 * 100  # Normalize to 0-100
        
        print(f"\n  Overall Risk Score: {overall_risk:.1f}/100")
        
        if overall_risk < 20:
            assessment = "✅ Low Risk - Favorable conditions"
        elif overall_risk < 40:
            assessment = "⚡ Moderate Risk - Monitor closely"
        elif overall_risk < 60:
            assessment = "⚠️  High Risk - Intervention recommended"
        else:
            assessment = "🚨 Very High Risk - Immediate action required"
        
        print(f"  Assessment: {assessment}")
        
        # Recommendations
        print(f"\n  Recommendations:")
        if overall_risk > 50:
            print("    • Implement emergency irrigation")
            print("    • Apply drought-resistant practices")
            print("    • Consider crop insurance activation")
        elif overall_risk > 30:
            print("    • Increase monitoring frequency")
            print("    • Prepare irrigation systems")
            print("    • Review crop water requirements")
        else:
            print("    • Continue normal monitoring")
            print("    • Maintain current practices")
        
        return {
            'risk_map': risk_map,
            'risk_categories': risk_categories,
            'overall_risk': overall_risk
        }


def create_advanced_visualizations(analyzer):
    """Create additional visualization plots"""
    
    fig = plt.figure(figsize=(20, 15))
    
    # Run analyses
    texture = analyzer.texture_analysis()
    connectivity = analyzer.connectivity_analysis()
    extreme = analyzer.extreme_value_analysis()
    risk = analyzer.risk_assessment()
    
    # Plot 1: Texture map
    ax1 = plt.subplot(3, 3, 1)
    im1 = ax1.imshow(texture['local_std'], cmap='plasma')
    ax1.set_title('Spatial Texture (Local Variability)', fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # Plot 2: Edge detection
    ax2 = plt.subplot(3, 3, 2)
    im2 = ax2.imshow(texture['edges'], cmap='gray')
    ax2.set_title('Moisture Boundaries', fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # Plot 3: Connectivity - dry zones
    ax3 = plt.subplot(3, 3, 3)
    im3 = ax3.imshow(connectivity['labeled_dry'], cmap='tab20')
    ax3.set_title(f'Dry Zone Connectivity ({connectivity["num_patches"]["dry"]} patches)', fontweight='bold')
    ax3.axis('off')
    
    # Plot 4: Connectivity - wet zones
    ax4 = plt.subplot(3, 3, 4)
    im4 = ax4.imshow(connectivity['labeled_wet'], cmap='tab20')
    ax4.set_title(f'Wet Zone Connectivity ({connectivity["num_patches"]["wet"]} patches)', fontweight='bold')
    ax4.axis('off')
    
    # Plot 5: Extreme values map
    ax5 = plt.subplot(3, 3, 5)
    im5 = ax5.imshow(extreme['extreme_map'], cmap='RdBu_r', vmin=-1, vmax=1)
    ax5.set_title('Extreme Value Zones', fontweight='bold')
    ax5.axis('off')
    cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046)
    cbar5.set_ticks([-0.67, 0, 0.67])
    cbar5.set_ticklabels(['Extreme Low', 'Normal', 'Extreme High'])
    
    # Plot 6: Risk map
    ax6 = plt.subplot(3, 3, 6)
    im6 = ax6.imshow(risk['risk_map'], cmap='YlOrRd')
    ax6.set_title('Multi-Factor Risk Assessment', fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, label='Risk Score', fraction=0.046)
    
    # Plot 7: Risk distribution pie chart
    ax7 = plt.subplot(3, 3, 7)
    risk_cats = risk['risk_categories']
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#c0392b']
    ax7.pie(risk_cats.values(), labels=risk_cats.keys(), autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax7.set_title('Risk Level Distribution', fontweight='bold')
    
    # Plot 8: Homogeneity map
    ax8 = plt.subplot(3, 3, 8)
    homogeneity_map = np.zeros_like(texture['local_std'])
    homogeneity_map[texture['homogeneous']] = 1
    homogeneity_map[texture['heterogeneous']] = 2
    im8 = ax8.imshow(homogeneity_map, cmap='RdYlGn_r', vmin=0, vmax=2)
    ax8.set_title('Spatial Homogeneity', fontweight='bold')
    ax8.axis('off')
    cbar8 = plt.colorbar(im8, ax=ax8, fraction=0.046)
    cbar8.set_ticks([0.33, 1, 1.67])
    cbar8.set_ticklabels(['Normal', 'Homogeneous', 'Heterogeneous'])
    
    # Plot 9: Overall summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = f"""
    ANALYSIS SUMMARY
    
    Risk Assessment:
    • Overall Risk: {risk['overall_risk']:.1f}/100
    • High Risk Areas: {risk['risk_categories']['High'] + risk['risk_categories']['Very High']} pixels
    
    Spatial Patterns:
    • Dry Patches: {connectivity['num_patches']['dry']}
    • Wet Patches: {connectivity['num_patches']['wet']}
    • Extreme Clusters: {extreme['clusters']['low'] + extreme['clusters']['high']}
    
    """
    
    ax9.text(0.1, 0.5, summary_text, transform=ax9.transAxes,
             fontsize=10, verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('NISAR Advanced Analysis Results', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig


# Example usage
if __name__ == "__main__":
    print("\n🚀 NISAR Advanced Analysis Module")
    print("   Load your soil moisture data and run:")
    print("   analyzer = NISARAdvancedAnalysis(sm_data)")
    print("   analyzer.change_detection()")
    print("   analyzer.texture_analysis()")
    print("   # ... and more!")
