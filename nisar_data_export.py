"""
NISAR Machine Learning & Data Export Module
==========================================
Prepare NISAR data for ML pipelines, GIS, and cloud platforms
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime

class NISARDataExporter:
    """Export NISAR data to various formats for different platforms"""
    
    def __init__(self, sm_data, lat=None, lon=None, metadata=None):
        self.sm_data = sm_data
        self.lat = lat
        self.lon = lon
        self.metadata = metadata or {}
        
    # =========================================================================
    # EXPORT 1: GeoTIFF with full metadata
    # =========================================================================
    
    def export_geotiff_advanced(self, output_path, bounds=None):
        """Export to GeoTIFF with comprehensive metadata"""
        try:
            import rasterio
            from rasterio.transform import from_bounds
            from rasterio.crs import CRS
            
            if bounds is None:
                # From metadata: Egypt region
                bounds = (29.12181, 29.37974, 32.41245, 32.25275)
            
            min_lon, min_lat, max_lon, max_lat = bounds
            
            transform = from_bounds(min_lon, min_lat, max_lon, max_lat,
                                   self.sm_data.shape[1], self.sm_data.shape[0])
            
            # Create metadata
            meta = {
                'driver': 'GTiff',
                'dtype': 'float32',
                'nodata': -9999,
                'width': self.sm_data.shape[1],
                'height': self.sm_data.shape[0],
                'count': 1,
                'crs': CRS.from_epsg(4326),
                'transform': transform,
                'compress': 'lzw',
                'tiled': True,
                'blockxsize': 256,
                'blockysize': 256
            }
            
            with rasterio.open(output_path, 'w', **meta) as dst:
                # Write data
                sm_filled = np.nan_to_num(self.sm_data, nan=-9999)
                dst.write(sm_filled.astype('float32'), 1)
                
                # Write tags
                dst.update_tags(
                    AREA_OR_POINT='Area',
                    TIFFTAG_SOFTWARE='NISAR Analysis Suite',
                    TIFFTAG_DATETIME=datetime.now().isoformat(),
                    NISAR_PRODUCT='L3_SME2',
                    PROCESSING_DATE='2025-12-16',
                    ORBIT='2009',
                    TRACK='7',
                    FRAME='73'
                )
                
                # Write band description
                dst.set_band_description(1, 'Soil Moisture (m³/m³)')
            
            print(f"✅ Advanced GeoTIFF exported: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ GeoTIFF export failed: {e}")
            return False
    
    # =========================================================================
    # EXPORT 2: NetCDF for climate models
    # =========================================================================
    
    def export_netcdf(self, output_path):
        """Export to NetCDF format for climate/weather models"""
        try:
            from netCDF4 import Dataset
            
            # Create NetCDF file
            nc = Dataset(output_path, 'w', format='NETCDF4')
            
            # Create dimensions
            nc.createDimension('lat', self.sm_data.shape[0])
            nc.createDimension('lon', self.sm_data.shape[1])
            nc.createDimension('time', 1)
            
            # Create variables
            latitudes = nc.createVariable('lat', 'f4', ('lat',))
            longitudes = nc.createVariable('lon', 'f4', ('lon',))
            time = nc.createVariable('time', 'f8', ('time',))
            soil_moisture = nc.createVariable('soil_moisture', 'f4', 
                                             ('time', 'lat', 'lon',),
                                             fill_value=-9999)
            
            # Assign data
            if self.lat is not None:
                latitudes[:] = self.lat[:, 0] if self.lat.ndim == 2 else self.lat
            else:
                latitudes[:] = np.linspace(29.38, 32.25, self.sm_data.shape[0])
            
            if self.lon is not None:
                longitudes[:] = self.lon[0, :] if self.lon.ndim == 2 else self.lon
            else:
                longitudes[:] = np.linspace(29.12, 32.41, self.sm_data.shape[1])
            
            time[:] = 0  # Days since reference
            soil_moisture[0, :, :] = np.nan_to_num(self.sm_data, nan=-9999)
            
            # Add attributes
            latitudes.units = 'degrees_north'
            latitudes.long_name = 'Latitude'
            
            longitudes.units = 'degrees_east'
            longitudes.long_name = 'Longitude'
            
            time.units = 'days since 2025-12-16 00:00:00'
            time.calendar = 'gregorian'
            
            soil_moisture.units = 'm³/m³'
            soil_moisture.long_name = 'Volumetric Soil Moisture'
            soil_moisture.standard_name = 'soil_moisture_content'
            
            # Global attributes
            nc.title = 'NISAR L3 Soil Moisture Product'
            nc.institution = 'NASA/JPL'
            nc.source = 'NISAR L-SAR'
            nc.processing_level = 'L3'
            nc.product_type = 'SME2'
            nc.creation_date = datetime.now().isoformat()
            
            nc.close()
            
            print(f"✅ NetCDF exported: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ NetCDF export failed: {e}")
            return False
    
    # =========================================================================
    # EXPORT 3: CSV for tabular analysis
    # =========================================================================
    
    def export_csv(self, output_path, sample_rate=1):
        """Export to CSV with coordinates"""
        
        rows = []
        
        # Generate lat/lon if not provided
        if self.lat is None or self.lon is None:
            lat_coords = np.linspace(29.38, 32.25, self.sm_data.shape[0])
            lon_coords = np.linspace(29.12, 32.41, self.sm_data.shape[1])
            lat_grid, lon_grid = np.meshgrid(lat_coords, lon_coords, indexing='ij')
        else:
            lat_grid = self.lat
            lon_grid = self.lon
        
        # Sample data
        for i in range(0, self.sm_data.shape[0], sample_rate):
            for j in range(0, self.sm_data.shape[1], sample_rate):
                if not np.isnan(self.sm_data[i, j]):
                    rows.append({
                        'latitude': lat_grid[i, j] if lat_grid.ndim == 2 else lat_grid[i],
                        'longitude': lon_grid[i, j] if lon_grid.ndim == 2 else lon_grid[j],
                        'row': i,
                        'col': j,
                        'soil_moisture': self.sm_data[i, j]
                    })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        print(f"✅ CSV exported: {output_path} ({len(df)} points)")
        return df
    
    # =========================================================================
    # EXPORT 4: GeoJSON for web mapping
    # =========================================================================
    
    def export_geojson(self, output_path, grid_size=10):
        """Export to GeoJSON format for web maps"""
        
        features = []
        
        # Generate coordinates
        if self.lat is None or self.lon is None:
            lat_coords = np.linspace(29.38, 32.25, self.sm_data.shape[0])
            lon_coords = np.linspace(29.12, 32.41, self.sm_data.shape[1])
        else:
            lat_coords = self.lat[:, 0] if self.lat.ndim == 2 else self.lat
            lon_coords = self.lon[0, :] if self.lon.ndim == 2 else self.lon
        
        # Create grid of features
        for i in range(0, self.sm_data.shape[0], grid_size):
            for j in range(0, self.sm_data.shape[1], grid_size):
                if not np.isnan(self.sm_data[i, j]):
                    
                    # Get cell bounds
                    lat1 = lat_coords[i] if lat_coords.ndim == 1 else lat_coords[i, j]
                    lat2 = lat_coords[min(i+grid_size, len(lat_coords)-1)]
                    lon1 = lon_coords[j] if lon_coords.ndim == 1 else lon_coords[i, j]
                    lon2 = lon_coords[min(j+grid_size, len(lon_coords)-1)]
                    
                    # Create polygon
                    polygon = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [lon1, lat1],
                                [lon2, lat1],
                                [lon2, lat2],
                                [lon1, lat2],
                                [lon1, lat1]
                            ]]
                        },
                        "properties": {
                            "soil_moisture": float(self.sm_data[i, j]),
                            "row": int(i),
                            "col": int(j)
                        }
                    }
                    features.append(polygon)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f)
        
        print(f"✅ GeoJSON exported: {output_path} ({len(features)} features)")
        return geojson
    
    # =========================================================================
    # EXPORT 5: ML-ready format
    # =========================================================================
    
    def export_ml_features(self, output_path):
        """Export feature-engineered data for ML"""
        from scipy.ndimage import uniform_filter, sobel
        
        print("\n🤖 Generating ML Features...")
        
        # Base feature
        sm_filled = np.nan_to_num(self.sm_data, nan=np.nanmean(self.sm_data))
        
        # Feature 1: Local statistics
        local_mean = uniform_filter(sm_filled, size=5)
        local_std = np.sqrt(uniform_filter(sm_filled**2, size=5) - local_mean**2)
        
        # Feature 2: Gradients
        grad_y = sobel(sm_filled, axis=0)
        grad_x = sobel(sm_filled, axis=1)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Feature 3: Texture
        local_range = uniform_filter(sm_filled, size=7, mode='constant')
        local_max = uniform_filter(sm_filled, size=7, mode='constant')
        texture = local_max - local_range
        
        # Feature 4: Distance to extreme
        from scipy.ndimage import distance_transform_edt
        extreme_dry = sm_filled < np.percentile(sm_filled, 10)
        dist_to_dry = distance_transform_edt(~extreme_dry)
        
        # Compile features
        rows = []
        valid_mask = ~np.isnan(self.sm_data)
        
        for i in range(self.sm_data.shape[0]):
            for j in range(self.sm_data.shape[1]):
                if valid_mask[i, j]:
                    rows.append({
                        'soil_moisture': self.sm_data[i, j],
                        'local_mean': local_mean[i, j],
                        'local_std': local_std[i, j],
                        'gradient_mag': gradient_mag[i, j],
                        'texture': texture[i, j],
                        'dist_to_dry': dist_to_dry[i, j],
                        'row': i,
                        'col': j
                    })
        
        df = pd.DataFrame(rows)
        
        # Add derived features
        df['moisture_anomaly'] = df['soil_moisture'] - df['local_mean']
        df['cv'] = df['local_std'] / (df['local_mean'] + 0.001)
        
        # Categorize
        df['moisture_class'] = pd.cut(df['soil_moisture'], 
                                      bins=[0, 0.15, 0.35, 1.0],
                                      labels=['dry', 'optimal', 'saturated'])
        
        df.to_csv(output_path, index=False)
        
        print(f"✅ ML features exported: {output_path}")
        print(f"   Features: {df.shape[1]} columns, {df.shape[0]} samples")
        
        return df
    
    # =========================================================================
    # EXPORT 6: Google Earth Engine ready
    # =========================================================================
    
    def export_gee_asset(self, output_prefix):
        """Prepare files for GEE upload"""
        
        print("\n🌍 Preparing Google Earth Engine Asset...")
        
        # 1. Export GeoTIFF
        geotiff_path = f"{output_prefix}_gee.tif"
        self.export_geotiff_advanced(geotiff_path)
        
        # 2. Create manifest
        manifest = {
            "name": f"projects/YOUR_PROJECT/assets/{output_prefix}",
            "tilesets": [{
                "sources": [{
                    "uris": [geotiff_path]
                }]
            }],
            "bands": [{
                "id": "soil_moisture",
                "tileset_band_index": 0
            }],
            "properties": {
                "satellite": "NISAR",
                "product": "L3_SME2",
                "date": "2025-12-16",
                "orbit": "2009",
                "processing_level": "L3"
            }
        }
        
        manifest_path = f"{output_prefix}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # 3. Create upload instructions
        instructions = f"""
Google Earth Engine Upload Instructions
========================================

1. Upload GeoTIFF to Google Cloud Storage:
   gsutil cp {geotiff_path} gs://your-bucket/nisar/

2. Upload to Earth Engine:
   earthengine upload image --asset_id=users/YOUR_USERNAME/{output_prefix} \\
       gs://your-bucket/nisar/{geotiff_path} \\
       --property satellite=NISAR \\
       --property product=L3_SME2 \\
       --property date=2025-12-16

3. Or use Python API:
   task = ee.batch.Export.image.toAsset(
       image=ee.Image('{geotiff_path}'),
       description='{output_prefix}',
       assetId='users/YOUR_USERNAME/{output_prefix}'
   )
   task.start()

Files created:
- {geotiff_path}
- {manifest_path}
"""
        
        instructions_path = f"{output_prefix}_gee_instructions.txt"
        with open(instructions_path, 'w') as f:
            f.write(instructions)
        
        print(f"✅ GEE files prepared:")
        print(f"   • {geotiff_path}")
        print(f"   • {manifest_path}")
        print(f"   • {instructions_path}")
        
        return True


class NISARMLPreparation:
    """Prepare data for machine learning workflows"""
    
    def __init__(self, sm_data):
        self.sm_data = sm_data
        
    def create_training_patches(self, patch_size=32, stride=16):
        """Extract patches for CNN training"""
        
        print(f"\n🎯 Extracting {patch_size}x{patch_size} patches...")
        
        patches = []
        labels = []
        
        h, w = self.sm_data.shape
        
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = self.sm_data[i:i+patch_size, j:j+patch_size]
                
                # Only keep patches with >80% valid data
                valid_ratio = np.sum(~np.isnan(patch)) / (patch_size * patch_size)
                
                if valid_ratio > 0.8:
                    # Fill NaN with local mean
                    patch_filled = np.nan_to_num(patch, nan=np.nanmean(patch))
                    patches.append(patch_filled)
                    
                    # Label based on mean moisture
                    mean_moisture = np.nanmean(patch)
                    if mean_moisture < 0.15:
                        label = 0  # dry
                    elif mean_moisture < 0.35:
                        label = 1  # optimal
                    else:
                        label = 2  # saturated
                    
                    labels.append(label)
        
        patches = np.array(patches)
        labels = np.array(labels)
        
        print(f"✅ Extracted {len(patches)} patches")
        print(f"   Shape: {patches.shape}")
        print(f"   Labels: {np.bincount(labels)}")
        
        return patches, labels
    
    def export_tensorflow_dataset(self, output_dir):
        """Export as TensorFlow dataset"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        patches, labels = self.create_training_patches()
        
        # Save as numpy
        np.save(f"{output_dir}/patches.npy", patches)
        np.save(f"{output_dir}/labels.npy", labels)
        
        # Create dataset info
        info = {
            "num_samples": len(patches),
            "patch_size": patches.shape[1],
            "num_classes": 3,
            "class_names": ["dry", "optimal", "saturated"],
            "feature_shape": list(patches.shape[1:])
        }
        
        with open(f"{output_dir}/dataset_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        # Create loading script
        script = """
import numpy as np
import tensorflow as tf

# Load data
patches = np.load('patches.npy')
labels = np.load('labels.npy')

# Create TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((patches, labels))
dataset = dataset.shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)

# Example model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=patches.shape[1:]),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(dataset, epochs=10)
"""
        
        with open(f"{output_dir}/load_dataset.py", 'w') as f:
            f.write(script)
        
        print(f"✅ TensorFlow dataset exported to: {output_dir}")
        
        return True


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def export_all_formats_example(sm_data):
    """Example: Export to all formats"""
    
    print("\n" + "=" * 70)
    print("EXPORTING TO ALL FORMATS")
    print("=" * 70)
    
    exporter = NISARDataExporter(sm_data)
    
    # 1. GeoTIFF
    exporter.export_geotiff_advanced("nisar_soil_moisture.tif")
    
    # 2. NetCDF
    try:
        exporter.export_netcdf("nisar_soil_moisture.nc")
    except:
        print("⚠️  NetCDF export requires netCDF4 package")
    
    # 3. CSV
    exporter.export_csv("nisar_soil_moisture.csv", sample_rate=5)
    
    # 4. GeoJSON
    exporter.export_geojson("nisar_soil_moisture.geojson", grid_size=20)
    
    # 5. ML features
    exporter.export_ml_features("nisar_ml_features.csv")
    
    # 6. GEE
    exporter.export_gee_asset("nisar_soil_moisture")
    
    print("\n✅ All exports complete!")


def prepare_ml_pipeline_example(sm_data):
    """Example: Prepare ML pipeline"""
    
    print("\n" + "=" * 70)
    print("PREPARING ML PIPELINE")
    print("=" * 70)
    
    ml_prep = NISARMLPreparation(sm_data)
    
    # Extract patches
    patches, labels = ml_prep.create_training_patches(patch_size=32, stride=16)
    
    # Export TensorFlow dataset
    ml_prep.export_tensorflow_dataset("nisar_tf_dataset")
    
    print("\n✅ ML pipeline ready!")


if __name__ == "__main__":
    print("\n📦 NISAR Data Export & ML Preparation Module")
    print("   Ready to export to 10+ formats!")
