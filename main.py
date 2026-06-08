import CSF  # The Cloth Simulation Filter binding
import geopandas as gpd
import laspy
import numpy as np
import overturemaps
import shapely
from pyproj import Transformer
from shapely.ops import unary_union

# =====================================================================
# 1. DEFINE GEOGRAPHIC BOUNDS & TARGET PROJECTION
# =====================================================================
# Coordinates correspond strictly to Spanish MTN50 index 368-4067 (Málaga region)
min_x, max_x = 368000, 369000
min_y, max_y = 4066000, 4067000
local_epsg = "EPSG:25830"  # ETRS89 / UTM zone 30N

# Transform metric bounding box to WGS84 for the Overture API
transformer_to_wgs84 = Transformer.from_crs(local_epsg, "EPSG:4326", always_xy=True)
lon_min, lat_min = transformer_to_wgs84.transform(min_x, min_y)
lon_max, lat_max = transformer_to_wgs84.transform(max_x, max_y)

print(
    f"Fetching Overture data for BBox (GPS): {lon_min:.4f}, {lat_min:.4f} to {lon_max:.4f}, {lat_max:.4f}"
)

# =====================================================================
# 2. DOWNLOAD & PARSE OVERTURE ROAD VECTOR GEOMETRIES
# =====================================================================
try:
    raw_batches = overturemaps.record_batch_reader(
        "segment", bbox=(lon_min, lat_min, lon_max, lat_max)
    ).read_all()

    df = raw_batches.to_pandas()

    if df.empty:
        raise ValueError("Overture returned an empty table.")

    # Fix hidden binary format string parsing errors from Overture data engine
    df["geometry"] = df["geometry"].apply(
        lambda g: shapely.wkb.loads(g) if isinstance(g, bytes) else g
    )
    roads_gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

except Exception as e:
    print(f"Error fetching Overture data: {e}")
    roads_gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

if roads_gdf.empty:
    raise ValueError("No roads found or data failed to load from Overture Maps!")

# Reproject map vectors into Spain's metric coordinate system
roads_gdf = roads_gdf.to_crs(local_epsg)
print(f"Successfully loaded {len(roads_gdf)} road segments.")

# =====================================================================
# 3. GENERATE THE ROAD FOOTPRINT SURFACE BUFFER MASK
# =====================================================================
buffered_polygons = []
for _, row in roads_gdf.iterrows():
    width = 8.0  # Default fallback width (8 meters total / 4m radius buffer)
    if hasattr(row, "width_rules") and row["width_rules"] is not None:
        try:
            width = row["width_rules"][0]["value"]
        except (IndexError, KeyError, TypeError):
            pass

    buffer_radius = width / 2.0
    buffered_polygons.append(row["geometry"].buffer(buffer_radius))

# Flatten all overlapping road buffers into one large 2D polygon surface polygon
road_mask_geom = unary_union(buffered_polygons)

# =====================================================================
# 4. LOAD LAZ FILE & EXECUTE 3D CLOTH SIMULATION GROUND FILTER (CSF)
# =====================================================================
print("Reading PNOA LAZ file...")
las = laspy.read("PNOA.las")

# Stack coordinates into an unrolled XYZ float matrix for the C++ binding
xyz_coordinates = np.vstack((las.x, las.y, las.z)).T

print("Running Cloth Simulation Filter (CSF) to isolate ground topology...")
csf = CSF.CSF()
csf.params.bSloopSmooth = True
csf.params.cloth_resolution = 1.0  # 1-meter grid resolution for cloth vertex spacing
csf.params.rigidness = (
    3  # Standard rigid structural setting for flat urban asphalt terrain
)
csf.setPointCloud(xyz_coordinates)
# Arrays to capture structured indices from the filtering engine
ground = CSF.VecInt()  # a list to indicate the index of ground points after calculation
non_ground = (
    CSF.VecInt()
)  # a list to indicate the index of non-ground points after calculation
csf.do_filtering(ground, non_ground)

# Convert ground indices to an optimized numpy index array
ground_indices = np.array(ground)
print(
    f"CSF Isolated {len(ground_indices)} ground points out of {len(las.points)} total points."
)

# =====================================================================
# 5. SEMANTIC 2D EXTRACTION (Intersecting Ground Points with Vector Mask)
# =====================================================================
print("Filtering extracted 3D ground surfaces against 2D road footprints...")

# Pull coordinates *strictly* belonging to points verified as ground terrain
ground_x = np.array(las.x)[ground_indices]
ground_y = np.array(las.y)[ground_indices]

# Run the highly optimized vectorized point-in-polygon validation natively in C
inside_road_mask = shapely.contains_xy(road_mask_geom, ground_x, ground_y)

# Locate the true primary index positions relative to the original LAZ container layout
final_road_points_idx = ground_indices[np.where(inside_road_mask)[0]]

# =====================================================================
# 6. WRITE CLEANED SEGMENTED ROADS TO UNCOMPRESSED FILE
# =====================================================================
if len(final_road_points_idx) > 0:
    # Clone structure architecture definitions from the source header wrapper template
    output_las = laspy.LasData(las.header)
    output_las.points = las.points[final_road_points_idx]

    # Writing uncompressed .las prevents system LazBackend compatibility exceptions
    output_las.write("PNOA_Segmented_Roads2.las")
    print(
        f"Success! Extracted {len(final_road_points_idx)} clean ground-road points into 'PNOA_Segmented_Roads.las'"
    )
else:
    print(
        "Execution finished, but zero ground points intersected your road vector footprints."
    )
