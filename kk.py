import argparse
import os

import CSF  # The Cloth Simulation Filter binding
import geopandas as gpd
import laspy
import numpy as np
import overturemaps
import shapely
from pyproj import Transformer
from scipy.spatial import KDTree
from shapely.geometry import LineString, MultiLineString, box
from shapely.ops import unary_union

# =====================================================================
# CONFIGURATION & GEOGRAPHIC BOUNDS
# =====================================================================
# Coordinates correspond strictly to Spanish MTN50 index 368-4067 (Málaga region)
MIN_X, MAX_X = 368000, 369000
MIN_Y, MAX_Y = 4066000, 4067000
LOCAL_EPSG = "EPSG:25830"  # ETRS89 / UTM zone 30N
INPUT_LAZ = "PNOA.las"  # PNOA_2024_AND_368-4067_H30_NPC01.laz
ADJUST_TO_Z = True
CACHE_FILE = "overture_cache.gpkg"  # Local storage file name

# Overture 'class' properties mapped to real-world metric total road widths
ROAD_WIDTH_DICTIONARY = {
    "motorway": 16.0,
    "trunk": 14.0,
    "primary": 12.0,
    "secondary": 10.0,
    "tertiary": 9.0,
    "residential": 7.5,
    "living_street": 6.0,
    "service": 4.5,
    "cycleway": 2.5,  # Dedicated bicycle track footprint
    "footway": 2.0,  # Sidewalks and pedestrian paths
    "pedestrian": 4.0,
    "path": 2.0,
}


def fetch_overture_roads():
    """Checks for a local cache file, otherwise downloads vector data from Overture."""
    # If cache exists, load it natively from disk instantly
    if os.path.exists(CACHE_FILE):
        print(f"Loading road network from local offline cache: '{CACHE_FILE}'...")
        roads_gdf = gpd.read_file(CACHE_FILE)
        return roads_gdf

    # Otherwise, execute the standard download pipeline
    transformer_to_wgs84 = Transformer.from_crs(LOCAL_EPSG, "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer_to_wgs84.transform(MIN_X, MIN_Y)
    lon_max, lat_max = transformer_to_wgs84.transform(MAX_X, MAX_Y)

    print(
        f"Cache not found. Fetching Overture data via API for BBox: {lon_min:.4f}, {lat_min:.4f} to {lon_max:.4f}, {lat_max:.4f}"
    )

    try:
        raw_batches = overturemaps.record_batch_reader(
            "segment", bbox=(lon_min, lat_min, lon_max, lat_max)
        ).read_all()

        df = raw_batches.to_pandas()
        if df.empty:
            raise ValueError("Overture returned an empty table.")

        df["geometry"] = df["geometry"].apply(
            lambda g: shapely.wkb.loads(g) if isinstance(g, bytes) else g
        )
        roads_gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        roads_gdf = roads_gdf.to_crs(LOCAL_EPSG)

        # FIX: Ensure 'class' column is explicitly converted to standard string type
        if "class" in roads_gdf.columns:
            roads_gdf["class"] = roads_gdf["class"].astype(str)

        print(f"Saving downloaded segments to local cache: '{CACHE_FILE}'...")

        # FIX: Instead of stripping data types, just drop columns containing un-saveable nested lists/dicts
        complex_columns = [
            "sources",
            "width_rules",
            "road_flags",
            "prohibited_transitions",
            "speed_limits",
        ]
        clean_roads_gdf = roads_gdf.drop(
            columns=[col for col in complex_columns if col in roads_gdf.columns]
        )

        clean_roads_gdf.to_file(CACHE_FILE, driver="GPKG")

        print(f"Successfully loaded {len(roads_gdf)} regional road segments.")
        return roads_gdf

    except Exception as e:
        print(f"Error fetching Overture data: {e}")
        return None


# =====================================================================
# MODE 1: POINT CLOUD EXTRACTION WITH LABELED DYNAMIC BUFFERS
# =====================================================================
def run_point_cloud_segmentation(roads_gdf):
    """Isolates true ground points and extracts point clouds within semantic road masks."""
    print("\n--- Running Mode: Point Cloud Segmentation (CSF) ---")
    buffered_polygons = []

    print("Calculating dynamic buffer spaces based on path labels...")
    for _, row in roads_gdf.iterrows():
        width = None
        if hasattr(row, "width_rules") and row["width_rules"] is not None:
            try:
                width = row["width_rules"][0]["value"]
            except (IndexError, KeyError, TypeError):
                pass

        if width is None:
            road_class = getattr(row, "class", "residential")
            width = ROAD_WIDTH_DICTIONARY.get(road_class, 8.0)

        buffer_radius = width / 2.0
        buffered_polygons.append(row["geometry"].buffer(buffer_radius))

    road_mask_geom = unary_union(buffered_polygons)

    print(f"Reading PNOA file: {INPUT_LAZ}...")
    las = laspy.read(INPUT_LAZ)
    xyz_coordinates = np.vstack((las.x, las.y, las.z)).T

    print("Running Cloth Simulation Filter (CSF) to isolate ground topology...")
    csf = CSF.CSF()
    csf.params.bSloopSmooth = True
    csf.params.cloth_resolution = 1.0
    csf.params.rigidness = 3
    csf.setPointCloud(xyz_coordinates)

    ground = CSF.VecInt()
    non_ground = CSF.VecInt()
    csf.do_filtering(ground, non_ground)

    ground_indices = np.array(ground)
    print(
        f"CSF Isolated {len(ground_indices)} ground points out of {len(las.points)} total points."
    )

    print(
        "Filtering extracted 3D ground surfaces against 2D semantic road footprints..."
    )
    ground_x = np.array(las.x)[ground_indices]
    ground_y = np.array(las.y)[ground_indices]

    inside_road_mask = shapely.contains_xy(road_mask_geom, ground_x, ground_y)
    final_road_points_idx = ground_indices[np.where(inside_road_mask)[0]]

    if len(final_road_points_idx) > 0:
        output_las = laspy.LasData(las.header)
        output_las.points = las.points[final_road_points_idx]
        output_las.write("PNOA_Segmented_Roads.las")
        print(
            f"Success! Extracted {len(final_road_points_idx)} clean points into 'PNOA_Segmented_Roads.las'"
        )
    else:
        print("Finished, but zero ground points intersected your road footprints.")


# =====================================================================
# MODE 2: EXPORT COLOR-CODED CAD LAYERS TO DXF (VIA FIONA ENGINE)
# =====================================================================
def run_vector_dxf_export(roads_gdf):
    """Crops vector centerlines, maps labels to CAD layers, and exports via Fiona."""
    print("\n--- Running Mode: CAD Vector Line Export (DXF) ---")
    print("Cropping road lines exactly to the 1x1 km tile bounding box...")
    bbox_polygon = box(MIN_X, MIN_Y, MAX_X, MAX_Y)
    cropped_roads = roads_gdf.clip(bbox_polygon).copy()

    if ADJUST_TO_Z:
        print(
            f"Reading PNOA reference data from {INPUT_LAZ} to build altitude matrix..."
        )
        las = laspy.read(INPUT_LAZ)
        pnoa_xy = np.vstack((las.x, las.y)).T
        pnoa_z = np.array(las.z)

        print(
            "Building spatial lookup tree (KD-Tree) for fast neighborhood indexing..."
        )
        spatial_tree = KDTree(pnoa_xy)

        print(
            "Snapping line vertices to the local 5th percentile Z-value (the floor)..."
        )
        snapped_geometries = []
        search_radius = 1.0  # Look up within a 3-meter radius circle around the vertex
        percentile_target = (
            5.0  # 5th percentile isolates true ground, ignoring objects above
        )

        for _, row in cropped_roads.iterrows():
            geom = row["geometry"]

            # FIX: Break MultiLineStrings down into individual simple lines
            sub_lines = (
                geom.geoms
                if isinstance(
                    geom,
                    (MultiLineString, shapely.geometry.collection.GeometryCollection),
                )
                else [geom]
            )

            processed_sub_lines = []
            for line in sub_lines:
                if not isinstance(line, LineString):
                    continue

                snapped_coords = []
                for x, y in line.coords:
                    # Query point cloud neighbors surrounding the coordinate coordinate vertex
                    neighbor_indices = spatial_tree.query_ball_point(
                        [x, y], r=search_radius
                    )

                    if len(neighbor_indices) > 0:
                        local_z_values = pnoa_z[neighbor_indices]
                        snapped_z = np.percentile(local_z_values, percentile_target)
                    else:
                        snapped_z = 0.0  # Fallback out-of-bounds boundary elevation

                    snapped_coords.append((x, y, snapped_z))

                if len(snapped_coords) >= 2:
                    processed_sub_lines.append(LineString(snapped_coords))

            # Re-bundle the 3D-snapped line components back into the array row structure
            if len(processed_sub_lines) == 1:
                snapped_geometries.append(processed_sub_lines[0])
            elif len(processed_sub_lines) > 1:
                snapped_geometries.append(MultiLineString(processed_sub_lines))
            else:
                snapped_geometries.append(
                    geom
                )  # Fallback to original layout if sampling failed

        cropped_roads["geometry"] = snapped_geometries

    print("Mapping labels ('class') to standalone CAD rendering layers...")
    # FIX 1: Use a capital 'Layer' and explicitly cast to strings to satisfy Fiona requirements
    cropped_roads["Layer"] = cropped_roads["class"].fillna("unclassified").astype(str)

    # FIX 2: Pull the matching capitalized column layout
    just_geometry_and_layers = cropped_roads[["geometry", "Layer"]]

    print("Exporting classified vector lines to DXF format...")
    output_filename = "Overture_Roads.dxf"

    # Run the export with the optimized structural schema configuration
    just_geometry_and_layers.to_file(output_filename, driver="DXF", engine="fiona")
    print(f"Success! Saved labeled vector blueprints directly to '{output_filename}'")


# =====================================================================
# CLI ARGUMENT HANDLING & ENTRYPOINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="PNOA LiDAR & Overture Maps Semantic Automation Framework Tool."
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["segment", "vector"],
        required=True,
        help="Pipeline operational execution path selection.",
    )

    args = parser.parse_args()
    roads_gdf = fetch_overture_roads()
    if roads_gdf is None or roads_gdf.empty:
        print(
            "Initialization halted: Pipeline dependencies could not fetch geographic vector networks."
        )
        return

    if args.mode == "segment":
        run_point_cloud_segmentation(roads_gdf)
    elif args.mode == "vector":
        run_vector_dxf_export(roads_gdf)


if __name__ == "__main__":
    main()
