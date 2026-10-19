"""Standard GIS outputs of a run (Phase 3.2): orthophoto and DSM/DTM/nDSM GeoTIFF (COG),
LAS/LAZ point cloud, INRIA-layout 3DGS .ply.

Everything a run produces lives in WGS84 Site ENU (ADR-0006); this package is where it
leaves that frame for a standard projected CRS (aerosplat.export.georef). Every export
refuses a run without georeference instead of writing an ungeoreferenced file, and every
file carries the run's known georeference error budget.

The heavy optional dependencies (rasterio for GeoTIFF, laspy for LAS) are imported only
by the modules that need them (`pip install -e .[gis]`).
"""
