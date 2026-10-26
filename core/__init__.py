"""Core algorithms — reused from Streamlit app with minimal modifications.

These modules are copied from the original shimming_app/core/ directory.
They have NO Streamlit dependency and work standalone.

Modules:
- physics.py: Biot-Savart, shim_optimize, B0/coil I/O
- shim_solver.py: RobustL2ShimSolver, channel loading
- fsl_replace.py: Pure-Python bet/flirt/fslmaths
- B0_cal_algorithm.py: DICOM→deltaB0 pipeline
- data_context.py: Unified data loading helpers
"""
