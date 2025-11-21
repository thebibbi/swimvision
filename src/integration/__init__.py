"""
Integration Module for External Systems
========================================

This module provides integrations with external mocap and tracking systems:

1. **SkellyTracker Integration**: Unified API for multiple pose estimators
2. **FreeMoCap Integration**: Multi-camera motion capture system
3. **Export Utilities**: BVH, FBX, C3D format export

Components:
-----------
- skellytracker_wrapper: Wraps our pose estimators for SkellyTracker compatibility
- freemocap_bridge: Connects our pipeline with FreeMoCap's multi-camera system
- export_formats: Exports data in standard mocap formats
"""

__all__ = []
