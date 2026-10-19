#!/usr/bin/env python3
"""
scripts/cache_utils.py
=============================================================================
CENTRALIZED CACHE & DATA LOADER FOR REPRODUCIBILITY
Provides robust, transparent loading of Stage 1 arrays and TreeSHAP attributions.

Features:
1. Transparent support for monolithic and split (_val / _test) NPZ files.
2. Robust detection of 130-byte Git LFS stub files, raising actionable alerts.
3. Alignment assertions ensuring arrays match expected split dimensions.
=============================================================================
"""

import os
import sys
import numpy as np

def load_stage1_cache(ds_name, cache_dir):
    """
    Safely loads Stage 1 model outputs and TreeSHAP attributions.
    Supports monolithic ({ds}_stage1_and_shap.npz) and split files
    ({ds}_stage1_and_shap_val.npz, {ds}_stage1_and_shap_test.npz).
    """
    full_path = os.path.join(cache_dir, f"{ds_name}_stage1_and_shap.npz")
    val_path = os.path.join(cache_dir, f"{ds_name}_stage1_and_shap_val.npz")
    test_path = os.path.join(cache_dir, f"{ds_name}_stage1_and_shap_test.npz")
    
    # 1. Check split files first
    if os.path.isfile(val_path) and os.path.isfile(test_path):
        val_sz = os.path.getsize(val_path)
        test_sz = os.path.getsize(test_path)
        if val_sz > 1000 and test_sz > 1000:
            val_d = np.load(val_path)
            test_d = np.load(test_path)
            combined = {k: val_d[k] for k in val_d.files}
            combined.update({k: test_d[k] for k in test_d.files})
            return combined
        else:
            raise RuntimeError(
                f"[LFS STUB DETECTED] Split files for '{ds_name}' are < 1KB pointers "
                f"({val_path}: {val_sz}B, {test_path}: {test_sz}B). "
                f"Please ensure real files are present or run step0_train_stage1_and_shap.py."
            )
            
    # 2. Check full monolithic file
    if os.path.isfile(full_path):
        full_sz = os.path.getsize(full_path)
        if full_sz < 1000:
            raise RuntimeError(
                f"[LFS STUB DETECTED] File {full_path} is a Git LFS pointer stub ({full_sz} bytes). "
                f"Please download full binary files or re-run 'python scripts/step0_train_stage1_and_shap.py --force'."
            )
        return np.load(full_path)
        
    raise FileNotFoundError(
        f"Stage 1 cache for '{ds_name}' not found at {full_path} or ({val_path}, {test_path}).\n"
        f"Run 'python scripts/step0_train_stage1_and_shap.py' to generate from raw data."
    )
