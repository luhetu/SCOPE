#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Display evaluation results for all models"""
import os
import json
import glob

def get_latest_result(model_dir):
    """Get the latest evaluation results from model directory"""
    log_files = glob.glob(os.path.join(model_dir, "*.log.json"))
    if not log_files:
        return None
    
    # Get the latest log file
    latest_log = max(log_files, key=os.path.getmtime)
    
    try:
        with open(latest_log, 'r') as f:
            lines = f.readlines()
            # Search from the end to get the last line containing evaluation results
            for line in reversed(lines):
                try:
                    data = json.loads(line.strip())
                    if 'bbox_mAP' in data:
                        return {
                            'epoch': data.get('epoch', 'N/A'),
                            'bbox_mAP': data.get('bbox_mAP', 0),
                            'bbox_mAP_50': data.get('bbox_mAP_50', 0),
                            'bbox_mAP_75': data.get('bbox_mAP_75', 0),
                            'segm_mAP': data.get('segm_mAP', 0),
                            'segm_mAP_50': data.get('segm_mAP_50', 0),
                            'segm_mAP_75': data.get('segm_mAP_75', 0),
                        }
                except:
                    continue
    except:
        pass
    return None

def print_results():
    """Print results of all models"""
    print("=" * 80)
    print("COCO Detection and Instance Segmentation Results Summary")
    print("=" * 80)
    print()
    
    models = [
        ('vit_det', 'ViT + Mask R-CNN'),
        ('vitcope_det', 'CoPE + Mask R-CNN'),
        ('vitscope_det', 'SCoPE + Mask R-CNN'),
    ]
    
    results_table = []
    
    for model_dir, model_name in models:
        full_path = os.path.join('checkpoint', model_dir)
        if not os.path.exists(full_path):
            print("{:20s}: Not trained".format(model_name))
            continue
        
        result = get_latest_result(full_path)
        if result is None:
            print("{:20s}: No evaluation results found".format(model_name))
            continue
        
        results_table.append((model_name, result))
    
    if not results_table:
        print("No evaluation results found!")
        return
    
    print()
    print("=" * 80)
    print("Detection (BBox) and Instance Segmentation (Mask) Performance Comparison")
    print("=" * 80)
    print()
    print("Model                 | APb   | APb50 | APb75 | APm   | APm50 | APm75 | Epoch")
    print("-" * 80)
    
    for model_name, result in results_table:
        print("{:20s} | {:5.1f} | {:5.1f} | {:5.1f} | {:5.1f} | {:5.1f} | {:5.1f} | {:>5}".format(
            model_name,
            result['bbox_mAP'] * 100,
            result['bbox_mAP_50'] * 100,
            result['bbox_mAP_75'] * 100,
            result['segm_mAP'] * 100,
            result['segm_mAP_50'] * 100,
            result['segm_mAP_75'] * 100,
            result['epoch']
        ))
    
    print()
    print("=" * 80)
    print("Paper reference values (ViT-Tiny):")
    print("  APb: 34.8  | APb50: 57.4  | APb75: 36.5")
    print("  APm: 32.5  | APm50: 54.3  | APm75: 33.7")
    print()
    print("Note: Current results are much lower than expected, mainly due to insufficient training epochs (12 vs 36 epochs)")
    print("=" * 80)

if __name__ == "__main__":
    print_results()






