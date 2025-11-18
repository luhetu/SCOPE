#!/usr/bin/env python3
"""
Diagnosis script: Check low result issues for COCO detection and ADE20K segmentation

Features:
1. Visualize prediction results (random 20 images)
2. Check category ID mapping
3. Verify evaluation process
4. Check prediction quantity statistics
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
import argparse

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    PYCOCO_AVAILABLE = True
except ImportError:
    PYCOCO_AVAILABLE = False
    print("⚠️  pycocotools not available, skipping COCO-specific checks")


def check_coco_category_ids(results_file, ann_file):
    """Check category IDs in COCO prediction results"""
    if not PYCOCO_AVAILABLE:
        print("⚠️  pycocotools not available, skipping category ID check")
        return
    
    print("\n" + "="*60)
    print("🔍 Checking COCO Category ID Mapping")
    print("="*60)
    
    # Load GT annotations
    coco_gt = COCO(ann_file)
    cat_ids = coco_gt.getCatIds()
    print(f"✅ COCO GT category ID range: {min(cat_ids)} - {max(cat_ids)}")
    print(f"   COCO category count: {len(cat_ids)}")
    print(f"   Example category IDs: {cat_ids[:5]}")
    
    # Load prediction results
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        if len(results) > 0:
            # Check category ID of first prediction
            first_pred = results[0]
            if 'category_id' in first_pred:
                pred_cat_id = first_pred['category_id']
                print(f"\n📊 Prediction result category ID check:")
                print(f"   First prediction category_id: {pred_cat_id}")
                
                if pred_cat_id in cat_ids:
                    print(f"   ✅ Category ID {pred_cat_id} is in COCO category list")
                else:
                    print(f"   ⚠️  Category ID {pred_cat_id} is not in COCO category list!")
                    print(f"   This may be a category ID mapping issue!")
                
                # Count all predicted category IDs
                all_cat_ids = [r['category_id'] for r in results if 'category_id' in r]
                unique_cat_ids = sorted(set(all_cat_ids))
                print(f"\n   All predicted category IDs: {unique_cat_ids[:10]}... ({len(unique_cat_ids)} unique categories)")
                
                # Check if there are consecutive IDs [0..79]
                if set(unique_cat_ids) == set(range(80)):
                    print(f"   ⚠️  Detected consecutive IDs [0..79], may need to map to COCO native IDs!")
                elif all(cid in cat_ids for cid in unique_cat_ids):
                    print(f"   ✅ All category IDs are in COCO category list")
                else:
                    invalid_ids = [cid for cid in unique_cat_ids if cid not in cat_ids]
                    print(f"   ⚠️  Found invalid category IDs: {invalid_ids[:10]}")
            else:
                print(f"   ⚠️  No 'category_id' field in prediction results")
        else:
            print(f"   ⚠️  Prediction results are empty!")
    else:
        print(f"   ⚠️  Prediction results file does not exist: {results_file}")


def check_prediction_statistics(results_file):
    """Check prediction quantity statistics"""
    print("\n" + "="*60)
    print("📊 Prediction Quantity Statistics")
    print("="*60)
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        if len(results) > 0:
            # Group by image ID
            img_id_to_preds = {}
            for pred in results:
                img_id = pred['image_id']
                if img_id not in img_id_to_preds:
                    img_id_to_preds[img_id] = []
                img_id_to_preds[img_id].append(pred)
            
            num_imgs = len(img_id_to_preds)
            num_preds = len(results)
            avg_preds_per_img = num_preds / num_imgs if num_imgs > 0 else 0
            
            print(f"✅ Statistics:")
            print(f"   Number of images: {num_imgs}")
            print(f"   Total predictions: {num_preds}")
            print(f"   Average predictions per image: {avg_preds_per_img:.2f}")
            
            # Check anomalies
            preds_per_img = [len(preds) for preds in img_id_to_preds.values()]
            min_preds = min(preds_per_img)
            max_preds = max(preds_per_img)
            
            print(f"   Predictions per image range: {min_preds} - {max_preds}")
            
            if avg_preds_per_img < 1:
                print(f"   ⚠️  Average predictions < 1, score_thr may be too high or model recall insufficient")
            elif avg_preds_per_img > 1000:
                print(f"   ⚠️  Average predictions > 1000, score_thr may be too low or NMS failed")
            else:
                print(f"   ✅ Prediction quantity is within reasonable range")
        else:
            print(f"   ⚠️  Prediction results are empty!")
    else:
        print(f"   ⚠️  Prediction results file does not exist: {results_file}")


def check_mask_sizes(results_file, img_dir):
    """Check mask sizes (requires image directory)"""
    print("\n" + "="*60)
    print("🖼️  Checking Mask Sizes")
    print("="*60)
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        if len(results) > 0:
            # Check mask sizes of first few predictions
            sample_preds = results[:5]
            print(f"📊 Checking mask sizes of first 5 predictions:")
            
            for i, pred in enumerate(sample_preds):
                if 'segmentation' in pred:
                    seg = pred['segmentation']
                    if isinstance(seg, dict) and 'size' in seg:
                        # RLE format
                        mask_h, mask_w = seg['size']
                        print(f"   Prediction {i+1}: RLE mask size {mask_h}×{mask_w}")
                    elif isinstance(seg, list):
                        # Polygon format
                        print(f"   Prediction {i+1}: Polygon format ({len(seg)} points)")
                    else:
                        print(f"   Prediction {i+1}: Unknown format")
                else:
                    print(f"   Prediction {i+1}: No mask")
        else:
            print(f"   ⚠️  Prediction results are empty!")
    else:
        print(f"   ⚠️  Prediction results file does not exist: {results_file}")


def sanity_check_coco_eval(ann_file):
    """Sanity check: Use GT boxes as predictions to verify evaluation script"""
    if not PYCOCO_AVAILABLE:
        print("⚠️  pycocotools not available, skipping sanity check")
        return
    
    print("\n" + "="*60)
    print("🧪 Sanity Check: Using GT Boxes as Predictions")
    print("="*60)
    
    coco_gt = COCO(ann_file)
    img_ids = coco_gt.getImgIds()[:10]  # Use only first 10 images for testing
    
    # Create "perfect" predictions (GT boxes + 0.9 score)
    fake_results = []
    for img_id in img_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        
        for ann in anns:
            fake_results.append({
                'image_id': img_id,
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],  # [x, y, w, h]
                'score': 0.9,
                'segmentation': ann['segmentation']
            })
    
    # Save temporary JSON
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(fake_results, f)
        temp_file = f.name
    
    try:
        # Run COCOeval
        coco_dt = coco_gt.loadRes(temp_file)
        coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
        coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Check if AP is close to upper bound
        ap = coco_eval.stats[0]  # mAP
        ap50 = coco_eval.stats[1]  # mAP@0.5
        
        print(f"\n📊 Sanity Check results:")
        print(f"   mAP: {ap:.4f}")
        print(f"   mAP@0.5: {ap50:.4f}")
        
        if ap > 0.8:
            print(f"   ✅ AP is close to upper bound, evaluation script is normal")
        else:
            print(f"   ⚠️  AP is low, evaluation script may have issues")
    finally:
        os.unlink(temp_file)


def main():
    parser = argparse.ArgumentParser(description='Diagnose low result issues for COCO detection and ADE20K segmentation')
    parser.add_argument('--task', type=str, choices=['det', 'seg'], required=True,
                        help='Task type: det (detection) or seg (segmentation)')
    parser.add_argument('--results_file', type=str,
                        help='Prediction results JSON file path (detection task)')
    parser.add_argument('--ann_file', type=str,
                        help='COCO annotation file path (detection task)')
    parser.add_argument('--img_dir', type=str,
                        help='Image directory (optional, for visualization)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🔍 Diagnosis Script - Checking Low Result Issues")
    print("="*60)
    
    if args.task == 'det':
        if not args.results_file or not args.ann_file:
            print("❌ Detection task requires --results_file and --ann_file")
            return
        
        # Check category ID mapping
        check_coco_category_ids(args.results_file, args.ann_file)
        
        # Check prediction quantity statistics
        check_prediction_statistics(args.results_file)
        
        # Check mask sizes
        if args.img_dir:
            check_mask_sizes(args.results_file, args.img_dir)
        
        # Sanity check
        sanity_check_coco_eval(args.ann_file)
        
    elif args.task == 'seg':
        print("\n⚠️  Segmentation task diagnosis function to be implemented")
        print("   Recommended checks:")
        print("   1. Check if pretrained weights are loaded correctly (see training logs)")
        print("   2. Check if training loss decreases normally")
        print("   3. Check if validation mIoU continues to rise")
    
    print("\n" + "="*60)
    print("✅ Diagnosis complete")
    print("="*60)


if __name__ == '__main__':
    main()








