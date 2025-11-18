#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
COCO Instance Segmentation Quick Self-Check Tool
For diagnosing low AP issues: category ID mapping, mask size, prediction count, etc.
"""
import json
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as mask_util
import os
import argparse
from pathlib import Path


def check_category_ids(pred_json, gt_ann_file, num_samples=10):
    """Check if category ID mapping is correct"""
    print("\n" + "="*60)
    print("🔍 Checking Category ID Mapping")
    print("="*60)
    
    # Load GT
    coco_gt = COCO(gt_ann_file)
    gt_cat_ids = sorted(coco_gt.getCatIds())
    print(f"✅ COCO GT category ID range: {min(gt_cat_ids)} - {max(gt_cat_ids)}")
    print(f"   COCO category count: {len(gt_cat_ids)}")
    
    # Load predictions
    with open(pred_json, 'r') as f:
        pred_data = json.load(f)
    
    if len(pred_data) == 0:
        print("❌ Prediction results are empty!")
        return False
    
    # Count category IDs in predictions
    pred_cat_ids = set()
    for item in pred_data[:num_samples*10]:  # Check more samples
        pred_cat_ids.add(item['category_id'])
    
    pred_cat_ids = sorted(list(pred_cat_ids))
    print(f"📊 Predicted category ID range: {min(pred_cat_ids) if pred_cat_ids else 'N/A'} - {max(pred_cat_ids) if pred_cat_ids else 'N/A'}")
    print(f"   Predicted category count: {len(pred_cat_ids)}")
    
    # Check if they match
    if set(pred_cat_ids) == set(gt_cat_ids):
        print("✅ Category ID mapping is correct!")
        return True
    else:
        print("❌ Category ID mapping is incorrect!")
        print(f"   In GT but not in prediction: {set(gt_cat_ids) - set(pred_cat_ids)}")
        print(f"   In prediction but not in GT: {set(pred_cat_ids) - set(gt_cat_ids)}")
        
        # Check if it's consecutive IDs starting from 0
        if min(pred_cat_ids) == 0 and max(pred_cat_ids) == len(pred_cat_ids) - 1:
            print("⚠️  Predictions use consecutive IDs [0..N-1], need to map to COCO native IDs!")
        return False


def check_mask_sizes(pred_json, gt_ann_file, img_dir, num_samples=5):
    """Check if mask sizes are correct (whether upsampled to original image size)"""
    print("\n" + "="*60)
    print("🔍 Checking Mask Sizes")
    print("="*60)
    
    coco_gt = COCO(gt_ann_file)
    
    with open(pred_json, 'r') as f:
        pred_data = json.load(f)
    
    if len(pred_data) == 0:
        print("❌ Prediction results are empty!")
        return False
    
    # Group by image ID
    img_preds = {}
    for item in pred_data:
        img_id = item['image_id']
        if img_id not in img_preds:
            img_preds[img_id] = []
        img_preds[img_id].append(item)
    
    # Check first few images
    checked = 0
    all_ok = True
    for img_id in list(img_preds.keys())[:num_samples]:
        img_info = coco_gt.loadImgs([img_id])[0]
        ori_h, ori_w = img_info['height'], img_info['width']
        
        # Check predicted mask sizes
        for pred in img_preds[img_id][:3]:  # Check first 3 predictions per image
            segm = pred['segmentation']
            if isinstance(segm, dict) and 'counts' in segm:
                # RLE format
                rle = mask_util.frPyObjects(segm, ori_h, ori_w)
                mask = mask_util.decode(rle)
                mask_h, mask_w = mask.shape[:2]
                
                if mask_h == ori_h and mask_w == ori_w:
                    status = "✅"
                else:
                    status = "❌"
                    all_ok = False
                
                if checked < 3:  # Only print first 3
                    print(f"{status} Image {img_id}: Original {ori_h}x{ori_w}, Mask {mask_h}x{mask_w}")
        
        checked += 1
    
    if all_ok:
        print("✅ All checked mask sizes are correct (match original image size)")
    else:
        print("❌ Found mask size mismatch! Masks need to be upsampled to original image size.")
    
    return all_ok


def check_prediction_stats(pred_json, num_samples=100):
    """Check prediction quantity statistics"""
    print("\n" + "="*60)
    print("🔍 Checking Prediction Quantity Statistics")
    print("="*60)
    
    with open(pred_json, 'r') as f:
        pred_data = json.load(f)
    
    if len(pred_data) == 0:
        print("❌ Prediction results are empty!")
        return False
    
    # Group by image ID
    img_preds = {}
    for item in pred_data:
        img_id = item['image_id']
        if img_id not in img_preds:
            img_preds[img_id] = []
        img_preds[img_id].append(item)
    
    # Statistics
    num_preds_per_img = [len(preds) for preds in img_preds.values()]
    avg_preds = np.mean(num_preds_per_img)
    min_preds = np.min(num_preds_per_img)
    max_preds = np.max(num_preds_per_img)
    
    print(f"📊 Total images: {len(img_preds)}")
    print(f"   Average predictions per image: {avg_preds:.2f}")
    print(f"   Minimum predictions per image: {min_preds}")
    print(f"   Maximum predictions per image: {max_preds}")
    
    # Check score distribution
    scores = [item['score'] for item in pred_data]
    print(f"\n📊 Score statistics:")
    print(f"   Average score: {np.mean(scores):.4f}")
    print(f"   Minimum score: {np.min(scores):.4f}")
    print(f"   Maximum score: {np.max(scores):.4f}")
    print(f"   Count with score>0.5: {sum(1 for s in scores if s > 0.5)}")
    print(f"   Count with score>0.1: {sum(1 for s in scores if s > 0.1)}")
    print(f"   Count with score<0.01: {sum(1 for s in scores if s < 0.01)}")
    
    # Check anomalies
    if avg_preds < 1:
        print("⚠️  Warning: Average predictions < 1, recall rate may be too low")
    if avg_preds > 1000:
        print("⚠️  Warning: Average predictions > 1000, NMS threshold may be too loose")
    
    return True


def visualize_predictions(pred_json, gt_ann_file, img_dir, output_dir, num_samples=5):
    """Visualize prediction results (for visual inspection)"""
    print("\n" + "="*60)
    print("🔍 Visualizing Prediction Results")
    print("="*60)
    
    coco_gt = COCO(gt_ann_file)
    
    with open(pred_json, 'r') as f:
        pred_data = json.load(f)
    
    if len(pred_data) == 0:
        print("❌ Prediction results are empty!")
        return False
    
    # Group by image ID
    img_preds = {}
    for item in pred_data:
        img_id = item['image_id']
        if img_id not in img_preds:
            img_preds[img_id] = []
        img_preds[img_id].append(item)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize first few images
    for i, img_id in enumerate(list(img_preds.keys())[:num_samples]):
        img_info = coco_gt.loadImgs([img_id])[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"⚠️  Image does not exist: {img_path}")
            continue
        
        img = cv2.imread(img_path)
        img_vis = img.copy()
        
        # Draw GT (green)
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        anns = coco_gt.loadAnns(ann_ids)
        for ann in anns:
            if 'segmentation' in ann and ann['segmentation']:
                rle = coco_gt.annToRle(ann)
                mask = mask_util.decode(rle)
                img_vis[mask > 0] = img_vis[mask > 0] * 0.7 + np.array([0, 255, 0]) * 0.3
        
        # Draw predictions (red)
        for pred in img_preds[img_id]:
            segm = pred['segmentation']
            if isinstance(segm, dict) and 'counts' in segm:
                rle = mask_util.frPyObjects(segm, img_info['height'], img_info['width'])
                mask = mask_util.decode(rle)
                img_vis[mask > 0] = img_vis[mask > 0] * 0.7 + np.array([0, 0, 255]) * 0.3
        
        # Save
        output_path = os.path.join(output_dir, f"vis_{img_id}.jpg")
        cv2.imwrite(output_path, img_vis)
        print(f"✅ Saved visualization: {output_path}")
    
    print(f"\n✅ Visualization complete, saved to: {output_dir}")
    print("   Green=GT, Red=Prediction")
    
    return True


def sanity_check_with_gt(pred_json, gt_ann_file):
    """Sanity check: Use GT as predictions, should get near-upper-bound AP"""
    print("\n" + "="*60)
    print("🔍 Sanity Check: Using GT as Predictions")
    print("="*60)
    
    from pycocotools.cocoeval import COCOeval
    
    coco_gt = COCO(gt_ann_file)
    coco_dt = coco_gt.loadRes(pred_json)
    
    # Create JSON with GT as predictions
    gt_as_pred = []
    for img_id in coco_gt.getImgIds()[:100]:  # Use only first 100 images for testing
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        anns = coco_gt.loadAnns(ann_ids)
        for ann in anns:
            if 'segmentation' in ann and ann['segmentation']:
                gt_as_pred.append({
                    'image_id': img_id,
                    'category_id': ann['category_id'],
                    'segmentation': ann['segmentation'],
                    'score': 0.9,  # Give a high score
                    'bbox': ann['bbox']
                })
    
    # Save临时JSON
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(gt_as_pred, f)
        temp_json = f.name
    
    try:
        coco_dt_gt = coco_gt.loadRes(temp_json)
        coco_eval = COCOeval(coco_gt, coco_dt_gt, 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        ap = coco_eval.stats[0]
        if ap > 0.9:
            print(f"✅ Sanity check passed: AP={ap:.3f} (should be close to 1.0)")
            return True
        else:
            print(f"❌ Sanity check failed: AP={ap:.3f} (should be close to 1.0, evaluation script may have issues)")
            return False
    finally:
        os.unlink(temp_json)


def main():
    parser = argparse.ArgumentParser(description='COCO Instance Segmentation Quick Self-Check Tool')
    parser.add_argument('--pred-json', required=True, help='Prediction results JSON file')
    parser.add_argument('--gt-ann', required=True, help='COCO GT annotation file')
    parser.add_argument('--img-dir', help='Image directory (for visualization)')
    parser.add_argument('--output-dir', default='./coco_debug_output', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to check')
    
    args = parser.parse_args()
    
    print("="*60)
    print("COCO Instance Segmentation Quick Self-Check Tool")
    print("="*60)
    
    # 1. Check category ID mapping
    check_category_ids(args.pred_json, args.gt_ann, args.num_samples)
    
    # 2. Check mask sizes
    if args.img_dir:
        check_mask_sizes(args.pred_json, args.gt_ann, args.img_dir, args.num_samples)
    
    # 3. Check prediction statistics
    check_prediction_stats(args.pred_json, args.num_samples)
    
    # 4. Visualize (if image directory is provided)
    if args.img_dir:
        visualize_predictions(args.pred_json, args.gt_ann, args.img_dir, 
                           args.output_dir, min(5, args.num_samples))
    
    # 5. Sanity check
    sanity_check_with_gt(args.pred_json, args.gt_ann)
    
    print("\n" + "="*60)
    print("✅ Self-check complete!")
    print("="*60)


if __name__ == '__main__':
    main()








