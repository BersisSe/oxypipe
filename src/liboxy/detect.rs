//! This module contains detection utilities like union intersection, NMS Bounding Box Checking.

use crate::types::BoundingBox;
use ndarray::ArrayView2;

/// Intersection over Union between two axis-aligned boxes.
#[inline]
pub fn iou(a: &BoundingBox, b: &BoundingBox) -> f32 {
    let inter_w = (a.x2.min(b.x2) - a.x1.max(b.x1)).max(0.0);
    let inter_h = (a.y2.min(b.y2) - a.y1.max(b.y1)).max(0.0);
    let inter = inter_w * inter_h;
    if inter <= 0.0 {
        return 0.0;
    }
    inter / (a.area() + b.area() - inter)
}

/// Greedy NMS sorts by score descending and suppresses boxes overlapping
/// any already-kept box with IoU greater than `threshold`.
pub fn nms(mut boxes: Vec<BoundingBox>, threshold: f32) -> Vec<BoundingBox> {
    boxes.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut kept = Vec::new();
    'outer: for candidate in boxes {
        for kept_box in &kept {
            if iou(&candidate, kept_box) > threshold {
                continue 'outer;
            }
        }
        kept.push(candidate);
    }
    kept
}

/// Decodes raw outputs into source-space boxes.
///
/// `scores` is `[anchors, 2]` (class 0 = background, class 1 = face),
/// `boxes` is `[anchors, 4]` with normalized `(x1, y1, x2, y2)`.
/// Coordinates are clipped to [0, 1], scaled to `(src_w, src_h)` pixels,
/// filtered by `score_threshold`, then passed through NMS.
pub fn decode(
    scores: &ArrayView2<f32>,
    boxes: &ArrayView2<f32>,
    src_w: u32,
    src_h: u32,
    score_threshold: f32,
    nms_iou: f32,
) -> Vec<BoundingBox> {
    let anchors = scores.dim().0.min(boxes.dim().0);
    let (w, h) = (src_w as f32, src_h as f32);
    let mut result = Vec::new();

    for i in 0..anchors {
        let score = scores[[i, 1]];
        if score <= score_threshold {
            continue;
        }
        let raw = boxes.row(i);
        let clip = |v: f32| v.clamp(0.0, 1.0);
        let (x1, y1, x2, y2) = (clip(raw[0]), clip(raw[1]), clip(raw[2]), clip(raw[3]));
        if x2 <= x1 || y2 <= y1 {
            continue;
        }
        result.push(BoundingBox {
            x1: x1 * w,
            y1: y1 * h,
            x2: x2 * w,
            y2: y2 * h,
            score,
        });
    }

    nms(result, nms_iou)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_at(x1: f32, y1: f32, x2: f32, y2: f32, score: f32) -> BoundingBox {
        BoundingBox {
            x1,
            y1,
            x2,
            y2,
            score,
        }
    }

    #[test]
    fn iou_identical_boxes_is_one() {
        let a = box_at(0.0, 0.0, 10.0, 10.0, 1.0);
        assert!((iou(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_disjoint_boxes_is_zero() {
        let a = box_at(0.0, 0.0, 10.0, 10.0, 1.0);
        let b = box_at(20.0, 20.0, 30.0, 30.0, 1.0);
        assert!(iou(&a, &b) == 0.0);
    }

    #[test]
    fn iou_half_overlap_is_one_third() {
        let a = box_at(0.0, 0.0, 10.0, 10.0, 1.0);
        let b = box_at(5.0, 0.0, 15.0, 10.0, 1.0);
        // inter = 50, union = 100 + 100 - 50 = 150
        assert!((iou(&a, &b) - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn nms_suppresses_overlapping_keeps_highest_score() {
        let boxes = vec![
            box_at(0.0, 0.0, 10.0, 10.0, 0.8),
            box_at(1.0, 1.0, 11.0, 11.0, 0.9),
            box_at(50.0, 50.0, 60.0, 60.0, 0.5),
        ];
        let kept = nms(boxes, 0.3);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].score, 0.9);
        assert_eq!(kept[1].score, 0.5);
    }

    #[test]
    fn nms_keeps_below_threshold_overlaps() {
        let boxes = vec![
            box_at(0.0, 0.0, 10.0, 10.0, 0.9),
            box_at(5.0, 0.0, 15.0, 10.0, 0.8), // IoU = 1/3
        ];
        let kept = nms(boxes, 0.5);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn decode_filters_threshold_and_scales_to_source() {
        let scores =
            ndarray::Array2::from_shape_vec((3, 2), vec![0.9, 0.1, 0.2, 0.8, 0.5, 0.49]).unwrap();
        let boxes = ndarray::Array2::from_shape_vec(
            (3, 4),
            vec![
                0.0, 0.0, 0.5, 0.5, // kept: score 0.8
                0.0, 0.0, 0.5, 0.5, // suppressed by NMS (same box)
                1.2, -0.1, 0.3, 0.2, // score 0.49 below threshold
            ],
        )
        .unwrap();

        let result = decode(&scores.view(), &boxes.view(), 640, 480, 0.7, 0.3);

        assert_eq!(result.len(), 1);
        let b = result[0];
        assert_eq!(b.x1, 0.0);
        assert_eq!(b.y1, 0.0);
        assert_eq!(b.x2, 320.0);
        assert_eq!(b.y2, 240.0);
        assert_eq!(b.score, 0.8);
    }

    #[test]
    fn decode_clips_out_of_range_coords() {
        let scores = ndarray::Array2::from_shape_vec((1, 2), vec![0.1, 0.9]).unwrap();
        let boxes = ndarray::Array2::from_shape_vec((1, 4), vec![0.5, -0.1, 1.2, 0.9]).unwrap();
        let result = decode(&scores.view(), &boxes.view(), 100, 100, 0.5, 0.3);
        assert_eq!(result.len(), 1);
        let b = result[0];
        assert_eq!(b.x2, 100.0); // 1.2 clipped to 1.0, scaled
        assert_eq!(b.y1, 0.0); // -0.1 clipped to 0.0
        assert_eq!(b.x1, 50.0); // untouched
        assert_eq!(b.y2, 90.0); // untouched
    }

    #[test]
    fn decode_drops_degenerate_boxes_after_clip() {
        let scores = ndarray::Array2::from_shape_vec((1, 2), vec![0.1, 0.9]).unwrap();
        // x2 < x1 after clipping -> zero-width -> dropped
        let boxes = ndarray::Array2::from_shape_vec((1, 4), vec![1.2, 0.0, 0.9, 1.0]).unwrap();
        let result = decode(&scores.view(), &boxes.view(), 100, 100, 0.5, 0.3);
        assert!(result.is_empty());
    }

    #[test]
    fn decode_empty_when_all_below_threshold() {
        let scores = ndarray::Array2::from_shape_vec((2, 2), vec![0.9, 0.1, 0.9, 0.2]).unwrap();
        let boxes = ndarray::Array2::from_shape_vec((2, 4), vec![0.0; 8]).unwrap();
        let result = decode(&scores.view(), &boxes.view(), 100, 100, 0.7, 0.3);
        assert!(result.is_empty());
    }
}
