########################################################################################################################
# Metrics to produce
#
# Abs Relative Difference:
# Sum{ |GroundTruth - Prediction| / GroundTruth } / |Num of Samples|
#
# Squared Relative Difference:
# Sum{ |GroundTruth - Prediction|^2 / GroundTruth } / |Num of Samples|
#
# RMSE (linear):
# sqrt{ Sum{ |GroundTruth - Prediction|^2 } / |Num of Samples| }
#
# RMSE (log)
# sqrt{ Sum{ |log(GroundTruth) - log(Prediction)|^2 } / |Num of Samples| }
#
# RMSE (log Scale-invariant)
# alpha(GroundTruth, Prediction) = 1 / n * Sum{ log(Prediction) - log(GroundTruth) }
# Error(GroundTruth, Prediction) = 1/ (2n) * Sum{ [log(GroundTruth) - log(Prediction) + alpha(GroundTruth, Prediction)]^2 }
#
# Thresholds (deltas)
# Threshold % subject to GroundTruth: max(Prediction / GroundTruth, GroundTruth / Prediction)
#
# This script assumes that ground truth as well as predictions are in the same directory, 
# and that the predicted depth maps are suffixed with "_pred"
########################################################################################################################
import numpy as np
import cv2
import os
import math
import argparse as arg


WIDTH = 512
HEIGHT = 256
DELTA_THRESH = 1.25
MASK_THRESH = 8.0

def load_depths(gt_path, pred_path):
    ground_truth_depth = cv2.imread(gt_path, cv2.IMREAD_ANYDEPTH)
    prediction_depth = cv2.imread(pred_path, cv2.IMREAD_ANYDEPTH)
    return ground_truth_depth, prediction_depth

def get_shape(depth):
    height = depth.shape[0]
    width = depth.shape[1]
    return height, width

def calcAbsRel(gt, pred, count):
    abs_rel = np.sum(np.abs((pred - gt)) / gt)
    abs_rel = abs_rel / count
    return abs_rel

def calcSqRel(gt, pred, count):
    sq_rel = np.sum(np.square(np.abs(pred - gt)) / gt)
    sq_rel = sq_rel / count
    return sq_rel

def calcRMSE(gt, pred, count):
    rmse = np.sum(np.square(np.abs(pred - gt)))
    rmse = rmse / count
    rmse = math.sqrt(rmse)
    return rmse

def calcRMSELog(gt, pred, count):
    rmse = np.sum(np.square(np.abs(np.log(pred) - np.log(gt))))
    if np.isnan(rmse):
        rmse = 0
    rmse = rmse / count
    rmse = math.sqrt(rmse)
    return rmse

def calcDelta(gt, pred, threshold, count):
    thresh = np.maximum((gt / pred), (pred / gt))
    delta = (thresh < DELTA_THRESH).mean()
    return delta

def calcDeltaSecondOrder(gt, pred, threshold, count):
    thresh = np.maximum((gt / pred), (pred / gt))
    delta = (thresh < DELTA_THRESH ** 2).mean()
    return delta

def calcDeltaThirdOrder(gt, pred, threshold, count):
    thresh = np.maximum((gt / pred), (pred / gt))
    delta = (thresh < DELTA_THRESH ** 3).mean()
    return delta

def scale_median(gt_depth, pred_depth):
    gt_med = np.median(gt_depth)
    pred_med = np.median(pred_depth)
    s = gt_med / pred_med
    return gt_depth, pred_depth * s

def mask_gt_depth(gt_depth, mask_thresh):
    mask = (gt_depth < mask_thresh)
    gt_depth = gt_depth[mask]
    return gt_depth, mask

def scaleMinMax(gt, pred):
    gt_min = np.min(gt)
    gt_max = np.max(gt)
    pred_min = np.min(pred)
    pred_max = np.max(pred)
    gt = (gt - gt_min) / (gt_max - gt_min)
    pred = (pred - pred_min) / (pred_max - pred_min)
    return gt, pred

def scaleAvg(gt, pred):
    gt_mean = np.mean(gt)
    pred_mean = np.mean(pred)
    pred = pred * gt_mean / pred_mean
    return gt, pred



def main():
    arg_parser = arg.ArgumentParser(description = "Calculates OmniDepth's error metrics")
    arg_parser.add_argument("--path", required = True, help = "Directory with the ground truth and the predicted "
                                                                    "depth maps in .exr format, assumes that the predicted depth maps have _pred in their filename")
    args = arg_parser.parse_args()
    PATH = args.path
    gt_depths = []
    pred_depths = []
    count = WIDTH * HEIGHT

    print("Gathering Ground Truth and Prediction files...")
    for file in os.listdir(PATH):
        if "_pred" in file:
            pred_depths.append(file)
        else:
            gt_depths.append(file)
    print("done.")

    # sort so that there is 1-1 correspondence
    pred_depths.sort()
    gt_depths.sort()

    abs_rel_errors = []
    sq_rel_errors = []
    rmse_errors = []
    rmse_log_errors = []
    delta_first_errors = []
    delta_second_errors = []
    delta_third_errors = []

    print("Calculating Metrics...")
    for index in range(len(pred_depths)):
        print("image: {} / {}".format(index, len(gt_depths)))
        gt_path = os.path.join(PATH, gt_depths[index])
        pred_path = os.path.join(PATH, pred_depths[index])
        gt_depth, pred_depth = load_depths(gt_path, pred_path)

        # mask ground truth depth invalid entries
        gt_depth, mask = mask_gt_depth(gt_depth, MASK_THRESH)
        pred_depth = pred_depth[mask]

        # median scaling
        gt_depth, pred_depth = scale_median(gt_depth, pred_depth)

        abs_rel_errors.append(calcAbsRel(gt_depth, pred_depth, count))
        sq_rel_errors.append(calcSqRel(gt_depth, pred_depth, count))
        rmse_errors.append(calcRMSE(gt_depth, pred_depth, count))
        rmse_log_errors.append(calcRMSELog(gt_depth, pred_depth, count))
        delta_first_errors.append(calcDelta(gt_depth, pred_depth, DELTA_THRESH, count))
        delta_second_errors.append(calcDeltaSecondOrder(gt_depth, pred_depth, DELTA_THRESH, count))
        delta_third_errors.append(calcDeltaThirdOrder(gt_depth, pred_depth, DELTA_THRESH, count))

    print("done.")

    # sums
    abs_rel_e = np.asarray(abs_rel_errors)
    sq_rel_e = np.asarray(sq_rel_errors)
    rmse_e = np.asarray(rmse_errors)
    rmse_log_e = np.asarray(rmse_log_errors)
    delta_first_e = np.asarray(delta_first_errors)
    delta_second_e = np.asarray(delta_second_errors)
    delta_third_e = np.asarray(delta_third_errors)

    abs_rel_e = np.sum(abs_rel_e) / len(abs_rel_errors)
    sq_rel_e = np.sum(sq_rel_e) / len(sq_rel_errors)
    rmse_e = np.sum(rmse_e) / len(rmse_errors)
    rmse_log_e = np.sum(rmse_log_e) / len(rmse_log_errors)
    delta_first_e = np.sum(delta_first_e) / len(delta_first_errors)
    delta_second_e = np.sum(delta_second_e) / len(delta_second_errors)
    delta_third_e = np.sum(delta_third_e) / len(delta_third_errors)

    print("Errors:")
    print("Abslute Relative Error: {}".format(abs_rel_e))
    print("Squared Relative Error: {}".format(sq_rel_e))
    print("RMSE: {}".format(rmse_e))
    print("RMSE(log): {}".format(rmse_log_e))
    print("delta < {}: {}". format(DELTA_THRESH, delta_first_e))
    print("delta < {}^2: {}".format(DELTA_THRESH, delta_second_e))
    print("delta < {}^3: {}".format(DELTA_THRESH, delta_third_e))


if __name__ == "__main__":
    main()
