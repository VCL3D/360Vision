########################################################################################################################
# Implements the depth metrics used in:
#
#   Eigen, D., Puhrsch, C. and Fergus, R., 2014. 
#   "Depth map prediction from a single image using a multi-scale deep network". 
#   In Advances in neural information processing systems (pp. 2366-2374).
#
#
# Abs Relative Difference:
# Sum{ |Prediction - GroundTruth| / GroundTruth } / |Num of Samples|
#
# Squared Relative Difference:
# Sum{ |Prediction - GroundTruth|^2 / GroundTruth } / |Num of Samples|
#
# RMSE (linear):
# sqrt{ Sum{ |Prediction - GroundTruth|^2 } / |Num of Samples| }
#
# RMSE (log)
# sqrt{ Sum{ |log(Prediction) - log(GroundTruth)|^2 } / |Num of Samples| }
#
# Thresholds (deltas)
# Threshold % subject to GroundTruth: max(Prediction / GroundTruth, GroundTruth / Prediction)
# 
# This script assumes that Ground Truth files, as well as Prediction files are in the same directory, and 
# the Prediction files are suffixed with "_pred".
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
    sq_rel = np.sum(np.square(np.abs(gt - pred)) / gt)
    sq_rel = sq_rel / count
    return sq_rel

def calcRMSE(gt, pred, count):
    rmse = np.sum(np.square(np.abs(pred - gt)))
    rmse = rmse / count
    rmse = math.sqrt(rmse)
    return rmse

def calcRMSELog(gt, pred, count):
    mask = pred > 0.0
    rmse = np.sum(np.square(np.log(pred[mask]) - np.log(gt[mask])))
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
        print("image: {}".format(index))
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

    abs_rel_e = np.mean(abs_rel_e)
    sq_rel_e = np.mean(sq_rel_e)
    rmse_e = np.mean(rmse_e)
    rmse_log_e = np.mean(rmse_log_e)
    delta_first_e = np.mean(delta_first_e)
    delta_second_e = np.mean(delta_second_e)
    delta_third_e = np.mean(delta_third_e)

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
