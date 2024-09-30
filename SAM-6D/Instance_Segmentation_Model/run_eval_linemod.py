import argparse
import os
import re
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

warnings.simplefilter(action="ignore", category=FutureWarning)


def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def get_detect_files(folder, pattern):
    """
    Return all file name that match the given format
    """
    pattern = re.compile(pattern)

    x_values = []

    # List all files in the given folder
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            # Extract the value of x and append to the list
            x_values.append(match.group(1))

    # Sort the x values
    x_values.sort()
    return x_values


def calculate_iou(mask1, mask2):
    """
    Compute IoU on 2 masks
    """
    # Convert masks to boolean arrays
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # Compute IoU
    iou = intersection / union if union != 0 else 0
    return iou


def compute_metrics(mask1, mask2):
    """
    Compute IoU, Dice, Pixel Accuracy, Precision, Recall, Specificity from 2 masks
    """
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    TP = np.logical_and(mask1, mask2).sum()
    FP = np.logical_and(~mask1, mask2).sum()
    FN = np.logical_and(mask1, ~mask2).sum()
    TN = np.logical_and(~mask1, ~mask2).sum()

    IoU = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    Dice = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    Pixel_Accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    Specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    return {
        "IoU": IoU,
        "Dice": Dice,
        "Pixel Accuracy": Pixel_Accuracy,
        "Precision": Precision,
        "Recall": Recall,
        "Specificity": Specificity,
    }


def main(config):
    # Init eval result df
    results_df = pd.DataFrame(
        columns=[
            "Object",
            "NumImages",
            "IoU",
            "Dice",
            "Pixel Accuracy",
            "Precision",
            "Recall",
            "Specificity",
        ]
    )

    for obj_id in config["OBJ_ID"]:
        obj_id = str(obj_id).zfill(2)
        obj_result_dir = os.path.join(config["RESULT_DIR"], "sam6d_results", obj_id)

        mean_metrics = {
            "IoU": [],
            "Dice": [],
            "Pixel Accuracy": [],
            "Precision": [],
            "Recall": [],
            "Specificity": [],
        }

        detected_image_id = get_detect_files(
            obj_result_dir,
            pattern=r"detection_ism_(\d{4})\.npz",
        )

        num_images = len(detected_image_id)
        print(f"Evaluating OBJ: {obj_id} | {num_images} images.")

        for image_id in detected_image_id:
            # Mask gt
            mask_gt_path = os.path.join(
                config["DATA_DIR"], obj_id, "mask", f"{image_id}.png"
            )
            mask_gt = cv2.imread(mask_gt_path, cv2.IMREAD_GRAYSCALE)

            npz_path = os.path.join(
                obj_result_dir,
                f"detection_ism_{image_id}.npz",
            )
            results = np.load(npz_path)

            # Best score mask
            best_iou = -1
            best_mask = None
            for mask_pred in results["segmentation"]:
                iou = calculate_iou(mask_pred, mask_gt)

                if best_iou < iou:
                    best_iou = iou
                    best_mask = mask_pred
            # best_idx = np.argmax(results['score'])
            # pred_mask = results['segmentation'][best_idx]

            metrics = compute_metrics(best_mask, mask_gt)

            # Append metrics to the respective lists
            for key in mean_metrics.keys():
                mean_metrics[key].append(metrics[key])
            # print(im_id, best_iou)

        # Calculate mean metrics
        mean_results = {key: np.mean(value) for key, value in mean_metrics.items()}

        # Add a new row to the DataFrame for the mean metrics
        new_row = {"Object": obj_id, "NumImages": num_images, **mean_results}

        results_df = pd.concat(
            [results_df, pd.DataFrame(new_row, index=[0])], ignore_index=True
        )

        # Print the mean results
        for metric, mean_value in mean_results.items():
            print(f"Mean {metric}: {mean_value:.6f}")
        print("\n----------------\n")

    # Save to CSV
    csv_path = os.path.join(config["RESULT_DIR"], config["OUTPUT_CSV"])

    results_df.to_csv(csv_path, index=False)


if __name__ == "__main__":

    # Parse Argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/eval/run_eval_linemod.yaml",
        help="Path to eval config yaml file",
    )

    args = parser.parse_args()
    config = load_yaml(args.config)
    print(config, "\n----------------\n")

    main(config)
