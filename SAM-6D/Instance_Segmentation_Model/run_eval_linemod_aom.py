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


def compute_hu_moments_normalized(binary_mask):
    # Compute raw Hu Moments
    moments = cv2.moments(binary_mask.astype(np.uint8))
    hu_moments = cv2.HuMoments(moments).flatten()

    # Apply the transformation
    transformed_hu_moments = -np.sign(hu_moments) * np.log(np.abs(hu_moments) + 1e-10)  # Add small value to avoid log(0)
    # Normalize to range [0, 1]
    min_val, max_val = np.min(transformed_hu_moments), np.max(transformed_hu_moments)
    normalized = (transformed_hu_moments - min_val) / (max_val - min_val + 1e-10)  # Avoid division by zero
    return normalized


def hu_moments_similarity(mask1, mask2):

    distance = cv2.matchShapes(mask2, mask1, cv2.CONTOURS_MATCH_I2, 0)
    
    # Calculate similarity as squared difference
    similarity = 1 - distance
    
    # similarity = 1 - np.sqrt(np.sum((hu1 - hu2)**2)) #1-distance
    
    return distance

def newscore(geo,sem,appe,vis_r,hu):
    
    new_score = (sem + appe + (vis_r * geo) + hu ) / (1+1+vis_r+1)
    return new_score


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
            "Score",
        ])

    for obj_id in config["OBJ_ID"]:
        
        info_results_df = pd.DataFrame(
        columns=[
            "Object_id",
            "Img_id",
            "Mask_bestScore_id",
            "bestScIoU",
            "bestScore",
            "Geo_bestscores",
            "Sem_bestscore",
            "App_bestscore",
            "Vis_R_bestscore" ,
            "Hu_bestscore" ,
            "Newsc_bestscore" , 
            "Mask_bestIou_id",
            "bestiouIoU",
            "bestiouScore",
            "Geo_bestiouscores",
            "Sem_bestiouscore",
            "App_bestiouscore",
            "Vis_R_bestiouscore",
            "Hu_bestiouscore",
            "Newsc_bestiouscore",
        ] )
        
        obj_id = str(obj_id).zfill(2)
        obj_result_dir = os.path.join(config["RESULT_DIR"], "sam6d_results_02", obj_id)
        obj_info_dir = os.path.join(config["RESULT_DIR"], "sam6d_results_02_02", obj_id)

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
        
        a = [19, 53, 67, 70, 105, 223, 228, 234, 235, 257, 262, 313, 314, 315, 316, 389, 474, 509, 510, 513, 514, 555, 556, 562, 569, 573, 580, 592, 593, 700, 801, 804, 963, 965, 968, 1123, 247, 679]

        for image_id_num in a:
            image_id = detected_image_id[image_id_num]
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

            npz_info_path = os.path.join(
                obj_info_dir,
                f"detection_{obj_id}_{image_id}.npz",
            )

            info = np.load(npz_info_path)

            # Best score mask
            best_iou = -1
            best_mask = None
            mask_bestiou_id = 0
            for mask_id in range(len(results["segmentation"])):
                mask_pred = results["segmentation"][mask_id]
                iou = calculate_iou(mask_pred, mask_gt)
                if best_iou < iou:
                    best_iou = iou
                    best_mask = mask_pred
                    mask_bestiou_id = mask_id
                    
            best_idx = np.argmax(results['score'])
            pred_mask = results['segmentation'][best_idx]

            mask_id = 0 
            best_sc_iou = calculate_iou(pred_mask, mask_gt)
            best_iou_iou = calculate_iou(best_mask, mask_gt)

            hu_sc_score = hu_moments_similarity(pred_mask, mask_gt)    
            hu_iou_score = hu_moments_similarity(best_mask, mask_gt)

            # geo,sem,appe,vis_r,hu
            newbest_sc_iou = newscore(info['geometric'][best_idx],info['semantic'][best_idx], info['appearance'][best_idx],info['visible_ratio'][best_idx],hu_sc_score) 
            
            newbest_iou_iou = newscore(info['geometric'][mask_bestiou_id],info['semantic'][mask_bestiou_id], info['appearance'][mask_bestiou_id],info['visible_ratio'][mask_bestiou_id],hu_iou_score) 

            new_info = {
                            "Object_id": obj_id,                
                            "Img_id": image_id,                   
                            "Mask_bestScore_id": best_idx,  
                            "bestScIoU": best_sc_iou,             
                            "bestScore": results['score'][best_idx],  
                            "Geo_bestscores": info['geometric'][best_idx],    
                            "Sem_bestscore": info['semantic'][best_idx],     
                            "App_bestscore": info['appearance'][best_idx],
                            "Vis_R_bestscore" : info['visible_ratio'][best_idx],
                            "Hu_bestscore" : hu_sc_score  , 
                            "Newsc_bestscore" : newbest_sc_iou, 
                            "Mask_bestIou_id": mask_bestiou_id,  
                            "bestiouIoU": best_iou_iou,           
                            "bestiouScore": results['score'][mask_bestiou_id],    
                            "Geo_bestiouscores": info['geometric'][mask_bestiou_id],
                            "Sem_bestiouscore": info['semantic'][mask_bestiou_id],  
                            "App_bestiouscore": info['appearance'][mask_bestiou_id],  
                            "Vis_R_bestiouscore" : info['visible_ratio'][mask_bestiou_id],
                            "Hu_bestiouscore" : hu_iou_score, 
                            "Newsc_bestiouscore" : newbest_iou_iou,
                        }   
            info_results_df = pd.concat(
                    [info_results_df, pd.DataFrame(new_info, index=[0])], ignore_index=True)
    
            metrics = compute_metrics(pred_mask, mask_gt)

            # Append metrics to the respective lists
            for key in mean_metrics.keys():
                mean_metrics[key].append(metrics[key])

        #save info path
        info_csv_path = os.path.join(config["OUTPUT_CSV"])  
        if not os.path.exists(info_csv_path):
                os.makedirs(info_csv_path)
            
        info_csv_path_sc = os.path.join(config["OUTPUT_CSV"],  f"info_ism_{obj_id}.csv" )  
        info_results_df.to_csv(info_csv_path_sc, index=False)
        print("\n------save-info------\n")
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
        csv_path = os.path.join(config["OUTPUT_CSV"], 'ism_eval_score.csv')
    
        results_df.to_csv(csv_path, index=False)


if __name__ == "__main__":

    # Parse Argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/eval/run_eval_linemod_sam_rpd_aom.yaml",
        help="Path to eval config yaml file",
    )

    args = parser.parse_args()
    config = load_yaml(args.config)
    print(config, "\n----------------\n")

    main(config)
