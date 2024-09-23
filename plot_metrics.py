import json
import matplotlib.pyplot as plt
import os
import argparse

"""
To run this script:

python diciotti/matteo/Synthetic_Images_Metrics_Toolkit/plot_metrics.py --directory diciotti/matteo/Synthetic_Images_Metrics_Toolkit/EXPERIMENTS_DATASET_SIZE/size_50

"""
#directory = "/home/matteolai/diciotti/StyleACGAN2-ADA/outdir/00061-ADNI_baseline2D-cond-mirror-auto2-custom/" 

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--directory", type=str, help="Path to the dataframe where the AUCs are stored")
args = parser.parse_args()

directory = args.directory
out_directory = directory+"/metrics_plots/" 
allowed_file_names = ["metric-fid50k_full.jsonl", "metric-fid50k.jsonl", 
    "metric-kid50k_full.jsonl", "metric-kid50k.jsonl",
     "metric-ppl_wend.jsonl", "metric-ppl_wfull.jsonl", "metric-ppl_zend.jsonl", "metric-ppl_zfull.jsonl", "metric-ppl2_wend.jsonl"
     ]

allowed_pr_file_names = ["metric-pr50k3_full.jsonl", "metric-pr50k3.jsonl",
     "metric-pr_auth.jsonl",
     "metric-prdc50k.jsonl"
]

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_metric(current_metric, current_metric_list, kimgs):
    start_visualization = 0
    stop_visualization = len(current_metric_list)
    current_metric_list, kimgs = current_metric_list[start_visualization:stop_visualization], kimgs[start_visualization:stop_visualization]
    plt.figure(figsize=(20, 6))
    plt.plot(kimgs, current_metric_list, marker='o', label=current_metric)
    plt.xlabel('kimg', fontsize=18)
    plt.ylabel(current_metric, fontsize=18)
    plt.title(current_metric + ' metric over kimg', fontsize=18)
    plt.grid(True)
    plt.xticks(rotation=45, fontsize=15)  # Rotate x-axis labels for readability
    plt.yticks(fontsize=15)
    plt.tight_layout()
    #plt.legend(loc='upper left', bbox_to_anchor=(0.75, 0.9), fontsize=12)
    make_dir(out_directory)
    outpath = out_directory + current_metric + '.png'
    plt.savefig(outpath)
    plt.close()

folder_files = os.listdir(directory)
file_names = [file_name for file_name in folder_files if file_name in allowed_file_names]

for file_name in file_names:
    with open(directory+'/'+file_name, 'r') as file:
        lines = file.readlines()
    print('Current metric: ', file_name)

    # Initialize the lists to save the value of each metrics for each kimg
    current_metric_list = []
    kimgs = []

    # Extract the metrics from each line
    for line in lines:
        data = json.loads(line)

        # Extract the metric for the current kimg
        current_metric  = list(data["results"].keys())[0]
        current_metric_list.append(data["results"][current_metric])

        # Extract the current kimg from "snapshot_pkl"
        snapshot_pkl = data["snapshot_pkl"]
        code = snapshot_pkl.split('-')[-1].split('.')[0]
        kimgs.append(code.lstrip('0'))
    
    plot_metric(current_metric, current_metric_list, kimgs)


pr_file_names = [file_name for file_name in folder_files if file_name in allowed_pr_file_names]
for pr_file_name in pr_file_names:
    # Plot precision and recall - note that you have to do it separately as their json file is structured differently
    with open(directory+"/"+pr_file_name, 'r') as file:
        lines = file.readlines()

    # Initialize the lists to save the value of each metrics for each kimg
    precision_list = []
    recall_list = []
    kimgs = []

    # Extract the metrics from each line
    for line in lines:
        data = json.loads(line)

        # Extract the current kimg from "snapshot_pkl"
        snapshot_pkl = data["snapshot_pkl"]
        code = snapshot_pkl.split('-')[-1].split('.')[0]
        kimgs.append(code.lstrip('0'))

        # Extract the metric for the current kimg
        for current_metric in list(data["results"].keys()):
            if current_metric == 'pr50k3_full_precision':
                precision_list.append(data["results"][current_metric])
            elif current_metric == 'pr50k3_full_recall':
                recall_list.append(data["results"][current_metric])

    for current_metric in list(data["results"].keys()):
        if current_metric == 'pr50k3_full_precision':
            plot_metric(current_metric, precision_list, kimgs)
        elif current_metric == 'pr50k3_full_recall':
            plot_metric(current_metric, recall_list, kimgs)
