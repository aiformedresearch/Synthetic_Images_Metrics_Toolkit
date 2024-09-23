import json
import matplotlib.pyplot as plt
import os
import argparse

"""
To run this script:

python /home/matteolai/diciotti/matteo/Synthetic_Images_Metrics_Toolkit/plot_subplot_metrics.py --directory /home/matteolai/diciotti/matteo/Synthetic_Images_Metrics_Toolkit/EXPERIMENTS_DATASET_SIZE/size_50

"""
# directory = "/home/matteolai/diciotti/StyleACGAN2-ADA/outdir/00061-ADNI_baseline2D-cond-mirror-auto2-custom/" 
# directory = "/home/matteolai/diciotti/StyleACGAN2-ADA/outdir/00064-ADNI_baseline2D-cond-mirror-auto2-custom-resumecustom/" 

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--directory", type=str, help="Path to the dataframe where the AUCs are stored")
args = parser.parse_args()

directory = args.directory
folder_files = os.listdir(directory)

out_directory = directory+"/metrics_plots/" 

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def read_jsonl_file(allowed_file_names):
    # Collect metrics data in a dictionary
    metrics_data = {}

    # Define the files to be processed
    file_names = [file_name for file_name in folder_files if file_name in allowed_file_names]

    for file_name in file_names:
        with open(directory + '/' + file_name, 'r') as file:
            lines = file.readlines()

        current_metric_list = []
        kimgs = []

        for line in lines:
            data = json.loads(line)
            current_metric = list(data["results"].keys())[0]
            current_metric_list.append(data["results"][current_metric])

            snapshot_pkl = data["snapshot_pkl"]
            code = snapshot_pkl.split('-')[-1].split('.')[0]
            kimgs.append(code.lstrip('0'))

        metrics_data[current_metric] = current_metric_list
    return metrics_data, kimgs

def plot_all_metrics(metrics_dict, kimgs):
    fig, ax1 = plt.subplots(figsize=(20, 6))

    ax2 = ax1.twinx()  # Secondary y-axis 
    ax3 = ax1.twinx()  # Secondary y-axis
    ax3.spines['right'].set_position(('outward', 60))  # Move ax3 further to the right


    for current_metric, current_metric_list in metrics_dict.items():
        start_visualization = 0
        stop_visualization = len(current_metric_list)-1
        kimgs = kimgs[start_visualization:stop_visualization]
        current_metric_list = current_metric_list[start_visualization:stop_visualization]
        if current_metric == 'fid50k_full' or current_metric == 'fid50k':
            ax1.plot(kimgs, current_metric_list, 'r', label=current_metric)
            ax1.plot(kimgs, current_metric_list, 'ro')
        elif current_metric == 'kid50k_full' or current_metric == 'kid50k':
            ax2.plot(kimgs, current_metric_list, 'orange', label=current_metric)
            ax2.plot(kimgs, current_metric_list, 'orange', marker='o')
        elif current_metric == 'ppl_wend':
            ax3.plot(kimgs, current_metric_list, 'dodgerblue', label=current_metric)
            ax3.plot(kimgs, current_metric_list, 'dodgerblue', marker='o')
        elif current_metric == 'ppl2_wend':
            ax3.plot(kimgs, current_metric_list, 'royalblue', label=current_metric)
            ax3.plot(kimgs, current_metric_list, 'royalblue', marker='o')
        elif current_metric == 'ppl_zend':
            ax3.plot(kimgs, current_metric_list, 'navy', label=current_metric)
            ax3.plot(kimgs, current_metric_list, 'navy', marker='o')
        elif current_metric == 'ppl_wfull':
            ax3.plot(kimgs, current_metric_list, 'mediumblue', label=current_metric)
            ax3.plot(kimgs, current_metric_list, 'mediumblue', marker='o')
        elif current_metric == 'ppl_zfull':
            ax3.plot(kimgs, current_metric_list, 'lightsteelblue', label=current_metric)
            ax3.plot(kimgs, current_metric_list, 'lightsteelblue', marker='o')
  
    ax1.set_ylabel('FID', fontsize=18)
    ax2.set_ylabel('KID', fontsize=18)
    ax3.set_ylabel('PPL', fontsize=18)
    ax1.set_xlabel('kimg', fontsize=18)

    ax1.set_title('Metrics over kimg', fontsize=18)
    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45, labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax3.tick_params(axis='y', labelsize=15)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()

    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=12)

    plt.tight_layout()

    make_dir(out_directory)
    plt.savefig(out_directory + 'all_metrics.png')
    plt.close()

def plot_precision_recall(metrics_dict, kimgs):

    fig, ax1 = plt.subplots(figsize=(20, 6))

    ax2 = ax1.twinx()  # Secondary y-axis for precision and recall

    for current_metric, current_metric_list in metrics_dict.items():
        start_visualization = 0
        stop_visualization = len(current_metric_list)
        kimgs = kimgs[start_visualization:stop_visualization]
        current_metric_list = current_metric_list[start_visualization:stop_visualization]

        if current_metric =='pr50k3_full_precision':
            ax1.plot(kimgs, current_metric_list, 'r', label=current_metric)
            ax1.plot(kimgs, current_metric_list, 'ro')
        elif current_metric == 'pr50k3_full_recall':
            ax2.plot(kimgs, current_metric_list, 'b', label=current_metric)
            ax2.plot(kimgs, current_metric_list, 'bo')

    ax1.set_xlabel('kimg', fontsize=18)
    ax1.set_ylabel('Precision', fontsize=18)
    ax2.set_ylabel('Recall', fontsize=18)

    ax1.set_title('Precision-Recall over kimg', fontsize=18)

    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45, labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.03, 1.0), fontsize=12)

    plt.tight_layout()

    make_dir(out_directory)
    plt.savefig(out_directory + 'all_metrics_pr.png')
    plt.close()

def plot_prdc(metrics_dict, kimgs):

    fig, ax1 = plt.subplots(figsize=(20, 6))

    ax2 = ax1.twinx()  # Secondary y-axis for precision and recall
    ax3 = ax1.twinx()  # Secondary y-axis for density
    ax3.spines['right'].set_position(('outward', 60))  # Move ax3 further to the right
    ax4 = ax1.twinx()  # Secondary y-axis for coverage
    ax4.spines['right'].set_position(('outward', 120))  # Move ax4 further to the right

    for current_metric, current_metric_list in metrics_dict.items():
        start_visualization = 0
        stop_visualization = len(current_metric_list)
        kimgs = kimgs[start_visualization:stop_visualization]
        current_metric_list = current_metric_list[start_visualization:stop_visualization]

        if current_metric =='precision':
            ax1.plot(kimgs, current_metric_list, 'r', label=current_metric)
            ax1.plot(kimgs, current_metric_list, 'ro')
        elif current_metric == 'recall':
            ax2.plot(kimgs, current_metric_list, 'b', label=current_metric)
            ax2.plot(kimgs, current_metric_list, 'bo')
        if current_metric =='density':
            ax1.plot(kimgs, current_metric_list, 'y', label=current_metric)
            ax1.plot(kimgs, current_metric_list, 'yo')
        elif current_metric == 'coverage':
            ax2.plot(kimgs, current_metric_list, 'c', label=current_metric)
            ax2.plot(kimgs, current_metric_list, 'co')

    ax1.set_xlabel('kimg', fontsize=18)
    ax1.set_ylabel('Precision', fontsize=18)
    ax2.set_ylabel('Recall', fontsize=18)
    ax3.set_ylabel('Density', fontsize=18)
    ax4.set_ylabel('Coverage', fontsize=18)

    ax1.set_title('Precision-Recall-Density-Coverage over kimg', fontsize=18)

    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45, labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax3.tick_params(axis='y', labelsize=15)
    ax4.tick_params(axis='y', labelsize=15)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()

    ax1.legend(lines1 + lines2 + lines3 + lines4, labels1 + labels2 + labels3 + labels4, loc='upper left', bbox_to_anchor=(1.03, 1.0), fontsize=12)

    plt.tight_layout()

    make_dir(out_directory)
    plt.savefig(out_directory + 'all_metrics_prdc.png')
    plt.close()

def plot_pr_auth(metrics_dict, kimgs):

    fig, ax1 = plt.subplots(figsize=(20, 6))

    ax2 = ax1.twinx()  # Secondary y-axis for precision and recall
    ax3 = ax1.twinx()  # Secondary y-axis for density
    ax3.spines['right'].set_position(('outward', 60))  # Move ax3 further to the right


    for current_metric, current_metric_list in metrics_dict.items():
        start_visualization = 0
        stop_visualization = len(current_metric_list)
        kimgs = kimgs[start_visualization:stop_visualization]
        current_metric_list = current_metric_list[start_visualization:stop_visualization]

        if current_metric =='a-precision':
            ax1.plot(kimgs, current_metric_list, 'r', label=current_metric)
            ax1.plot(kimgs, current_metric_list, 'ro')
        elif current_metric == 'b-recall':
            ax2.plot(kimgs, current_metric_list, 'b', label=current_metric)
            ax2.plot(kimgs, current_metric_list, 'bo')
        if current_metric =='authenticity':
            ax1.plot(kimgs, current_metric_list, 'g', label=current_metric)
            ax1.plot(kimgs, current_metric_list, 'go')

    ax1.set_xlabel('kimg', fontsize=18)
    ax1.set_ylabel('a-Precision', fontsize=18)
    ax2.set_ylabel('b-Recall', fontsize=18)
    ax3.set_ylabel('Authenticity', fontsize=18)

    ax1.set_title('a-Precision, b-Recall and Authenticity over kimg', fontsize=18)

    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45, labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax3.tick_params(axis='y', labelsize=15)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()

    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left', bbox_to_anchor=(1.03, 1.0), fontsize=12)

    plt.tight_layout()

    make_dir(out_directory)
    plt.savefig(out_directory + 'all_metrics_pr_auth.png')
    plt.close()


# ------------------- FID, KID, PPL -------------------

allowed_file_names = ["metric-fid50k_full.jsonl", "metric-fid50k.jsonl", 
    "metric-kid50k_full.jsonl", "metric-kid50k.jsonl",
     "metric-ppl_wend.jsonl", "metric-ppl_wfull.jsonl", "metric-ppl_zend.jsonl", "metric-ppl_zfull.jsonl", "metric-ppl2_wend.jsonl"
     ]

metrics, kimgs = read_jsonl_file(allowed_file_names)
plot_all_metrics(metrics, kimgs)

# ------------------- Precision and Recall -------------------

precision_list = []
recall_list = []
kimgs_pr = []

# Collect metrics data in a dictionary
metrics_data = {}
pr_allowed_file_names = ["metric-pr50k3_full.jsonl", "metric-pr50k3.jsonl"]
pr_file_names = [file_name for file_name in folder_files if file_name in pr_allowed_file_names]
pr_file_name = pr_file_names[0]

with open(directory + "/" + pr_file_name, 'r') as file:
    lines = file.readlines()

for line in lines:
    data = json.loads(line)
    snapshot_pkl = data["snapshot_pkl"]
    code = snapshot_pkl.split('-')[-1].split('.')[0]
    kimgs_pr.append(code.lstrip('0'))

    for current_metric in list(data["results"].keys()):
        if current_metric == 'pr50k3_full_precision' or current_metric == 'pr50k3_precision':
            precision_list.append(data["results"][current_metric])
        elif current_metric == 'pr50k3_full_recall' or current_metric == 'pr50k3_recall':
            recall_list.append(data["results"][current_metric])

metrics_data['pr50k3_full_precision'] = precision_list
metrics_data['pr50k3_full_recall'] = recall_list

plot_precision_recall(metrics_data, kimgs_pr)


# ------------------- Precision, Recall, Density and Coverage -------------------

file_name_prdc = "metric-prdc50k.jsonl"

precision_list = []
recall_list = []
density_list = []
coverage_list = []
kimgs_prdc = []

# Collect metrics data in a dictionary
prdc_data = {}

with open(directory + "/" + file_name_prdc, 'r') as file:
    lines = file.readlines()

for line in lines:
    data = json.loads(line)
    snapshot_pkl = data["snapshot_pkl"]
    code = snapshot_pkl.split('-')[-1].split('.')[0]
    kimgs_prdc.append(code.lstrip('0'))

    for current_metric in list(data["results"].keys()):
        if current_metric == 'precision':
            precision_list.append(data["results"][current_metric])
        elif current_metric == 'recall':
            recall_list.append(data["results"][current_metric])
        elif current_metric == 'density':
            density_list.append(data["results"][current_metric])
        elif current_metric == 'coverage':
            coverage_list.append(data["results"][current_metric])

prdc_data['precision'] = precision_list
prdc_data['recall'] = recall_list
prdc_data['density'] = density_list
prdc_data['coverage'] = coverage_list

plot_prdc(prdc_data, kimgs_prdc)


# ------------------- a-Precision, b-Recall, and Authenticity -------------------
file_name_pr_auth = "metric-pr_auth.jsonl"

precision_list = []
recall_list = []
authenticity_list = []
kimgs_pr_auth = []

# Collect metrics data in a dictionary
pr_auth_data = {}

with open(directory + "/" + file_name_pr_auth, 'r') as file:
    lines = file.readlines()

for line in lines:
    data = json.loads(line)
    snapshot_pkl = data["snapshot_pkl"]
    code = snapshot_pkl.split('-')[-1].split('.')[0]
    kimgs_pr_auth.append(code.lstrip('0'))

    for current_metric in list(data["results"].keys()):
        if current_metric == 'a_precision_c':
            precision_list.append(data["results"][current_metric])
        elif current_metric == 'b_recall_c':
            recall_list.append(data["results"][current_metric])
        elif current_metric == 'authenticity_c':
            authenticity_list.append(data["results"][current_metric])


pr_auth_data['a-precision'] = precision_list
pr_auth_data['b-recall'] = recall_list
pr_auth_data['authenticity'] = authenticity_list


plot_pr_auth(pr_auth_data, kimgs_pr_auth)