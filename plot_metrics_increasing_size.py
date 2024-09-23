import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_jsonl(file_path):
    """Reads a JSONL file and returns a list of dictionaries."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_results(data):
    """Extracts the values from the 'results' field of the JSON and returns them as a numpy array."""
    if not data:
        return np.array([])  # Return empty array if no data
    
    # Get the list of result keys from the first entry
    first_results_keys = list(data[0]['results'].keys())
    
    # Extract the values of the result keys from each entry
    results_array = []
    for entry in data:
        results_values = [entry['results'].get(key, None) for key in first_results_keys]
        results_array.append(results_values)
    
    return np.array(results_array), first_results_keys

def create_results_df(file_path):
    """Creates a pandas DataFrame from the results array and keys from a single JSONL file."""
    # Read the JSONL file
    data = read_jsonl(file_path)
    
    # Extract the results and keys
    results_array, result_keys = extract_results(data)

    # Convert the results to a pandas DataFrame
    results_df = pd.DataFrame(results_array, columns=result_keys)

    return results_df

def create_results_df_final(directory_path0, folders, allowed_file_names):
    """Creates a pandas DataFrame concatenating the results array and keys 
        from all the JSONL files for each metric and for each dataset size.
        
        The results are stored in a DataFrame with a multi-index:
        
        size       index
                |          | metric1 | metric2 | metric3 | ...
        size_50 |          |   ...   |   ...   |   ...   | ...
                |    0     |   ...   |   ...   |   ...   | ...
                |    1     |   ...   |   ...   |   ...   | ...
                |    ...   |   ...   |   ...   |   ...   | ...
                | kimg_max |   ...   |   ...   |   ...   | ...
        size_100|          |   ...   |   ...   |   ...   | ...
                |    0     |   ...   |   ...   |   ...   | ...
                |    1     |   ...   |   ...   |   ...   | ...
                |    ...   |   ...   |   ...   |   ...   | ...
                | kimg_max |   ...   |   ...   |   ...   | ...
        size_500|          |   ...   |   ...   |   ...   | ...
        ...     |          |   ...   |   ...   |   ...   | ...

        """
    # Initialize list to store dataframes for all sizes
    all_size_dfs = []
    
    for folder in folders:
        directory_path = directory_path0 + folder
        file_names = os.listdir(directory_path)

        # Filter file names to keep only allowed files
        file_names = [file_name for file_name in file_names if file_name in allowed_file_names]

        # Initialize a DataFrame for this size
        results_df_cur_size = pd.DataFrame()

        for file_name in file_names:    
            file_path = directory_path + file_name
            results_df = create_results_df(file_path)

            # Concatenate along columns (axis=1) for the current size's metrics
            if results_df_cur_size.empty:
                results_df_cur_size = results_df
            else:
                results_df_cur_size = pd.concat([results_df_cur_size, results_df], axis=1)

            #print(f"Results Array shape for file {file_name}: {results_df_cur_size.shape}")
        
        # Add a new level to index corresponding to the folder (size)
        results_df_cur_size['size'] = folder  # Add size as a column

        all_size_dfs.append(results_df_cur_size)

        #print(f"Results Array shape for folder {folder}: {results_df_cur_size.shape}")
    
    # Concatenate all sizes along a new axis (size) and set the multi-index
    results_df_final = pd.concat(all_size_dfs)
    results_df_final = results_df_final.set_index(['size', results_df_final.index])

    print(f"Final DataFrame shape: {results_df_final.shape}")

    return results_df_final

def plot_metrics_for_each_kimg(results_df_final, metrics, filenames, sizes, figures_path):
    """Plot the metrics for each dataset size for each kimgs."""

    kimg_max = len(results_df_final.loc[f"size_{sizes[0]}/"][metrics[0]])
    for i, metric in enumerate(metrics):
        for kimgs in range(kimg_max):
            y_value = []
            for size in sizes:
                #print(f"Metric: {metric}, Size: {size}, Kimgs: {kimgs*200}")
                y_value.append(results_df_final.loc[f"size_{size}/", kimgs][metric])

            fig, ax = plt.subplots(figsize=(10, 7))

            ax.plot(sizes, y_value, marker='o', linestyle='-', label=metric)
            #ax.plot(sizes, y_value)
            ax.set_title(f"{metric} - Kimgs: {kimgs*200}", fontsize=20)
            ax.set_xlabel("Dataset size", fontsize=15)
            ax.set_ylabel(metric, fontsize=15)
            ax.set_xticks(sizes)
            ax.set_xticklabels(sizes, rotation=45, fontsize=15)
            ax.tick_params(axis='y', labelsize=15)
        
            ax.legend(fontsize=12)
            if metric != ['fid50k_full', 'kid50k_full'] and metric != ['ppl_zfull']:
                ax.set_ylim([-0.05, 1.05])

            plt.tight_layout()
            plt.savefig(f"{figures_path}/{filenames[i]}_{kimgs*200}.png")
            plt.close()

    print("Saved the plots with the metrics for each kimg as: {metrics}_{kimgs}.png")


def plot_best_metric_for_each_ds(results_df_final, metrics, filenames, sizes, figures_path):
    """Plot the best results for each metric for each dataset size."""

    for i, metric in enumerate(metrics):
        y_value = []
        for size in sizes:
            #print(f"Metric: {metric}, Size: {size}, Kimgs: {kimgs*200}")
            df_cur_size = results_df_final.loc[f"size_{size}/"][metric]

            # Get the best value for each metric
            if metric == ['fid50k_full', 'kid50k_full'] or metric == ['ppl_zfull']:
                best_value = df_cur_size.min(axis=0)
            else:
                best_value = df_cur_size.max(axis=0)
            
            # Append the best value for the current size
            y_value.append(best_value)

        fig, ax = plt.subplots(figsize=(10, 7))

        ax.plot(sizes, y_value, marker='o')
        ax.plot(sizes, y_value)
        ax.set_title(f"{metric} best value", fontsize=20)
        ax.set_xlabel("Dataset size", fontsize=15)
        ax.set_ylabel(metric, fontsize=15)
        ax.set_xticks(sizes)
        ax.set_xticklabels(sizes, rotation=45, fontsize=15)
        ax.tick_params(axis='y', labelsize=15)
        if metric != ['fid50k_full', 'kid50k_full'] and metric != ['ppl_zfull']:
            ax.set_ylim([-0.05, 1.05])

        plt.tight_layout()
        plt.savefig(f"{figures_path}/{filenames[i]}_best.png")
        plt.close()
    
    print("Saved best metric for each dataset size as: {metrics}_best.png")

def plot_summary(results_df_final, metrics, filenames, sizes, figures_path):
    """Plot a summary value for realness, diversity and authenticity."""

    all_metrics = results_df_final.columns
    kimg_max = len(results_df_final.loc[f"size_{sizes[0]}/"][metrics[0]])
    for kimgs in range(kimg_max):
        realness_values = []
        diversity_values = []
        authenticity_values = []
        for size in sizes:
            realness_value = 0
            diversity_value = 0
            authenticity_value = 0

            for i, metric in enumerate(all_metrics):
                # Normalize each metric in the range [0, 1]
                realness_metrics = [['kid50k_full'],# 'fid50k_full'], 
                        ['pr50k3_full_precision', 'a_precision_c', 'precision', 'density']]# a_precision_m
                diversity_metrics = ['pr50k3_full_recall', 'b_recall_c', 'recall', 'coverage']# b_recall_m
                authenticity_metrics = ['authenticity_c']#, 'authenticity_m']
                                        # m: mean   --> embedding center = np.mean(real_features,axis=0)
                                        # c: center --> embedding center = [10, 10, ..., 10]

                if metric in realness_metrics[0]:
                    value_max = results_df_final.loc[f"size_{size}/"][metric][0]
                    value = 1 - results_df_final.loc[f"size_{size}/"][metric][kimgs] / value_max
                    realness_value += value
                elif metric in ['ppl_zfull']:
                    # value_max = results_df_final.loc[f"size_{size}/"][metric][0]
                    # value = 1 - results_df_final.loc[f"size_{size}/"][metric][0] / value_max
                    # realness_value += value
                    pass
                elif metric in realness_metrics[1]: 
                    realness_value += results_df_final.loc[f"size_{size}/"][metric][kimgs]
                elif metric in diversity_metrics:
                    diversity_value += results_df_final.loc[f"size_{size}/"][metric][kimgs]
                elif metric in authenticity_metrics:
                    authenticity_value += results_df_final.loc[f"size_{size}/"][metric][kimgs]

            # Normalize the summary values      
            realness_value /= (len(realness_metrics[0])+ len(realness_metrics[1]))
            diversity_value /= len(diversity_metrics)
            authenticity_value /= len(authenticity_metrics)

            # Append the summary values
            realness_values.append(realness_value)
            diversity_values.append(diversity_value)
            authenticity_values.append(authenticity_value)

        fig, ax = plt.subplots(figsize=(10, 7))

        ax.plot(sizes, realness_values, marker='o', color='blue', label='Realness')
        ax.plot(sizes, realness_values, color='blue')
        ax.plot(sizes, diversity_values, marker='o', color='red', label='Diversity')
        ax.plot(sizes, diversity_values, color='red')
        ax.plot(sizes, authenticity_values, marker='o', color='green', label='Authenticity')
        ax.plot(sizes, authenticity_values, color='green')
        ax.set_title("Realness, diversity, and authenticity", fontsize=22)
        ax.set_xlabel("Dataset size", fontsize=18)
        ax.set_ylabel("Quantitative evaluation", fontsize=18)
        ax.set_xticks(sizes)
        ax.set_xticklabels(sizes, rotation=45, fontsize=15)
        ax.tick_params(axis='y', labelsize=15)
        if metric not in realness_metrics[0]:
            ax.set_ylim([-0.05, 1.05])

        ax.legend(loc='best', fontsize=15)

        plt.tight_layout()
        plt.savefig(f"{figures_path}/summary_metrics_{kimgs*200}.png")
        plt.close()

    print("Saved plots with summary values (realness, diversity, and authenticity) as: summary_metrics_{kimg}.png")

def plot_grouped_metrics(results_df_final, metrics, filenames, sizes, figures_path):
    fid_kid = ['fid50k_full', 'kid50k_full']
    realness_metrics = ['pr50k3_full_precision', 'a_precision_c', 'precision', 'density']
    diversity_metrics = ['pr50k3_full_recall', 'b_recall_c', 'recall', 'coverage']
    authenticity_metrics = ['authenticity_c']

    
    for cur_metrics in [fid_kid, realness_metrics, diversity_metrics, authenticity_metrics]:
        for kimgs in range(len(results_df_final.loc[f"size_{sizes[0]}/"][metrics[0]])):  
            fig, ax = plt.subplots(figsize=(10, 7))     
            ax1 = ax.twinx()  
            for metric in cur_metrics:
                y_value = []
                for size in sizes:
                    #print(f"Metric: {metric}, Size: {size}, Kimgs: {kimgs*200}")
                    y_value.append(results_df_final.loc[f"size_{size}/"][metric][kimgs])

                if metric != fid_kid[1]:
                    ax.plot(sizes, y_value, marker='o', linestyle='-', label=metric)
                    ax.set_xlabel("Dataset size", fontsize=15)
                    ax.set_ylabel(metric, fontsize=15)
                    ax.set_xticks(sizes)
                    ax.set_xticklabels(sizes, rotation=45, fontsize=15)
                    ax.tick_params(axis='y', labelsize=15)
                    ax.legend(fontsize=15)
                else:
                    ax1.plot(sizes, y_value, marker='o', linestyle='-', color='orange', label=metric)
                    ax1.set_ylabel(metric, fontsize=15)
                    ax1.tick_params(axis='y', labelsize=15)
                    ax1.legend(fontsize=15)

                    
                if cur_metrics == fid_kid:
                    title = "FID&KID"
                elif cur_metrics == realness_metrics:
                    title = "Realness"
                elif cur_metrics == diversity_metrics:
                    title = "Diversity"
                elif cur_metrics == authenticity_metrics:
                    title = "Authenticity"
                ax.set_title(title, fontsize=25)

                if cur_metrics != fid_kid:
                    ax.set_ylim([-0.05, 1.05])

                plt.tight_layout()
            plt.savefig(f"{figures_path}/{title}_{kimgs*200}.png")
            plt.close()
    print("Saved the plots with the metrics grouped as: {title}_{kimgs}.png")

def main(directory_path0):
    # -----------------------------------------------------------
    #        Create the DataFrame to collect the metrics
    # -----------------------------------------------------------

    folders = ["size_50/", "size_100/", "size_500/", "size_1000/", "size_2000/", "size_3227/"]

    allowed_file_names = [
        "metric-fid50k_full.jsonl", "metric-fid50k.jsonl", 
        "metric-kid50k_full.jsonl", "metric-kid50k.jsonl",
        "metric-ppl_wend.jsonl", "metric-ppl_wfull.jsonl", "metric-ppl_zend.jsonl", "metric-ppl_zfull.jsonl", "metric-ppl2_wend.jsonl",
        "metric-pr50k3_full.jsonl", "metric-pr50k3.jsonl",
        "metric-prdc50k.jsonl",
        "metric-pr_auth.jsonl"
    ]

    # Create the DataFrame
    results_df_final = create_results_df_final(directory_path0, folders, allowed_file_names)

    # -----------------------------------------------------------
    #                  Plot the metrics
    # -----------------------------------------------------------

    figures_path = directory_path0 + "/plots_all_sizes/"
    os.makedirs(figures_path, exist_ok=True)
    figures_path2 = directory_path0+"/plots_grouped/"
    os.makedirs(figures_path2, exist_ok=True)

    metrics = [
        ['fid50k_full', 'kid50k_full'], 
        ['pr50k3_full_precision', 'pr50k3_full_recall'], 
        ['ppl_zfull'], 
        ['a_precision_c', 'b_recall_c', 'authenticity_c'], 
        ['a_precision_m', 'b_recall_m', 'authenticity_m'], 
        ['precision', 'recall', 'density', 'coverage']
        ]
    filenames = ["FID_KID", "PR", "PPL", "PR_auth_C", "PR_auth_M", "PRDC"]
    sizes = [50, 100, 500, 1000, 2000, 3227]

    # Plot the metrics for each kimgs
    plot_metrics_for_each_kimg(results_df_final, metrics, filenames, sizes, figures_path)

    # Plot only the best results for each metric
    plot_best_metric_for_each_ds(results_df_final, metrics, filenames, sizes, figures_path)

    # Plot the metrics divided by groups
    plot_grouped_metrics(results_df_final, metrics, filenames, sizes, figures_path2)

    # Plot a summary value for realness, diversity and authenticity
    plot_summary(results_df_final, metrics, filenames, sizes, figures_path)
        
    print(f"Done. All plots saved in {figures_path}")


if __name__ == "__main__":
    directory_path0 = '/home/matteolai/diciotti/matteo/Synthetic_Images_Metrics_Toolkit/EXPERIMENTS_DATASET_SIZE_val/'

    main(directory_path0)