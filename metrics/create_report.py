# SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Spacer, Paragraph, PageBreak, ListFlowable, ListItem, Flowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

from metrics import metric_utils
import dnnlib

# Define the legend mapping metric keys to human-readable labels
metric_labels = {
    "fid": "FID",
    "kid": "KID",
    "is_mean": "IS",
    "precision": "Precision",
    "pr_precision": "Precision (NVIDIA)",
    "density": "Density",
    "a_precision_c": "α-Precision",
    "recall": "Recall",
    "pr_recall": "Recall (NVIDIA)",
    "coverage": "Coverage",
    "b_recall_c": "β-Recall",
    "authenticity_c": "Authenticity"
}

metric_references = {
    "FID": "Paper: <a href='https://arxiv.org/abs/1706.08500'>\"GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium\"</a>, Heusel et al. 2017. <br/>  Implementation: Karras et al., https://github.com/NVlabs/stylegan2-ada-pytorch",
    "KID": "Paper: <a href='https://arxiv.org/abs/1801.01401'>\"Demystifying MMD GANs\"</a>, Binkowski et al. 2018 <br/>  Implementation: Karras et al., https://github.com/NVlabs/stylegan2-ada-pytorch",
    "IS": "Paper: <a href='https://arxiv.org/abs/1606.03498'>\"Improved Techniques for Training GANs\"</a>, Salimans et al. 2016 <br/>  Implementation: Karras et al., https://github.com/NVlabs/stylegan2-ada-pytorch",
    "Precision (NVIDIA)": "Paper: <a href='https://arxiv.org/abs/1904.06991'>\"Improved Precision and Recall Metric for Assessing Generative Models\"</a>, Kynkäänniemi et al. 2019 <br/>  Implementation: Karras et al, https://github.com/NVlabs/stylegan2-ada-pytorch",
    "Recall (NVIDIA)": "Paper: <a href='https://arxiv.org/abs/1904.06991'>\"Improved Precision and Recall Metric for Assessing Generative Models\"</a>, Kynkäänniemi et al. 2019 <br/>  Implementation: Karras et al, https://github.com/NVlabs/stylegan2-ada-pytorch",
    "Precision": "Paper: <a href='https://arxiv.org/abs/1904.06991'>\"Improved Precision and Recall Metric for Assessing Generative Models\"</a>, Kynkäänniemi et al. 2019 <br/>  Implementation: Naeem et al., https://github.com/clovaai/generative-evaluation-prdc",
    "Recall": "Paper: <a href='https://arxiv.org/abs/1904.06991'>\"Improved Precision and Recall Metric for Assessing Generative Models\"</a>, Kynkäänniemi et al. 2019 <br/>  Implementation: Naeem et al., https://github.com/clovaai/generative-evaluation-prdc",
    "Density": "Paper: <a href='https://proceedings.mlr.press/v119/naeem20a/naeem20a.pdf'>\"Reliable Fidelity and Diversity Metrics for Generative Models\"</a>, Naeem et al., 2020 <br/>  Implementation: Naeem et al., https://github.com/clovaai/generative-evaluation-prdc",
    "Coverage": "Paper: <a href='https://proceedings.mlr.press/v119/naeem20a/naeem20a.pdf'>\"Reliable Fidelity and Diversity Metrics for Generative Models\"</a>, Naeem et al., 2020 <br/>  Implementation: Naeem et al., https://github.com/clovaai/generative-evaluation-prdc",
    "α-Precision": "Paper: <a href='https://proceedings.mlr.press/v162/alaa22a/alaa22a.pdf'>\"How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models\"</a>, Alaa et al., 2022 <br/>  Implementation: Alaa et al., https://github.com/vanderschaarlab/evaluating-generative-models",
    "β-Recall": "Paper: <a href='https://proceedings.mlr.press/v162/alaa22a/alaa22a.pdf'>\"How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models\"</a>, Alaa et al., 2022 <br/>  Implementation: Alaa et al., https://github.com/vanderschaarlab/evaluating-generative-models",
    "Authenticity": "Paper: <a href='https://proceedings.mlr.press/v162/alaa22a/alaa22a.pdf'>\"How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models\"</a>, Alaa et al., 2022 <br/>  Implementation: Alaa et al., https://github.com/vanderschaarlab/evaluating-generative-models",
}

def add_page_number(canvas, doc):
    """
    Adds page numbers at the bottom of each page.
    """
    page_num = canvas.getPageNumber()
    text = f"{page_num}"
    canvas.setFont("Helvetica", 10)
    canvas.drawRightString(7.5 * inch, 0.5 * inch, text)

class TableAndImage(Flowable):
    def __init__(self, table, image_path, img_width=200, img_height=200):
        super().__init__()
        self.table = table
        self.image_path = image_path
        self.img_width = img_width
        self.img_height = img_height
    
    def draw(self):
        pass

    def wrap(self, availWidth, availHeight):
        return availWidth, availHeight

    def split(self):
        return [self.table, Image(self.image_path, width=self.img_width, height=self.img_height)]

def extract_metrics_from_csv(folder_path):
    csv_path = os.path.join(folder_path, 'metrics.csv')
    if not os.path.exists(csv_path):
        print(f"No CSV file found at {csv_path}")
        return {}

    known_flags = {"fid", "kid", "is_", "pr", "prdc", "pr_auth"}
    metrics = {}

    with open(csv_path, newline='') as csvfile:
        reader = list(csv.DictReader(csvfile))
        if not reader:
            return {}

        # Filter rows to only known metrics
        filtered_rows = [row for row in reader if row['flag'] in known_flags]

        # Pick only the latest row for each submetric
        for row in reversed(filtered_rows):
            metric = row['metric']
            if metric not in metrics:
                try:
                    value = float(row['score'])
                except ValueError:
                    value = row['score']
                metrics[metric] = value

    return metrics

def plot_metrics_triangle(metrics, metric_folder):

    # Extract fidelity, diversity, and generalization metrics while filtering out None values
    fidelity_metrics = {k: metrics.get(k, None) for k in ["precision", "pr50k3_precision", "density", "a_precision_c"]}
    fidelity_metrics = {k: np.clip(v, 0, 1) for k, v in fidelity_metrics.items() if v is not None}
    fidelity_mean = np.mean(list(fidelity_metrics.values()))

    diversity_metrics = {k: metrics.get(k, None) for k in ["recall", "pr50k3_recall", "coverage", "b_recall_c"]}
    diversity_metrics = {k: np.clip(v, 0, 1) for k, v in diversity_metrics.items() if v is not None}
    diversity_mean = np.mean(list(diversity_metrics.values()))

    generalization_metrics = {k: np.clip(metrics.get(k, None), 0, 1) for k in ["authenticity_c"] if metrics.get(k) is not None}
    generalization_mean = np.mean(list(generalization_metrics.values())) if generalization_metrics else 0

    # Filter labels to match available metrics
    selected_metrics = set(fidelity_metrics.keys()) | set(diversity_metrics.keys()) | set(generalization_metrics.keys())
    filtered_metric_labels = {k: v for k, v in metric_labels.items() if k in selected_metrics}

    triangle_vertices = np.array([
        [0, 1],
        [-np.sqrt(3.7)/2, -0.9],
        [np.sqrt(3.7)/2, -0.9]
    ])
    centroid = np.mean(triangle_vertices, axis=0)

    fig, ax = plt.subplots(dpi=600)
    ax.set_aspect('equal')
    
    for i in range(3):
        start, end = triangle_vertices[i], triangle_vertices[(i + 1) % 3]
        ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', lw=2)
    
    # Place labels on the triangle vertices
    ax.text(0, 1.10, 'Generalization', ha='center', fontsize=15, weight='bold', color='blue')
    ax.text(-np.sqrt(4)/2 - 0.05, -1, 'Diversity', ha='center', fontsize=15, weight='bold', color='red')
    ax.text(np.sqrt(4)/2 + 0.02, -1, 'Fidelity', ha='center', fontsize=15, weight='bold', color='green')
    ax.text(0, 1.01, '1', ha='center', fontsize=15)
    ax.text(-np.sqrt(3.7)/2 - 0.05, -0.88, '1', ha='center', fontsize=15)
    ax.text(np.sqrt(3.7)/2 + 0.05, -0.88, '1', ha='center', fontsize=15)
    ax.text(centroid[0]+0.05, centroid[1]+0.01, '0', ha='center', fontsize=15)

    # Plot the centroid (center point for [0,0,0] metrics)
    ax.scatter(centroid[0], centroid[1], color='black', zorder=5)

    # Scale the means along the axes from the centroid to the vertices
    metrics_means = np.array([generalization_mean, diversity_mean, fidelity_mean])
    scaled_points = centroid + metrics_means[:, None] * (triangle_vertices - centroid)

    # Plot the mean points on the triangle
    ax.scatter(scaled_points[:, 0], scaled_points[:, 1], color=['blue', 'red', 'forestgreen'], s=100, zorder=10)
    ax.plot(scaled_points[:, 0], scaled_points[:, 1], 'gray', linestyle='--', lw=2)
    
    # Draw semi-transparent lines from centroid to each vertex
    ax.plot([centroid[0], triangle_vertices[0][0]], [centroid[1], triangle_vertices[0][1]], color='blue', alpha=0.3, lw=1)  # Line to Generalization
    ax.plot([centroid[0], triangle_vertices[1][0]], [centroid[1], triangle_vertices[1][1]], color='red', alpha=0.3, lw=1)  # Line to Diversity
    ax.plot([centroid[0], triangle_vertices[2][0]], [centroid[1], triangle_vertices[2][1]], color='green', alpha=0.3, lw=1)  # Line to Fidelity
    
    # Define symbols for different metric categories
    fidelity_symbols = ['o', 'v', 'D', 's']  # Circle, triangle, diamond, square
    diversity_symbols = ['p', '^', '*', 'X']  # Pentagon, up triangle, star, cross
    generalization_symbols = ['H']  # Hexagon

    # Plot individual fidelity metrics with different shades of green
    fidelity_metric_points = []
    for (metric, value), symbol in zip(fidelity_metrics.items(), fidelity_symbols):
        if metric is not None:
            scaled_fidelity = centroid + value * (triangle_vertices[2] - centroid)
            point = ax.scatter(scaled_fidelity[0], scaled_fidelity[1], marker=symbol, color='lime', alpha=0.7, s=50, zorder=15)
            fidelity_metric_points.append((point, filtered_metric_labels[metric]))

    # Plot individual diversity metrics with different shades of red
    diversity_metric_points = []
    for (metric, value), symbol in zip(diversity_metrics.items(), diversity_symbols):
        scaled_diversity = centroid + value * (triangle_vertices[1] - centroid)
        point = ax.scatter(scaled_diversity[0], scaled_diversity[1], marker=symbol, color="firebrick", alpha=0.7, s=50, zorder=15)
        diversity_metric_points.append((point, filtered_metric_labels[metric]))

    # Plot generalization metrics with blue
    generalization_metric_points = []
    for (metric, value), symbol in zip(generalization_metrics.items(), generalization_symbols):
        if metric is not None:
            scaled_generalization = centroid + value * (triangle_vertices[0] - centroid)
            point = ax.scatter(scaled_generalization[0], scaled_generalization[1], marker=symbol, color="lightskyblue", alpha=0.7, s=50, zorder=15)
            generalization_metric_points.append((point, filtered_metric_labels[metric]))

    # Connect the mean points with lines to form a triangle inside the main triangle
    ax.plot([scaled_points[0][0], scaled_points[1][0]], [scaled_points[0][1], scaled_points[1][1]], 'gray', linestyle='--', lw=2)
    ax.plot([scaled_points[1][0], scaled_points[2][0]], [scaled_points[1][1], scaled_points[2][1]], 'gray', linestyle='--', lw=2)
    ax.plot([scaled_points[2][0], scaled_points[0][0]], [scaled_points[2][1], scaled_points[0][1]], 'gray', linestyle='--', lw=2)

    # Build the legend dynamically
    legend_handles = [p[0] for p in fidelity_metric_points + diversity_metric_points + generalization_metric_points]
    legend_labels = [p[1] for p in fidelity_metric_points + diversity_metric_points + generalization_metric_points]
    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(-0.35, 1.1), fontsize=12.5)
        
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(metric_folder, "figures/metrics_triangle.png")
    plt.savefig(metric_utils.get_unique_filename(plot_path))
    plt.close()

def save_metrics_to_pdf(args, metrics, metric_folder, out_pdf_path):
    doc = SimpleDocTemplate(out_pdf_path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    # Create table data: header row + rows for each metric present in both `metrics` and `metric_labels`
    ref_mapping = {}
    references = []
    ref_counter = 1
    data = [["Metric [ref]", "Value", "Range"]]
    for key, label in metric_labels.items():
        value = metrics.get(key, None)
        if value is not None:
            ref_text = metric_references[label]
            if ref_text not in ref_mapping:
                ref_mapping[ref_text] = ref_counter
                references.append(f"[{ref_counter}] {ref_text}")
                ref_counter += 1
            ref_number = ref_mapping[ref_text]
            metric_display = f"{label} [{ref_number}]"
            if label not in ["FID", "KID", "IS"]:
                data.append([metric_display, f"{value:.4f}", "[0, 1]  ↑"])
            elif label == "IS":
                mean, std = metrics.get("is_mean"), metrics.get("is_std")
                data.append([metric_display, f"{mean:.4f} ± {std:.4f}", "[0, ∞]  ↑"])
            elif label in ["FID", "KID"]:
                data.append([metric_display, f"{value:.4f}", "[0, ∞]  ↓"])
   
    table = Table(data)
    
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ])
    table.setStyle(style)

    justified_style = ParagraphStyle(
    'Justified',
    parent=styles['BodyText'],
    alignment=TA_JUSTIFY
    )

    # Title
    title = Paragraph("Synthetic image quality report", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))    
    
    # Intro text
    intro_paragraph = Paragraph(
        'This report has been generated using the <b><a href="https://github.com/aiformedresearch/Synthetic_Images_Metrics_Toolkit" color="blue">'
        'Synthetic Images Metrics Toolkit</a></b>. This toolkit provides a comprehensive evaluation of the quality of synthetic images using established metrics.<br/><br/>'
        'The evaluation focuses on the following key aspects:',
        styles['BodyText'])
    elements.append(intro_paragraph)

    # Bullet list for key aspects
    bullet_list = ListFlowable([
        ListItem(Paragraph("<b>Fidelity</b>: Evaluates how realistic the synthetic images appear compared to real ones.", styles['BodyText'])),
        ListItem(Paragraph("<b>Diversity</b>: Assesses whether the generated images adequately represent the diversity of the real dataset.", styles['BodyText'])),
        ListItem(Paragraph("<b>Generalization</b>: Determines if the generated images are novel or if they resemble memorized training samples.", styles['BodyText']))
    ], bulletType='bullet')
    elements.append(bullet_list)
    #elements.append(Spacer(1, 12))

    # Dataset comparison section
    closing_paragraph = Paragraph(
        'Dataset comparison:',
        styles['Heading3']
    )
    elements.append(closing_paragraph)

    dataset = dnnlib.util.construct_class_by_name(**args.dataset_kwargs)
    num_real = len(dataset)
    if args.use_pretrained_generator:
        num_syn = args.num_gen
        phrase_gen = f"<b>{num_syn}</b> synthetic images generated by {args.network_path}"
    else:
        dataset_s = dnnlib.util.construct_class_by_name(**args.dataset_synt_kwargs)
        num_syn = len(dataset_s)
        phrase_gen = f"<b>{num_syn}</b> synthetic images from {args.dataset_synt_kwargs['path_data']}"
    
    recap_paragraph  = ListFlowable([
        ListItem(Paragraph(f"<b>{num_real}</b> real images from {args.dataset_kwargs['path_data']}", styles['BodyText'])),
        ListItem(Paragraph(phrase_gen, styles['BodyText'])),
    ], bulletType='bullet')
    elements.append(recap_paragraph)
    elements.append(Spacer(1, 10))

    # Subtitle: Quantitative assessment
    #elements.append(PageBreak())
    subtitle_quant = Paragraph("Quantitative assessment", styles['Heading2'])
    elements.append(subtitle_quant)
    #elements.append(Spacer(1, 12))
    
    # Layout table and image side by side
    plot_path = os.path.join(metric_folder, "figures/metrics_triangle.png")
    table_and_image = Table(
        [[table, Image(metric_utils.get_latest_figure(plot_path), width=250, height=190)]],
        colWidths=[250, 250] 
    )
    elements.append(table_and_image)

    elements.append(Spacer(1, 12))

    additional_text1 = Paragraph(
        "<b>Metrics interpretation:</b> The arrow direction indicates the preferred trend for each metric: ↑ indicates better performance with higher values, and ↓ indicates better performance with lower values.",
        justified_style
    )
    additional_text_2 = Paragraph(
        "<b>Plot interpretation:</b> Metrics are grouped into categories. Each metric has value in [0,1], with 1 representing the optimal value. To provide an overall assessment of the model's performance in each category, the average value of all metrics within that category is displayed.",
        justified_style
    )
    two_texts = Table(
        [[additional_text1, additional_text_2]],
        colWidths=[250, 250] 
    )
    elements.append(two_texts)
    two_texts.setStyle(TableStyle([
    ('VALIGN', (0, 0), (-1, -1), 'TOP')]))
    
    if metrics.get("a_precision_c", None) is not None:
        elements.append(PageBreak())
        subtitle_knn = Paragraph("A closer look: α-precision, β-recall, and authenticity", styles['Heading3'])
        elements.append(subtitle_knn)   

        # Layout with the three images side by side
        prec_rec_path = os.path.join(metric_folder, "figures/alpha_precision_beta_recall_curves_c.png")
        additional_text1 = Paragraph(
            'The scores of <b>α-precision</b> and <b>β-recall</b> are computed from the curves shown, which are derived by computing precision and recall across several values of the α and β parameters. These parameters set thresholds to define what is considered "typical" data, helping to reduce the influence of outliers on the final evaluation. The Δ score represents the deviation from to the ideal curve (optimal performance), while the AUC (which serves as the actual score) corresponds to the area under the curve.',
            justified_style
        )
        pr_images = Table(
            [[Image(metric_utils.get_latest_figure(prec_rec_path), width=350, height=250), 
            additional_text1]],
            colWidths=[350, 150] 
        )
        elements.append(pr_images)
        pr_images.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        # Additional text
        elements.append(Spacer(1, 12))
        auth_path= os.path.join(metric_folder, "figures/authenticity_distribution_c.png")
        batch_size = min(1024, num_real, num_syn)
        num_batches = int(np.ceil(num_syn / batch_size))
        additional_text_2 = Paragraph(
            f"<b>Authenticity</b> measures the fraction of synthetic data not memorized from the training set. To compute this score, batches of {batch_size} synthetic images are compared with batches of {batch_size} real ones (with batch_size = min(1024, #real_imgs, #synth_imgs)), and the final score is calculated as the average across these {num_batches} batches.",
            justified_style
        )
        auth_images = Table(
            [[Image(metric_utils.get_latest_figure(auth_path), width=350, height=250), additional_text_2]],
            colWidths=[350, 150] 
        )
        elements.append(auth_images)
        auth_images.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

    # Subtitle for Qualitative assessment
    elements.append(PageBreak())
    subtitle_qual = Paragraph("Qualitative assessment", styles['Heading2'])
    elements.append(subtitle_qual)
    
    subtitle_vis = Paragraph("Images visualization", styles['Heading3'])
    elements.append(subtitle_vis)

    # Text for qualitative assessment
    qualitative_text = Paragraph(
        "Sample Comparison: A real image vs. a synthetic image",   
        styles['BodyText']
    )
    elements.append(qualitative_text)
    elements.append(Spacer(1, 12))

    # Layout with qualitative visualization
    fidelity_path = os.path.join(metric_folder, "figures/samples_visualization.png")
    fidelity_image = Image(metric_utils.get_latest_figure(fidelity_path), width=400, height=250)
    elements.append(fidelity_image)

    # Text for generalization assessment
    generalization_path = os.path.join(metric_folder, "figures/knn_analysis.png")
    if metric_utils.get_latest_figure(generalization_path) is not None:
        subtitle_knn = Paragraph("k-NN analysis", styles['Heading3'])
        elements.append(subtitle_knn)

        # Layout with qualitative generalization assessment
        generalization_image = Image(metric_utils.get_latest_figure(generalization_path), width=400, height=250)
        elements.append(generalization_image)
        generalization_text = Paragraph(
            "This visualization displays the k-nearest neighbors (k-NN) of real training images, helping to assess whether the model memorizes training data. "
            f"The first column shows the {args.knn_configs['num_real']} real images that have the highest cosine similarity to any synthetic sample. "
            f"Each subsequent column presents the top {args.knn_configs['num_synth']} most similar synthetic images (out of {num_syn} generated samples) for each real image."
        )

        elements.append(generalization_text)

    # Add references section
    if references:
        elements.append(Paragraph("References", styles['Heading2']))
        for ref in references:
            elements.append(Paragraph(ref, styles['Normal']))
            elements.append(Spacer(1, 6))
    
    doc.build(elements, onLaterPages=add_page_number, onFirstPage=add_page_number)
    print(f"Metrics successfully saved to {out_pdf_path}")

def generate_metrics_report(args):
    metric_folder = args.run_dir

    out_file_path = metric_folder+"/report_metrics_toolkit.pdf"
    out_file_path = metric_utils.get_unique_filename(out_file_path)

    if not os.path.isdir(metric_folder):
        print(f"Error: Folder '{metric_folder}' does not exist or is not accessible.")
        exit(1)

    metrics = extract_metrics_from_csv(metric_folder)

    print("Generating the report...")   
    plot_metrics_triangle(metrics, metric_folder)
    save_metrics_to_pdf(args, metrics, metric_folder, out_file_path)








