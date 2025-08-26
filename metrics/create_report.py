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
from reportlab.lib.units import inch
from PIL import Image as PILImage
import torch
import base64
from io import BytesIO

from metrics import metric_utils
import dnnlib

# Define the legend mapping metric keys to human-readable labels
metric_labels = {
    "fid": "FID",
    "kid": "KID",
    "is_mean": "IS",
    "precision": "Precision",
    "density": "Density",
    "a_precision": "α-Precision",
    "recall": "Recall",
    "coverage": "Coverage",
    "b_recall": "β-Recall",
    "authenticity": "Authenticity"
}

metric_references = {
    "FID": "<b>Paper</b>: <a href='https://arxiv.org/abs/1706.08500'>\"GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium\"</a>, Heusel et al. 2017. <br/>  <b>Implementation</b>: Karras et al., https://github.com/NVlabs/stylegan2-ada-pytorch",
    "KID": "<b>Paper</b>: <a href='https://arxiv.org/abs/1801.01401'>\"Demystifying MMD GANs\"</a>, Binkowski et al. 2018 <br/>  <b>Implementation</b>: Karras et al., https://github.com/NVlabs/stylegan2-ada-pytorch",
    "IS": "<b>Paper</b>: <a href='https://arxiv.org/abs/1606.03498'>\"Improved Techniques for Training GANs\"</a>, Salimans et al. 2016 <br/>  <b>Implementation</b>: Karras et al., https://github.com/NVlabs/stylegan2-ada-pytorch",
    "Precision": "<b>Paper</b>: <a href='https://arxiv.org/abs/1904.06991'>\"Improved Precision and Recall Metric for Assessing Generative Models\"</a>, Kynkäänniemi et al. 2019 <br/>  <b>Implementation</b>: Naeem et al., https://github.com/clovaai/generative-evaluation-prdc",
    "Recall": "<b>Paper</b>: <a href='https://arxiv.org/abs/1904.06991'>\"Improved Precision and Recall Metric for Assessing Generative Models\"</a>, Kynkäänniemi et al. 2019 <br/>  <b>Implementation</b>: Naeem et al., https://github.com/clovaai/generative-evaluation-prdc",
    "Density": "<b>Paper</b>: <a href='https://proceedings.mlr.press/v119/naeem20a/naeem20a.pdf'>\"Reliable Fidelity and Diversity Metrics for Generative Models\"</a>, Naeem et al., 2020 <br/>  <b>Implementation</b>: Naeem et al., https://github.com/clovaai/generative-evaluation-prdc",
    "Coverage": "<b>Paper</b>: <a href='https://proceedings.mlr.press/v119/naeem20a/naeem20a.pdf'>\"Reliable Fidelity and Diversity Metrics for Generative Models\"</a>, Naeem et al., 2020 <br/>  <b>Implementation</b>: Naeem et al., https://github.com/clovaai/generative-evaluation-prdc",
    "α-Precision": "<b>Paper</b>: <a href='https://proceedings.mlr.press/v162/alaa22a/alaa22a.pdf'>\"How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models\"</a>, Alaa et al., 2022 <br/>  <b>Implementation</b>: Alaa et al., https://github.com/vanderschaarlab/evaluating-generative-models",
    "β-Recall": "<b>Paper</b>: <a href='https://proceedings.mlr.press/v162/alaa22a/alaa22a.pdf'>\"How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models\"</a>, Alaa et al., 2022 <br/>  <b>Implementation</b>: Alaa et al., https://github.com/vanderschaarlab/evaluating-generative-models",
    "Authenticity": "<b>Paper</b>: <a href='https://proceedings.mlr.press/v162/alaa22a/alaa22a.pdf'>\"How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models\"</a>, Alaa et al., 2022 <br/>  <b>Implementation</b>: Alaa et al., https://github.com/vanderschaarlab/evaluating-generative-models",
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

    known_flags = {"fid", "kid", "is_", "prdc", "pr_auth"}
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
    fidelity_metrics = {k: metrics.get(k, None) for k in ["precision", "density", "a_precision"]}
    fidelity_metrics = {k: np.clip(v, 0, 1) for k, v in fidelity_metrics.items() if v is not None}
    fidelity_mean = np.mean(list(fidelity_metrics.values()))

    diversity_metrics = {k: metrics.get(k, None) for k in ["recall", "coverage", "b_recall"]}
    diversity_metrics = {k: np.clip(v, 0, 1) for k, v in diversity_metrics.items() if v is not None}
    diversity_mean = np.mean(list(diversity_metrics.values()))

    generalization_metrics = {k: np.clip(metrics.get(k, None), 0, 1) for k in ["authenticity"] if metrics.get(k) is not None}
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
        ax.legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(-0.2, 1.1), fontsize=12.5)
        
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(metric_folder, "figures/metrics_triangle.png")
    plt.savefig(metric_utils.get_unique_filename(plot_path))
    plt.close()
   
def get_image_with_scaled_dimensions(path, max_width=None, max_height=None):
    img = PILImage.open(path)
    orig_width_px, orig_height_px = img.size

    dpi = img.info.get("dpi", (72, 72))[0]  # Default to 72 if DPI info missing

    # Convert pixel size to points
    orig_width_pt = orig_width_px * 72 / dpi
    orig_height_pt = orig_height_px * 72 / dpi

    # If both max dimensions are set, scale proportionally to fit within both
    if max_width and max_height:
        ratio = min(max_width / orig_width_pt, max_height / orig_height_pt)
    elif max_width:
        ratio = max_width / orig_width_pt
    elif max_height:
        ratio = max_height / orig_height_pt
    else:
        ratio = 1.0  # no resizing

    new_width = min(int(orig_width_pt * ratio), 456.0)
    new_height = min(int(orig_height_pt * ratio), 535.0)

    return Image(path, width=new_width, height=new_height)

def save_metrics_to_pdf(args, metrics, metric_folder, out_pdf_path):
    doc = SimpleDocTemplate(out_pdf_path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    # Create table data: header row + rows for each metric present in both `metrics` and `metric_labels`
    ref_mapping = {}
    references = []
    ref_counter = 1
    data = [["Metric [ref]", "Value", "Range", "Trend"]]
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
                data.append([metric_display, f"{value:.4f}", "[0, 1]", "↑"])
            elif label == "IS":
                mean, std = metrics.get("is_mean"), metrics.get("is_std")
                data.append([metric_display, f"{mean:.4f} ± {std:.4f}", "[0, ∞]", "↑"])
            elif label in ["FID", "KID"]:
                data.append([metric_display, f"{value:.4f}", "[0, ∞]", "↓"])
   
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
    
    # Logo
    EMBEDDED_LOGO_BASE64 = "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" 

    logo_data = base64.b64decode(EMBEDDED_LOGO_BASE64)
    logo_buffer = BytesIO(logo_data)

    # Create image
    logo_image = Image(logo_buffer, width=200, height=200)
    elements.append(logo_image)

    # --------------------------------------- Introduction ------------------------------------------------

    # Intro text
    elements.append(Spacer(1, 50))
    intro_paragraph = Paragraph(
        'This report was generated using the <b><a href="https://github.com/aiformedresearch/Synthetic_Images_Metrics_Toolkit" color="blue">'
        'Synthetic Images Metrics (SIM) Toolkit</a></b> (v3.2), created at the University of Bologna. This toolkit provides a comprehensive evaluation of synthetic image quality using established quantitative and qualitative metrics.<br/><br/>'
        'The SIM Toolkit evaluates synthetic images based on the following key dimensions:',
        justified_style)
    elements.append(intro_paragraph)

    # Bullet list for key dimensions
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

    # --------------------------------------- Quantitative assessment ------------------------------------------------

    # Subtitle: Quantitative assessment
    elements.append(PageBreak())
    subtitle_quant = Paragraph("Quantitative assessment", styles['Heading2'])
    elements.append(subtitle_quant)
    #elements.append(Spacer(1, 12))

    intro_paragraph = Paragraph(
        "<b>Metrics interpretation:</b><br/> The ideal trend for each metric is indicated by arrow direction:",
        justified_style
    )
    list_items = [
        ListItem(Paragraph(f"↑ indicates better performance with higher values;", justified_style), leftIndent=0),
        ListItem(Paragraph(f"↓ indicates better performance with lower values.", justified_style), leftIndent=0),
    ]

    bullet_list = ListFlowable(
        list_items,
        bulletType='bullet',
        start=None,
        leftIndent=10
    )

    metrics_table = Table(
        [[table, [intro_paragraph, bullet_list]]],
        colWidths=[300, 150]
    )

    elements.append(metrics_table)
    metrics_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    caption = Paragraph(f"<font color='gray'><i>From: {os.path.join(args.run_dir, 'metrics.csv')}</i></font>", styles['BodyText'])
    elements.append(caption)

    # -------------------------------------------

    elements.append(Spacer(1, 20))
    triangle_path = metric_utils.get_latest_figure(os.path.join(metric_folder, "figures/metrics_triangle.png"))
    additional_text1 = Paragraph(
        "<b>Plot interpretation:</b><br/> Metrics are grouped into categories. Each metric has value in [0,1], with 1 representing the optimal value. To provide an overall assessment of the model's performance in each category, the average value of all metrics within that category is displayed.",
        justified_style
    )
    traingle = Table(
        [[get_image_with_scaled_dimensions(triangle_path, max_width=350), 
        additional_text1]],
        colWidths=[300, 150] 
    )
    elements.append(traingle)
    traingle.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    triangle_caption = Paragraph(f"<font color='gray'><i>From: {triangle_path}</i></font>", styles['BodyText'])
    elements.append(triangle_caption)

    # --------------------------------------- Qualitative assessment ------------------------------------------------

    # Subtitle for Qualitative assessment
    elements.append(PageBreak())
    
    subtitle_vis = Paragraph("Visualization of real and synthetic samples", styles['Heading2'])
    elements.append(subtitle_vis)

    intro_text = Paragraph(
        f"To qualitatively assess <b>fidelity</b> and <b>diversity</b>, compare the real and synthetic image grids below. "
        "For valid metric computation, ensure matching shape, value range, and data type.",   
        justified_style
    )
    elements.append(intro_text)

    # Real images visualization
    dataset_real = dnnlib.util.construct_class_by_name(**args.dataset_kwargs)
    if args.data_type.lower()=='2d':
        n, ch, w_r, h_r = dataset_real._raw_shape
        real_text = Paragraph(
            f"<b>Real samples</b>: value range ∈ [{dataset_real._min}, {dataset_real._max}] | size: {w_r} × {h_r} | dtype: {dataset_real._dtype}",   
            styles['BodyText']
        )
    elif args.data_type.lower()=='3d':
        n, ch, w_r, h_r, z_r = dataset_real._raw_shape
        real_text = Paragraph(
            f"<b>Real samples</b>: value range ∈ [{dataset_real._min}, {dataset_real._max}] | size: {w_r} × {h_r} × {z_r} | dtype: {dataset_real._dtype}",   
            styles['BodyText']
        )
    elements.append(real_text)
    elements.append(Spacer(1, 5))

    # Layout with qualitative visualization
    real_path = metric_utils.get_latest_figure(os.path.join(metric_folder, "figures/samples_real.png"))
    real_grid = get_image_with_scaled_dimensions(real_path, max_height=240)
    elements.append(real_grid)
    real_caption = Paragraph(f"<font color='gray'><i>Full resolution file: {real_path}</i></font>", styles['BodyText'])
    elements.append(real_caption)

    # ----------------------------------

    if  args.use_pretrained_generator:
        import copy
        n_to_generate = 10
        device = torch.device(f'cuda' if torch.cuda.is_available() and args.num_gpus > 0 else 'cpu')
        G = copy.deepcopy(args.G).eval().requires_grad_(False).to(device)
        z = torch.randn([n_to_generate, *(G.z_dim if isinstance(G.z_dim, (list, tuple)) else [G.z_dim])], device=device)
        # define c as a vector with batch_size elements sampled from 0 and 1:
        half_n = n_to_generate // 2
        c = [torch.tensor([1, 0]) for _ in range(half_n)] + [torch.tensor([0, 1]) for _ in range(half_n)]       
        if n_to_generate % 2 != 0:
            c.append(torch.tensor([1, 0]) if torch.randint(0, 2, (1,)).item() == 1 else torch.tensor([0, 1]))
        c = torch.from_numpy(np.stack(c)).pin_memory().to(device)
        if dataset._use_labels:
            batch = args.run_generator(z, c, args).to(device)
        else:
            batch = args.run_generator(z, args).to(device)
        if args.data_type.lower()=='2d':
            n, ch, w_s, h_s = batch.shape
            synt_text = Paragraph(
                f"<b>Synthetic samples</b>: value range ∈ [{batch.min()}, {batch.max()}] | size: {w_s} × {h_s} | dtype: {batch.dtype}",  
                styles['BodyText'])
        elif args.data_type.lower()=='3d':
            n, ch, w_s, h_s, z_s= batch.shape
            synt_text = Paragraph(
                f"<b>Synthetic samples</b>: value range ∈ [{batch.min()}, {batch.max()}] | size: {w_s} × {h_s} × {z_s} | dtype: {batch.dtype}",  
                styles['BodyText']
                )
    else:
        dataset_synt = dnnlib.util.construct_class_by_name(**args.dataset_synt_kwargs)
        if args.data_type.lower()=='2d':
            n, ch, w_s, h_s = dataset_synt._raw_shape
            synt_text = Paragraph(
                f"<b>Synthetic samples</b>: value range ∈ [{dataset_synt._min}, {dataset_synt._max}] | size: {w_s} × {h_s} | dtype: {dataset_synt._dtype}",  
                styles['BodyText']
            )
        elif args.data_type.lower()=='3d':
            n, ch, w_s, h_s, z_s= dataset_synt._raw_shape
            synt_text = Paragraph(
                f"<b>Synthetic samples</b>: value range ∈ [{dataset_synt._min}, {dataset_synt._max}] | size: {w_s} × {h_s} × {z_s} | dtype: {dataset_synt._dtype}",  
                styles['BodyText']
                )
            
    elements.append(synt_text)
    elements.append(Spacer(1, 5))
    
    # Layout with qualitative visualization
    synt_path = metric_utils.get_latest_figure(os.path.join(metric_folder, "figures/samples_synt.png"))
    synt_grid = get_image_with_scaled_dimensions(synt_path, max_height=240)
    elements.append(synt_grid)
    synt_caption = Paragraph(f"<font color='gray'><i>Full resolution file: {synt_path}</i></font>", styles['BodyText'])
    elements.append(synt_caption)


    # --------------------------------------- Metrics interpretation ------------------------------------------------
 
    elements.append(PageBreak())
    subtitle_2d_emb = Paragraph("Embedding space visualization: PCA and t-SNE", styles['Heading2'])
    elements.append(subtitle_2d_emb)
    elements.append(Spacer(1, 12))

    text_2d_emb = Paragraph("All metrics in this report are computed using high-dimensional embeddings extracted from pretrained networks. To support qualitative interpretation, we include 2D visualizations using dimensionality reduction techniques:", justified_style)
    elements.append(text_2d_emb)

    list_items = [
        ListItem(Paragraph("<b>Principal Component Analysis (PCA)</b>, a linear projection that preserves the directions of greatest variance. It helps reveal global structure but may miss subtler variations;", justified_style), leftIndent=0),
        ListItem(Paragraph("<b>t-distributed Stochastic Neighbor Embedding (t-SNE)</b>, a non-linear, probabilistic method that highlights local relationships and clustering. However, it does not preserve global distances.", justified_style), leftIndent=0),
    ]

    bullet_list = ListFlowable(
        list_items,
        bulletType='bullet',
        start=None,
        leftIndent=10
    )

    elements.append(bullet_list)

    text_2d_emb2 = Paragraph("Together, these visualizations offer complementary insights into the structure of the embedding space, supporting qualitative interpretation of metric results such as fidelity, diversity, and generalization.", justified_style)
    elements.append(text_2d_emb2)

    # ----------------- IS ---------------------
    is_tsne_path = os.path.join(metric_folder, "figures/tsne_is.png")
    is_pca_path = os.path.join(metric_folder, "figures/pca_is.png")
    is_path = os.path.join(metric_folder, "figures/is_probs.png")
    if os.path.exists(is_tsne_path):
        elements.append(Spacer(1, 24))
        intro_is = Paragraph("Here is an example of PCA and t-SNE visualization of the synthetic images, used to compute the inception score:",
                             justified_style)
        elements.append(intro_is)

        is_tsne = Table(
            [[get_image_with_scaled_dimensions(metric_utils.get_latest_figure(is_pca_path), max_width=225), 
            get_image_with_scaled_dimensions(metric_utils.get_latest_figure(is_tsne_path), max_width=225)]],
            colWidths=[225, 225] 
        )
        elements.append(is_tsne) 

        caption = Table(
            [[Paragraph(f"<font color='gray'><i>From: {is_pca_path}</i></font>", styles['BodyText']), 
            Paragraph(f"<font color='gray'><i>From: {is_tsne_path}</i></font>", styles['BodyText'])]],
            colWidths=[225, 225] 
        )
        elements.append(caption)

        elements.append(PageBreak())
        subtitle_is = Paragraph("Inception Score (IS)", styles['Heading3'])
        elements.append(subtitle_is)   

        if args.data_type.lower() == '2d':
            embedder = "Inception Net"
            link = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/"
            dataset_embedder = "ImageNet"
        elif args.data_type.lower() == '3d':
            embedder = "3D-ResNet"
            link = "https://github.com/Tencent/MedicalNet"
            dataset_embedder = "23 medical imaging datasets from different modalities (MRI and CT) and distinctive scan regions (e.g. brain, heart, prostate, spleen, etc.)"
        text_is = Paragraph(
            f'The IS measures how realistic and diverse synthetic images are by analyzing the class distributions output by a pretrained <a href="{link}" color="blue">{embedder}</a>. ' 
            'The idea is that realistic images should be classified with high confidence (low entropy), and diverse images should be classified across many different classes. '
            'The PCA and t-SNE visualizations shown in the previous page help us interpret the IS score by showing how the predicted class distributions (softmax outputs) are spread in a lower-dimensional space.<br/><br/>'
            f'<b>IS score</b>: {metrics["is_mean"]} ± {metrics["is_std"]}<br/><br/>'
            'To help interpret the score and assess its validity, we visualize the following distributions:',
            justified_style
        )
        elements.append(text_is)   

        list_items = [
            ListItem(Paragraph('<b>Max softmax probabilities</b>: High values indicate confident predictions — a sign of realism. Low values may suggest that the images are blurry, unstructured, or otherwise difficult to classify.', justified_style), leftIndent=0),
            ListItem(Paragraph('<b>Top-1 predicted class</b>: A narrow distribution (i.e., only a few classes predicted) reflects low diversity, while a broader spread suggests that synthetic images cover a wider variety of visual categories.', justified_style), leftIndent=0),
        ]

        bullet_list = ListFlowable(
            list_items,
            bulletType='bullet',
            start=None,
            leftIndent=10
        )

        elements.append(bullet_list)
        elements.append(Spacer(1, 12))

        is_image = get_image_with_scaled_dimensions(is_path, max_width=450)
        elements.append(is_image)
        is_caption = Paragraph(f"<font color='gray'><i>From: {is_path}</i></font>", styles['BodyText'])
        elements.append(is_caption)

        final_text = Paragraph(f'<b>! Domain-specific considerations:</b> <br/>The IS relies on a classifier trained on {dataset_embedder}. '
                    'If your synthetic images belong to a different domain, the classifier’s outputs may become unreliable, and <b>IS is no longer a valid measure of quality</b>. '
                    'In such cases, the model may assign arbitrary or semantically meaningless labels, invalidating the score. <br/>', justified_style)
        elements.append(final_text)
        
    # ----------------- FID ---------------------
    fid_tsne_path = os.path.join(metric_folder, "figures/tsne_fid.png")
    fid_pca_path = os.path.join(metric_folder, "figures/pca_fid.png")
    if os.path.exists(fid_tsne_path):
        elements.append(PageBreak())
        subtitle_fid = Paragraph("Fréchet Inception Distance (FID)", styles['Heading3'])
        elements.append(subtitle_fid)   

        if args.data_type.lower() == '2d':
            embedder = "Inception Net"
            link = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/"
        elif args.data_type.lower() == '3d':
            embedder = "3D-ResNet"
            link = "https://github.com/Tencent/MedicalNet"
        text_fid = Paragraph(
            f'FID measures the similarity between real and synthetic images by comparing the distributions of their feature embeddings, extracted from a pretrained <a href="{link}" color="blue">{embedder}</a>. ' 
            'These 2048-dimensional embeddings are modeled as multivariate Gaussians, and the FID score is computed as the Fréchet distance between them.<br/>'
            'To support qualitative interpretation, the embeddings are visualized in a reduced two-dimensional space using PCA and t-SNE. '
            'Being PCA a linear transformation, it allows for the projection of the multidimensional Gaussian distributions into 2D, which is shown in the corresponding plot. '
            'However, this projection captures only a portion of the total variance, which is indicated in the image title.<br/>'
            'Lower FID values reflect a smaller distance between the two distributions, indicating greater similarity in terms of image <b>fidelity and diversity</b>.<br/><br/>'
            f'<b>FID score</b>: {metrics["fid"]}',
            justified_style
        )
        elements.append(text_fid)   

        fid_tsne = Table(
            [[get_image_with_scaled_dimensions(metric_utils.get_latest_figure(fid_pca_path), max_width=225), 
            get_image_with_scaled_dimensions(metric_utils.get_latest_figure(fid_tsne_path), max_width=225)]],
            colWidths=[225, 225] 
        )
        elements.append(fid_tsne) 

        caption = Table(
            [[Paragraph(f"<font color='gray'><i>From: {fid_pca_path}</i></font>", styles['BodyText']), 
            Paragraph(f"<font color='gray'><i>From: {fid_tsne_path}</i></font>", styles['BodyText'])]],
            colWidths=[225, 225] 
        )
        elements.append(caption)

    # ----------------- KID ---------------------
    kid_tsne_path = os.path.join(metric_folder, "figures/tsne_kid.png")
    kid_pca_path = os.path.join(metric_folder, "figures/pca_kid.png")
    if os.path.exists(kid_tsne_path):
        subtitle_kid = Paragraph("Kernel Inception Distance (KID)", styles['Heading3'])
        elements.append(subtitle_kid)   

        if args.data_type.lower() == '2d':
            embedder = "Inception Net"
            link = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/"
        elif args.data_type.lower() == '3d':
            embedder = "3D-ResNet"
            link = "https://github.com/Tencent/MedicalNet"

        text_kid = Paragraph(
            f'KID evaluates the similarity between real and synthetic images using the embeddings extracted from the same pretrained model used for FID: <a href="{link}" color="blue">{embedder}</a>. ' 
            'Unlike FID, KID does not assume a Gaussian distribution. '
            'Instead, it computes the Maximum Mean Discrepancy (MMD) with a polynomial kernel to quantify the difference between the feature distributions of real and generated images. '
            'This makes KID an unbiased estimator that is particularly well-suited for small sample sizes.<br/>'
            'Like FID, lower KID values indicate closer alignment between the real and synthetic data distributions, capturing both <b>fidelity and diversity</b> in a single score.<br/><br/>'
            f'<b>KID score</b>: {metrics["kid"]}',
            justified_style
        )
        elements.append(text_kid)   

        if not os.path.exists(metric_utils.get_latest_figure(fid_tsne_path)):
            kid_tsne = Table(
                [[get_image_with_scaled_dimensions(metric_utils.get_latest_figure(kid_pca_path), max_width=225), 
                get_image_with_scaled_dimensions(metric_utils.get_latest_figure(kid_tsne_path), max_width=225)]],
                colWidths=[225, 225] 
            )
            elements.append(kid_tsne) 

            caption = Table(
                [[Paragraph(f"<font color='gray'><i>From: {kid_pca_path}</i></font>", styles['BodyText']), 
                Paragraph(f"<font color='gray'><i>From: {kid_tsne_path}</i></font>", styles['BodyText'])]],
                colWidths=[225, 225] 
            )
            elements.append(caption)

    # ----------------- PRDC ---------------------
    prdc_tsne_path = os.path.join(metric_folder, "figures/tsne_prdc.png")
    prdc_pca_path = os.path.join(metric_folder, "figures/pca_prdc.png")
    if os.path.exists(prdc_tsne_path):
        elements.append(PageBreak())
        subtitle_prdc = Paragraph("Precision, recall, density, and coverage", styles['Heading3'])
        elements.append(subtitle_prdc)   

        if args.data_type.lower() == '2d':
            embedder = "VGG-16 (from Tensorflow)"
            link = "https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16"
        elif args.data_type.lower() == '3d':
            embedder = "3D-ResNet"
            link = "https://github.com/Tencent/MedicalNet"

        text_prdc = Paragraph(
            f'Precision, Recall (P&amp;R), density and coverage (D&amp;C) are complementary metrics used to assess the fidelity and diversity of synthetic images. These metrics compare feature embeddings of real and synthetic images extracted via a pretrained <a href="{link}" color="blue">{embedder}</a>, which maps each image into a 4096-dimensional space. ',
            justified_style
        )
        elements.append(text_prdc)   

        if args.data_type.lower()=='2d':
            if args.padding:
                if (w_r<224 or h_r<224):
                    method = "zero-padding. Alternatively, you can set to use the PIL.BICUBIC resizer through the configuration file."
                else:
                    method = "PIL.BICUBIC resizer."
            elif not args.padding:
                if (w_r<224 or h_r<224):
                    method = "PIL.BICUBIC resizer. Alternatively, you can set to perform zero-padding through the configuration file."
                else:
                    method = "PIL.BICUBIC resizer."
            text_prdc = Paragraph(
            f'Images are resized to 224 x 224 to meet the network\'s input requirements using {method} ', justified_style
            )
            elements.append(text_prdc)  

        text_prdc = Paragraph(
            f'The support of each distribution is estimated using a k-nearest neighbors (k-NN) approach, with k = {args.nhood_size["prdc"]}. '
            'For each sample, a hypersphere is defined by the distance to its k-th nearest neighbor within the same set, approximating the local manifold of the distribution. '
            'Note that this method is sensitive to outliers, since an outlier can inflate the estimated support.<br/><br/>'
            'Given the estimated supports of real and synthetic distributions, P&amp;R are computed:<br/>'
            '- <b>Precision</b>: <i>what fraction of synthetic images look realistic?</i> Precision measures the proportion of synthetic samples that fall within the support of the real data distribution, reflecting <b>fidelity</b>.<br/>'
            '- <b>Recall</b>: <i>how much variety from the real distribution do synthetic images capture?</i> Recall quantifies the proportion of real samples that are covered by the synthetic distribution, indicating <b>diversity</b>.<br/><br/>'
            'To mitigate the sensitivity of P&amp;R to outliers, D&amp;C were introduced as more robust alternatives:<br/>'
            '- <b>Density</b>: <i>how well do synthetic images populate realistic regions?</i> Density counts how many real-sample neighborhoods contain synthetic images, representing <b>fidelity</b>.<br/>'
            '- <b>Coverage</b>: <i>to what extent do synthetic images represent the full variety of real image types?</i> Coverage evaluates how much of the real distribution is covered by at least one synthetic sample\'s neighborhood, serving as a proxy for <b>diversity</b>.<br/><br/>'
            'All four metrics range from 0 to 1, with higher values indicating a better match between the real and synthetic distributions.<br/><br/>'
            f'Current scores:<br/>'
            f'- <b>Precision</b>: {metrics["precision"]}<br/>'
            f'- <b>Recall</b>: {metrics["recall"]}<br/>'
            f'- <b>Density</b>: {metrics["density"]}<br/>'
            f'- <b>Coverage</b>: {metrics["coverage"]}<br/>',
            justified_style
        )
        elements.append(text_prdc)   

        prdc_tsne = Table(
            [[get_image_with_scaled_dimensions(metric_utils.get_latest_figure(prdc_pca_path), max_width=225), 
            get_image_with_scaled_dimensions(metric_utils.get_latest_figure(prdc_tsne_path), max_width=225)]],
            colWidths=[225, 225] 
        )
        elements.append(prdc_tsne) 

        caption = Table(
            [[Paragraph(f"<font color='gray'><i>From: {prdc_pca_path}</i></font>", styles['BodyText']), 
            Paragraph(f"<font color='gray'><i>From: {prdc_tsne_path}</i></font>", styles['BodyText'])]],
            colWidths=[225, 225] 
        )
        elements.append(caption)

    # -------------------------- alpha-precision, beta-recall, authenticity -------------------------

    pr_auth_tsne_path = os.path.join(metric_folder, "figures/tsne_pr_auth.png")
    pr_auth_pca_path = os.path.join(metric_folder, "figures/pca_pr_auth.png")
    pr_auth_OC_tsne_path = os.path.join(metric_folder, "figures/tsne_pr_auth_OC.png")
    pr_auth_OC_pca_path = os.path.join(metric_folder, "figures/pca_pr_auth_OC.png")
    if os.path.exists(pr_auth_tsne_path):
        elements.append(PageBreak())
        subtitle_pr_auth = Paragraph("α-precision, β-recall, and authenticity", styles['Heading3'])
        elements.append(subtitle_pr_auth)   

        if args.data_type.lower() == '2d':
            embedder = "VGG-16 (from Tensorflow)"
            link = "https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16"
        elif args.data_type.lower() == '3d':
            embedder = "3D-ResNet"
            link = "https://github.com/Tencent/MedicalNet"
        
        text_pr_auth = Paragraph(
            f'α-precision, β-recall, and authenticity are computed using embeddings extracted from a pretrained <a href="{link}" color="blue">{embedder}</a>, which maps each image into a 2048-dimensional feature space. '
        )
        elements.append(text_pr_auth)  

        if args.data_type.lower()=='2d':
            if args.padding:
                if (w_r<299 or h_r<299):
                    method = "zero-padding. Alternatively, you can set to use the PIL.BICUBIC resizer through the configuration file."
                else:
                    method = "PIL.BICUBIC resizer."
            elif not args.padding:
                if (w_r<299 or h_r<299):
                    method = "PIL.BICUBIC resizer. Alternatively, you can set to perform zero-padding through the configuration file."
                else:
                    method = "PIL.BICUBIC resizer."
            text_pr_auth = Paragraph(
            f'Given that the pretrained network requires input of size 299 x 299, images are adapted via {method} '
            )
            elements.append(text_pr_auth)     
        text_pr_auth = Paragraph(    
            'For a qualitative assessment, we visualize the distribution of real and synthetic embeddings in a reduced 2-dimensional space using PCA and t-SNE:',
            justified_style
        )
        elements.append(text_pr_auth)   

        pr_auth_tsne = Table(
            [[get_image_with_scaled_dimensions(metric_utils.get_latest_figure(pr_auth_pca_path), max_width=225), 
            get_image_with_scaled_dimensions(metric_utils.get_latest_figure(pr_auth_tsne_path), max_width=225)]],
            colWidths=[225, 225] 
        )
        elements.append(pr_auth_tsne) 

        caption = Table(
            [[Paragraph(f"<font color='gray'><i>From: {pr_auth_pca_path}</i></font>", styles['BodyText']), 
            Paragraph(f"<font color='gray'><i>From: {pr_auth_tsne_path}</i></font>", styles['BodyText'])]],
            colWidths=[225, 225] 
        )
        elements.append(caption)

        text_pr_auth = Paragraph(
            f'The 2048-dimensional embeddings are then projected onto a 32-dimensional hypersphere using a <b>One-Class (OC)</b> classifier. '
            'In this transformed space, samples considered "typical" lie near the center of the hypersphere, while outliers are located closer to the boundary. '
            f'Below, we visualize the OC-transformed embeddings using PCA and the t-SNE:',
            justified_style
        )
        elements.append(text_pr_auth)   

        elements.append(Spacer(1, 5))

        pr_auth_tsne = Table(
            [[get_image_with_scaled_dimensions(metric_utils.get_latest_figure(pr_auth_OC_pca_path), max_width=225), 
            get_image_with_scaled_dimensions(metric_utils.get_latest_figure(pr_auth_OC_tsne_path), max_width=225)]],
            colWidths=[225, 225] 
        )
        elements.append(pr_auth_tsne) 

        caption = Table(
            [[Paragraph(f"<font color='gray'><i>From: {pr_auth_OC_pca_path}</i></font>", styles['BodyText']), 
            Paragraph(f"<font color='gray'><i>From: {pr_auth_OC_tsne_path}</i></font>", styles['BodyText'])]],
            colWidths=[225, 225] 
        )
        elements.append(caption)
        text_pr_auth = Paragraph(
            f'In this space, α-precision and β-recall are computed by varying the values of α and β, which act as thresholds to define concentric regions around the data center. '
            'These thresholds determine which samples are considered <i>typical</i> (i.e., within the hypersphere of radius α or β).<br/><br/>',
            justified_style
        )
        elements.append(text_pr_auth)  

        #elements.append(PageBreak())
        text_pr_auth = Paragraph(
            'By varying these thresholds from 0 (no samples considered <i>typical</i>) to 1 (all samples considered <i>typical</i>), we generate two curves:<br/>'
            f'- <b>α-precision curve</b>: at each α level, we compute the proportion of synthetic samples that fall within the α-quantile radius of the real data distribution — reflecting how well synthetic data matches the distribution of the <i>typical</i> real data  (i.e., <b>fidelity</b>);<br/>'
            f'- <b>β-recall curve</b>: at each β level, we compute the proportion of real samples that are (i) closer to a synthetic neighbor than a real one, and (ii) whose closest synthetic neighbor lies within the β-quantile region of the synthetic distribution — reflecting how well synthetic data covers the typical real samples (i.e., <b>diversity</b>). <br/><br/>'
            f'The final scores correspond to the area under the curve (AUC):<br/>'
            f'- <b>α-precision</b>: {metrics["a_precision"]}<br/>'
            f'- <b>β-recall</b>: {metrics["b_recall"]}<br/>',
            justified_style
        )
        elements.append(text_pr_auth)   

        # ------------- alpha-precision and beta-recall curves -------------
        prec_rec_path = os.path.join(metric_folder, "figures/alpha_precision_beta_recall_curves.png")
        additional_text1 = Paragraph(
            'Note: in the original implementation, the final scores were derived by computing the deviation from the ideal curve (optimal performance), indicated by the Δ score. '
            'However, this approach is sensitive to both underperformance and overperformance (i.e., being too "precise" or too "spread out"). '
            'For this reason, we report ∆ in the plots for reference, but use the AUC-based scores as the primary metrics. ',
            justified_style
        )
        pr_images = Table(
            [[get_image_with_scaled_dimensions(metric_utils.get_latest_figure(prec_rec_path), max_width=350), 
            additional_text1]],
            colWidths=[350, 150] 
        )
        elements.append(pr_images)
        pr_images.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        caption = Paragraph(f"<font color='gray'><i>From: {prec_rec_path}</i></font>", styles['BodyText'])
        elements.append(caption)

        # ---------- authenticity --------------
        
        elements.append(PageBreak())
        auth_path= os.path.join(metric_folder, "figures/authenticity_distribution.png")
        batch_size = min(1024, num_real, num_syn)
        num_batches = int(np.ceil(num_syn / batch_size))
        text_pr_auth = Paragraph(
            f"<b>Authenticity</b> measures the fraction of synthetic data not memorized from the training set, quantitatively assessyng <b>generalization</b>. "
            "A synthetic image is considered authentic if its closest synthetic neighbor is farther awar than its closest real neighbor. "
            f"To compute this score, batches of {batch_size} synthetic images are compared with batches of {batch_size} real ones (with batch_size = min(1024, #real_imgs, #synth_imgs)), and the final score is calculated as the average across these {num_batches} batches.<br/><br/>"
            f'Score: <br/>'
            f'<b>Authenticity</b>: {metrics["authenticity"]}<br/>',
            justified_style
        )
        elements.append(text_pr_auth)   

        # Additional text
        elements.append(Spacer(1, 12))

        auth_images = get_image_with_scaled_dimensions(metric_utils.get_latest_figure(auth_path), max_width=350)
        elements.append(auth_images)
        caption = Paragraph(f'<font color="gray"><i>From: {auth_path}</i></font>', styles['BodyText'])
        elements.append(caption)

    # ---------------------------------------
    elements.append(PageBreak())

    # Text for generalization assessment
    generalization_path = os.path.join(metric_folder, "figures/knn_analysis.png")
    if os.path.exists(generalization_path):
        generalization_path = metric_utils.get_latest_figure(generalization_path)
        subtitle_knn = Paragraph("k-NN analysis", styles['Heading3'])
        elements.append(subtitle_knn)

        generalization_text = Paragraph(
            "The k-nearest neighbors (k-NN) visualization provides a qualitative assessment of <b>generalization</b>, aiming to identify potential memorization of training data by the generative model. <br/>"
            "In this analysis:<br/>"
            f"- The first column shows the {args.knn_configs['num_real']} real images that have the highest cosine similarity to any synthetic sample. <br/>"
            f"- Each subsequent column presents the top {args.knn_configs['num_synth']} most similar synthetic images (from {num_syn} generated samples) for each real image. ",
            justified_style
        )
        elements.append(generalization_text)

        if os.path.exists(pr_auth_tsne_path):
            generalization_text2 = Paragraph(
                "Similarity is computed using the same embedding space used for the authenticity score, enabling to visually inspect which synthetic samples are closest to real data points and could have contributed to a lower authenticity score. "
                "If the retrieved synthetic samples in a row are very similar to one another, this may indicate limited <b>diversity</b> in the generated set. ",
                justified_style
            )
            elements.append(generalization_text2)
        elements.append(Spacer(1, 12))

        # Layout with qualitative generalization assessment
        generalization_image = get_image_with_scaled_dimensions(generalization_path, max_width=450)
        elements.append(generalization_image)
        knn_caption = Paragraph(f"<font color='gray'><i>From: {generalization_path}</i></font>", styles['BodyText'])
        elements.append(knn_caption)

        if args.use_pretrained_generator:
            text_knn = "The number below each real image indicates its index in the original dataset, starting from 0."
        else:
            text_knn = "The number below each image indicates its index in the original dataset, starting from 0. This allows users to locate and view the corresponding image at full resolution, enabling a more detailed comparison between real and synthetic samples."

        idx_text = Paragraph(text_knn, justified_style)
        elements.append(idx_text)

    # --------------------------------------- References ------------------------------------------------

    # Add references section
    if references:
        elements.append(PageBreak())
        elements.append(Paragraph("References", styles['Heading2']))
        for ref in references:
            elements.append(Paragraph(ref, styles['Normal']))
            elements.append(Spacer(1, 6))
    
    doc.build(elements, onLaterPages=add_page_number, onFirstPage=add_page_number)
    print(f"Report successfully saved to {out_pdf_path}")

def generate_metrics_report(args):
    metric_folder = args.run_dir

    out_file_path = metric_folder+"/report_sim_toolkit.pdf"
    out_file_path = metric_utils.get_unique_filename(out_file_path)

    if not os.path.isdir(metric_folder):
        print(f"Error: Folder '{metric_folder}' does not exist or is not accessible.")
        exit(1)

    metrics = extract_metrics_from_csv(metric_folder)

    print("Generating the report...")   
    plot_metrics_triangle(metrics, metric_folder)
    save_metrics_to_pdf(args, metrics, metric_folder, out_file_path)








