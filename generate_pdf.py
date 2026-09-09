"""
UAQE 16-Phase Complete Technical Learning Guide - PDF Generation Script
Built with ReportLab for precise multi-page document layout and exact 8-page budgeting.
"""

import os
import re
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.pdfgen import canvas

# Geometry & Dimensions
PAGE_WIDTH, PAGE_HEIGHT = A4  # 595.27 x 841.89 pt
MARGIN_X = 28.0
MARGIN_TOP = 32.0
MARGIN_BOTTOM = 28.0
USABLE_WIDTH = PAGE_WIDTH - (2 * MARGIN_X)  # 539.27 pt
USABLE_HEIGHT = PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM  # 781.89 pt

# Color Palette: Technical Engineering Theme (White / Slate / Indigo)
C_PRIMARY = colors.HexColor('#1e40af')        # Indigo / Deep Blue
C_PRIMARY_LIGHT = colors.HexColor('#eff6ff')  # Soft Blue Tint (Cards/Headers)
C_PRIMARY_BORDER = colors.HexColor('#bfdbfe') # Subtle Blue Border
C_SECONDARY = colors.HexColor('#0f172a')      # Dark Slate Body Text
C_MUTED = colors.HexColor('#475569')          # Slate Muted / Metadata
C_LIGHT_BG = colors.HexColor('#f8fafc')       # Slate-50 Background
C_CARD_BG = colors.HexColor('#ffffff')
C_BORDER = colors.HexColor('#cbd5e1')         # Slate-300 Border
C_LINE = colors.HexColor('#e2e8f0')           # Slate-200 Subtle Divider

# Status Semantic Colors
C_SAFE = colors.HexColor('#166534')           # Green Dark
C_SAFE_BG = colors.HexColor('#f0fdf4')        # Green Tint
C_SAFE_BORDER = colors.HexColor('#bbf7d0')

C_WARN = colors.HexColor('#9a3412')           # Amber / Orange Dark
C_WARN_BG = colors.HexColor('#fffbeb')        # Amber Tint
C_WARN_BORDER = colors.HexColor('#fde68a')

C_CRIT = colors.HexColor('#991b1b')           # Red Dark
C_CRIT_BG = colors.HexColor('#fef2f2')        # Red Tint
C_CRIT_BORDER = colors.HexColor('#fecaca')

C_ACCENT = colors.HexColor('#4338ca')         # Deep Indigo Accent


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and render running headers and 'Page X of Y' footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, total_pages):
        self.saveState()
        
        # Running Footer (All pages)
        self.setFont('Helvetica', 7.5)
        self.setFillColor(C_MUTED)
        self.drawString(MARGIN_X, 15, "Universal AI Quantization Engine (UAQE) -- Complete 16-Phase Technical Learning Guide")
        self.drawRightString(PAGE_WIDTH - MARGIN_X, 15, f"Page {self._pageNumber} of {total_pages}")
        
        self.setStrokeColor(C_BORDER)
        self.setLineWidth(0.5)
        self.line(MARGIN_X, 23, PAGE_WIDTH - MARGIN_X, 23)
        
        # Running Header (Pages 2 to 8)
        if self._pageNumber > 1:
            self.setFont('Helvetica-Bold', 7.5)
            self.setFillColor(C_PRIMARY)
            self.drawString(MARGIN_X, PAGE_HEIGHT - 18, "UAQE TECHNICAL LEARNING GUIDE")
            self.setFont('Helvetica', 7.5)
            self.setFillColor(C_MUTED)
            self.drawString(MARGIN_X + 155, PAGE_HEIGHT - 18, "|  Autonomous Optimization, Hardware Verification & Rigorous Benchmarking")
            self.drawRightString(PAGE_WIDTH - MARGIN_X, PAGE_HEIGHT - 18, "ECE / AI Edge Engineering Reference")
            self.setStrokeColor(C_LINE)
            self.setLineWidth(0.5)
            self.line(MARGIN_X, PAGE_HEIGHT - 22, PAGE_WIDTH - MARGIN_X, PAGE_HEIGHT - 22)
            
        self.restoreState()


def build_pdf(filename="UAQE_16_Phase_Complete_Guide.pdf"):
    print(f"Starting PDF generation: {filename}")
    
    # Styles
    base_styles = getSampleStyleSheet()
    
    styles = {
        'DocTitle': ParagraphStyle(
            'DocTitle',
            parent=base_styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=15.5,
            leading=18.5,
            textColor=colors.HexColor('#0f172a'),
            alignment=0
        ),
        'DocSubtitle': ParagraphStyle(
            'DocSubtitle',
            parent=base_styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=8.5,
            leading=11,
            textColor=C_PRIMARY,
            alignment=0
        ),
        'MetaText': ParagraphStyle(
            'MetaText',
            parent=base_styles['Normal'],
            fontName='Helvetica',
            fontSize=7,
            leading=9,
            textColor=C_MUTED
        ),
        'SectionHeader': ParagraphStyle(
            'SectionHeader',
            parent=base_styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=9.2,
            leading=11.5,
            textColor=C_PRIMARY
        ),
        'SubHeader': ParagraphStyle(
            'SubHeader',
            parent=base_styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=8,
            leading=10,
            textColor=colors.HexColor('#1e293b')
        ),
        'Body': ParagraphStyle(
            'Body',
            parent=base_styles['Normal'],
            fontName='Helvetica',
            fontSize=7.1,
            leading=9.2,
            textColor=C_SECONDARY
        ),
        'BodyBold': ParagraphStyle(
            'BodyBold',
            parent=base_styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=7.1,
            leading=9.2,
            textColor=C_SECONDARY
        ),
        'BodyMuted': ParagraphStyle(
            'BodyMuted',
            parent=base_styles['Normal'],
            fontName='Helvetica',
            fontSize=6.8,
            leading=8.5,
            textColor=C_MUTED
        ),
        'Code': ParagraphStyle(
            'Code',
            parent=base_styles['Normal'],
            fontName='Courier-Bold',
            fontSize=6.8,
            leading=8.5,
            textColor=colors.HexColor('#0f172a')
        ),
        'Formula': ParagraphStyle(
            'Formula',
            parent=base_styles['Normal'],
            fontName='Courier-Bold',
            fontSize=8,
            leading=10,
            textColor=C_PRIMARY,
            alignment=1
        ),
        'TableHead': ParagraphStyle(
            'TableHead',
            parent=base_styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=7,
            leading=8.5,
            textColor=colors.HexColor('#0f172a')
        ),
        'TableCell': ParagraphStyle(
            'TableCell',
            parent=base_styles['Normal'],
            fontName='Helvetica',
            fontSize=6.7,
            leading=8.3,
            textColor=C_SECONDARY
        ),
        'TableCellBold': ParagraphStyle(
            'TableCellBold',
            parent=base_styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=6.7,
            leading=8.3,
            textColor=C_SECONDARY
        ),
        'CalloutTitle': ParagraphStyle(
            'CalloutTitle',
            parent=base_styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=7.2,
            leading=9,
            textColor=colors.HexColor('#0f172a')
        ),
        'CalloutBody': ParagraphStyle(
            'CalloutBody',
            parent=base_styles['Normal'],
            fontName='Helvetica',
            fontSize=6.8,
            leading=8.5,
            textColor=C_SECONDARY
        )
    }

    def callout_box(title, text, kind="important", width=USABLE_WIDTH):
        color_map = {
            "verified": (C_SAFE, C_SAFE_BG, C_SAFE_BORDER, "VERIFIED ARCHITECTURE / EMPIRICAL EVIDENCE"),
            "warning": (C_WARN, C_WARN_BG, C_WARN_BORDER, "CRITICAL WARNING / HARDWARE PITFALL"),
            "critical": (C_CRIT, C_CRIT_BG, C_CRIT_BORDER, "STRICT FAILURE STATE / REJECTION"),
            "remember": (C_ACCENT, C_PRIMARY_LIGHT, C_PRIMARY_BORDER, "CORE PRINCIPLE / REMEMBER"),
            "important": (C_PRIMARY, C_PRIMARY_LIGHT, C_PRIMARY_BORDER, "KEY ARCHITECTURAL CONCEPT")
        }
        title_color, bg_color, border_color, default_tag = color_map.get(kind, color_map["important"])
        tag = f"<b>[{default_tag if not title else title}]</b> "
        p = Paragraph(f"<font color='{title_color.hexval()}'>{tag}</font>{text}", styles['CalloutBody'])
        t = Table([[p]], colWidths=[width])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), bg_color),
            ('BOX', (0,0), (-1,-1), 0.5, border_color),
            ('LINELEFT', (0,0), (-1,-1), 2.5, title_color),
            ('TOPPADDING', (0,0), (-1,-1), 2.5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
            ('LEFTPADDING', (0,0), (-1,-1), 5),
            ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ]))
        return t

    def phase_badge(num, title):
        content = [
            [Paragraph(f"<b>PHASE {num:02d}</b>", styles['TableHead']),
             Paragraph(f"<b>{title.upper()}</b>", styles['SectionHeader'])]
        ]
        t = Table(content, colWidths=[62, USABLE_WIDTH - 62])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,0), C_PRIMARY_LIGHT),
            ('BACKGROUND', (1,0), (1,0), colors.HexColor('#f8fafc')),
            ('BOX', (0,0), (-1,-1), 0.5, C_PRIMARY_BORDER),
            ('LINEAFTER', (0,0), (0,0), 1, C_PRIMARY),
            ('ALIGN', (0,0), (0,0), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 2.2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2.2),
            ('LEFTPADDING', (0,0), (-1,-1), 4),
            ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ]))
        return t

    def flow_box(steps, width=USABLE_WIDTH):
        n = len(steps)
        cols = []
        col_w = []
        cell_w = (width - ((n-1)*14)) / n
        for i, step in enumerate(steps):
            cols.append(Paragraph(f"<font color='{C_PRIMARY.hexval()}'><b>{step}</b></font>", styles['TableCellBold']))
            col_w.append(cell_w)
            if i < n - 1:
                cols.append(Paragraph("<font color='#64748b'><b>&rarr;</b></font>", styles['TableCellBold']))
                col_w.append(14)
        t = Table([cols], colWidths=col_w)
        ts = [
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ('LEFTPADDING', (0,0), (-1,-1), 1),
            ('RIGHTPADDING', (0,0), (-1,-1), 1),
        ]
        for i in range(0, 2*n - 1, 2):
            ts.append(('BACKGROUND', (i,0), (i,0), colors.HexColor('#f8fafc')))
            ts.append(('BOX', (i,0), (i,0), 0.5, C_PRIMARY_BORDER))
        t.setStyle(TableStyle(ts))
        return t

    story = []

    # =========================================================================
    # PAGE 1 -- UAQE OVERVIEW + PHASES 1-3
    # =========================================================================
    title_table_data = [
        [
            Paragraph("<b>UNIVERSAL AI QUANTIZATION ENGINE (UAQE)</b>", styles['DocTitle']),
            Paragraph("<b>STUDY &amp; VIVA GUIDE</b><br/><font color='#64748b'>ECE / Edge-AI Core Reference</font>", styles['TableHead'])
        ],
        [
            Paragraph("<b>Complete 16-Phase Technical Learning Guide: Architecture, Mathematics &amp; Hardware Execution</b>", styles['DocSubtitle']),
            Paragraph("<font color='#64748b'>Status: Production-Grade | Real Evidence</font>", styles['MetaText'])
        ]
    ]
    t_title = Table(title_table_data, colWidths=[USABLE_WIDTH - 110, 110])
    t_title.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ALIGN', (1,0), (1,-1), 'RIGHT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ('TOPPADDING', (0,0), (-1,-1), 1),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_title)
    story.append(Spacer(1, 3))

    def_p = Paragraph(
        "<b>UAQE</b> is an autonomous, capability-driven AI model optimization and orchestration engine that: "
        "<b>(1)</b> accepts supported AI models and datasets via drag-and-drop ingestion, <b>(2)</b> inspects metadata, shapes, and task contracts, "
        "<b>(3)</b> establishes empirical FP32 baseline metrics, <b>(4)</b> autonomously generates candidate optimization strategies, "
        "<b>(5)</b> executes calibration-guided quantization (INT8), sensitivity-aware structured pruning, and lossless weight compression, "
        "<b>(6)</b> benchmarks pure inference latency, memory (RSS), and accuracy, <b>(7)</b> enforces strict percentage-point safety policies, "
        "<b>(8)</b> selects the verified Pareto winner, and <b>(9)</b> packages a cryptographically hashed, deployable edge bundle.",
        styles['Body']
    )
    t_def = Table([[def_p]], colWidths=[USABLE_WIDTH])
    t_def.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_PRIMARY_LIGHT),
        ('BOX', (0,0), (-1,-1), 0.75, C_PRIMARY_BORDER),
        ('LINELEFT', (0,0), (-1,-1), 3, C_PRIMARY),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_def)
    story.append(Spacer(1, 3))

    story.append(Paragraph("<b>THE FIVE CANONICAL CONCEPTUAL LAYERS OF UAQE</b>", styles['SubHeader']))
    story.append(Spacer(1, 1))
    layers_data = [
        [
            Paragraph("<b>1. UNDERSTAND</b>", styles['TableCellBold']),
            Paragraph("<b>2. TRANSFORM</b>", styles['TableCellBold']),
            Paragraph("<b>3. EXECUTE</b>", styles['TableCellBold']),
            Paragraph("<b>4. MEASURE</b>", styles['TableCellBold']),
            Paragraph("<b>5. DECIDE</b>", styles['TableCellBold'])
        ],
        [
            Paragraph("Inspect model graph, dtypes, task signature, and dataset distribution.", styles['TableCell']),
            Paragraph("Apply INT8 QDQ / TFLite quantization, pruning, and Huffman/RLE compression.", styles['TableCell']),
            Paragraph("Invoke target runtime (ONNXRuntime / TFLite Interpreter) on real input batches.", styles['TableCell']),
            Paragraph("Profile pure inference latency, throughput, RSS memory, and accuracy loss.", styles['TableCell']),
            Paragraph("Gating policy: classify candidates into EXCELLENT, ACCEPTABLE, or CRITICAL.", styles['TableCell'])
        ]
    ]
    w5 = USABLE_WIDTH / 5.0
    t_layers = Table(layers_data, colWidths=[w5]*5)
    t_layers.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_LINE),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_layers)
    story.append(Spacer(1, 3))

    story.append(flow_box(["FP32 TRAINED MODEL", "CALIBRATION & QUANTIZATION", "INT8 SERIALIZED ARTIFACT", "EDGE RUNTIME (SIMD/NEON)"]))
    story.append(Spacer(1, 3))

    # PHASE 1
    story.append(phase_badge(1, "Neural Network Fundamentals & Edge Compute Primitives"))
    story.append(Spacer(1, 2))
    p1_text = (
        "<b>Neuron Mathematical Formulation:</b> An artificial neuron evaluates <i>y = f(w &middot; x + b)</i>, where <b>w</b> is the weight vector, "
        "<b>x</b> is the input activation vector, <b>b</b> is the scalar bias, and <i>f(&middot;)</i> is a non-linear activation function. "
        "<b>Weights:</b> Learnable kernel parameters that capture spatial/semantic correlation. <b>Bias:</b> Learnable offset ensuring activation independence from origin. "
        "<b>Layers:</b> Successive functional blocks (Dense/Linear, Convolutional 2D, Depthwise Separable, Pooling, Normalization). "
        "<b>Tensors &amp; Shapes:</b> Multidimensional numerical arrays; e.g., PyTorch vision tensors use NCHW <i>[Batch, Channels, Height, Width]</i>, "
        "whereas edge runtimes (TFLite/Edge TPU) prefer NHWC <i>[Batch, Height, Width, Channels]</i> for contiguous memory spatial locality. "
        "<b>Data Types (dtype):</b> Bit representation determining dynamic range and arithmetic hardware path. "
        "<b>Activations &amp; Logits:</b> Unnormalized real outputs prior to normalization. <b>Softmax:</b> Normalizes logits into probabilities: "
        "<i>P(y<sub>i</sub>) = exp(z<sub>i</sub>) / &sum; exp(z<sub>j</sub>)</i>. "
        "<b>Convolution &amp; Feature Maps:</b> Sliding 2D kernel computing cross-correlations; output channels represent learned feature filters. "
        "<b>MAC (Multiply-Accumulate):</b> The fundamental atomic operation of deep learning: <i>Acc = Acc + (w<sub>i</sub> &times; x<sub>i</sub>)</i>. "
        "One MAC comprises 1 multiplication + 1 addition = 2 FLOPs. Edge optimization focuses directly on replacing 32-bit FP MACs with 8-bit INT MACs."
    )
    story.append(Paragraph(p1_text, styles['Body']))
    story.append(Spacer(1, 3))

    # PHASE 2
    story.append(phase_badge(2, "Model Storage, Serialization Formats & Graph Representation"))
    story.append(Spacer(1, 2))
    p2_text = (
        "<b>Unified Definition:</b> A deployable deep learning model strictly consists of: <b>Model = Architecture Graph + Parameter Weights + Serialization Metadata</b>.<br/>"
        "&bull; <b>.pt / .pth (PyTorch):</b> Pickled Python bytecode storing state dictionaries. High flexibility during training, but dangerous for edge deployment due to arbitrary code execution risks and lack of native embedded C++ runtime support.<br/>"
        "&bull; <b>SafeTensors:</b> Hugging Face zero-copy, memory-mapped tensor format. Safe against arbitrary code execution; contains no executable graph representation.<br/>"
        "&bull; <b>ONNX (Open Neural Network Exchange):</b> Protocol Buffer graph serialization representing operators as standard schemas. Ideal for server/desktop cross-platform inference via ONNX Runtime (ORT); uses QDQ nodes (QuantizeLinear / DequantizeLinear) for simulated and true INT8 execution.<br/>"
        "&bull; <b>TFLite (.tflite):</b> FlatBuffers-based binary format designed for microcontrollers, mobile, and edge Linux boards. Zero parsing overhead (memory-mappable directly from flash memory into RAM); stores per-channel quantization scale and zero-point vectors directly in operator headers.<br/>"
        "&bull; <b>Bare-Metal Embedded Formats (.bin / .mem / .hex):</b> Raw contiguous byte streams or hex dumps utilized by FPGA block RAM, DSPs, and custom ASIC DMA controllers.<br/>"
        "&bull; <b>Storage Representation vs. Runtime Representation:</b> On-disk compressed or pruned weights (e.g., zipped or sparse-indexed) occupy minimal flash storage, but must be decompressed or mapped into active DRAM tables upon invocation by runtime kernels.<br/>"
        "&bull; <b>Graph Metadata:</b> Embedded input/output tensor names, expected shapes, quantization scale/zero-point arrays, and operator domain versions."
    )
    story.append(Paragraph(p2_text, styles['Body']))
    story.append(Spacer(1, 3))

    # PHASE 3
    story.append(phase_badge(3, "Numerical Precision & Quantization Paradigms"))
    story.append(Spacer(1, 2))
    p3_table_data = [
        [Paragraph("<b>Precision</b>", styles['TableHead']), Paragraph("<b>Bits / Bytes</b>", styles['TableHead']), Paragraph("<b>Dynamic Range</b>", styles['TableHead']), Paragraph("<b>Edge Hardware Suitability &amp; UAQE Context</b>", styles['TableHead'])],
        [Paragraph("<b>FP32</b> (Single)", styles['TableCellBold']), Paragraph("32 b / 4 B", styles['TableCell']), Paragraph("1.18e-38 to 3.40e+38", styles['TableCell']), Paragraph("Baseline training precision; excessive memory bandwidth and energy on edge CPUs.", styles['TableCell'])],
        [Paragraph("<b>FP16</b> (Half)", styles['TableCellBold']), Paragraph("16 b / 2 B", styles['TableCell']), Paragraph("5.96e-8 to 65504", styles['TableCell']), Paragraph("Reduces memory by 50%; supported on modern mobile GPUs and Apple Neural Engine.", styles['TableCell'])],
        [Paragraph("<b>INT8</b> (Integer)", styles['TableCellBold']), Paragraph("8 b / 1 B", styles['TableCell']), Paragraph("[-128, 127] or [0, 255]", styles['TableCell']), Paragraph("<b>UAQE Core Target:</b> 4&times; memory footprint reduction, 2&ndash;4&times; compute throughput via SIMD.", styles['TableCell'])],
        [Paragraph("<b>INT4</b> (Sub-byte)", styles['TableCellBold']), Paragraph("4 b / 0.5 B", styles['TableCell']), Paragraph("[-8, 7] or [0, 15]", styles['TableCell']), Paragraph("High compression for LLM weight-only quantization; severe accuracy loss on small CNNs.", styles['TableCell'])],
    ]
    t_p3 = Table(p3_table_data, colWidths=[65, 55, 100, USABLE_WIDTH - 220])
    t_p3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_LINE),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_p3)
    story.append(Spacer(1, 2))
    p3_concepts = (
        "<b>Scale (S):</b> Positive real floating-point scalar mapping continuous real numbers into discrete integer steps: <i>S = (max - min) / (q<sub>max</sub> - q<sub>min</sub>)</i>.<br/>"
        "<b>Zero Point (Z):</b> Integer value corresponding exactly to real zero (0.0), guaranteeing that real zero (frequently padded in CNN convolutions) maps without error.<br/>"
        "<b>Symmetric vs. Asymmetric:</b> Symmetric sets <i>Z = 0</i> (maps [-max, +max] to [-127, 127]), eliminating zero-point cross-term math in MAC loops. "
        "Asymmetric sets <i>Z &ne; 0</i>, preserving high resolution for strictly non-negative activation distributions (e.g., ReLU outputs).<br/>"
        "<b>Per-Tensor vs. Per-Channel:</b> Per-tensor shares one (S, Z) across an entire layer. Per-channel assigns an individual scale <i>S<sub>c</sub></i> per output filter, "
        "preventing a single large outlier weight in one channel from compressing the numerical dynamic range of all other channels.<br/>"
        "<b>INT8 &times; INT8 &rarr; INT32 Accumulator:</b> Multiplying two 8-bit signed integers yields up to a 15-bit signed product (127 &times; 127 = 16,129). "
        "Summing these across 512 or 1024 channels easily overflows an 8-bit or 16-bit register. Edge hardware strictly utilizes <b>32-bit integer accumulators</b>."
    )
    story.append(Paragraph(p3_concepts, styles['Body']))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2 -- PHASES 4-6
    # =========================================================================
    story.append(phase_badge(4, "Quantization Mathematics, Numerical Formulations & Bias Scaling"))
    story.append(Spacer(1, 2))
    
    math_table_data = [
        [
            Paragraph("<b>Symmetric Quantization:</b><br/><font color='#1e40af'><b>q = round(x / S)</b></font><br/><b>Dequantization:</b><br/><font color='#1e40af'><b>x&#770; = q &times; S</b></font>", styles['TableCell']),
            Paragraph("<b>Asymmetric Quantization:</b><br/><font color='#1e40af'><b>q = clamp(round(x / S) + Z, q<sub>min</sub>, q<sub>max</sub>)</b></font><br/><b>Dequantization:</b><br/><font color='#1e40af'><b>x&#770; = (q - Z) &times; S</b></font>", styles['TableCell']),
            Paragraph("<b>Requantization Multiplier:</b><br/><font color='#1e40af'><b>M = (S<sub>x</sub> &times; S<sub>w</sub>) / S<sub>y</sub></b></font><br/><b>Fixed-Point Implementation:</b><br/><font color='#1e40af'><b>M &approx; M<sub>0</sub> &times; 2<sup>-n</sup></b></font>", styles['TableCell'])
        ]
    ]
    t_math = Table(math_table_data, colWidths=[USABLE_WIDTH/3.0]*3)
    t_math.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 0.5, C_PRIMARY_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_LINE),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_math)
    story.append(Spacer(1, 3))

    p4_details = (
        "<b>Mathematical Parameter Definitions:</b><br/>"
        "&bull; <b>Scale Factor (S):</b> Determined by dynamic range clipping: <i>S = (x<sub>max</sub> - x<sub>min</sub>) / (q<sub>max</sub> - q<sub>min</sub>)</i>.<br/>"
        "&bull; <b>Zero Point (Z):</b> Integer alignment: <i>Z = round(-x<sub>min</sub> / S) + q<sub>min</sub></i>. Guarantees exact representation of 0.0 to prevent zero-padding errors.<br/>"
        "&bull; <b>Clipping, Saturation &amp; Rounding:</b> Floating values outside <i>[x<sub>min</sub>, x<sub>max</sub>]</i> are clamped to avoid integer wrap-around. Nearest integer rounding introduces round-off error <i>&epsilon; &le; S/2</i>.<br/>"
        "&bull; <b>Outlier Management:</b> Activations often follow heavy-tailed Gaussian distributions. A single extreme outlier will drastically inflate <i>S</i>, collapsing 99% of normal weights into 2 or 3 discrete integer levels. UAQE handles this through percentile calibration and per-channel weight scaling.<br/>"
        "&bull; <b>Bias Quantization Math:</b> In a convolution layer, the accumulator computes: <i>Accumulator = &sum; (x<sub>q</sub> &times; w<sub>q</sub>)</i>. "
        "The effective scale of each product is <i>S<sub>x</sub> &times; S<sub>w</sub></i>. To add the bias directly to the accumulator without expensive runtime alignment, "
        "<b>the bias is quantized with scale S<sub>bias</sub> = S<sub>x</sub> &times; S<sub>w</sub></b> and stored as a <b>32-bit integer (INT32)</b>.<br/>"
        "&bull; <b>Requantization Down-Scaling:</b> The 32-bit accumulated sum must be converted back to an 8-bit integer activation <i>y<sub>q</sub></i> with scale <i>S<sub>y</sub></i> before passing to the next layer: "
        "<i>y<sub>q</sub> = round(Accumulator &times; M) + Z<sub>y</sub></i>. To avoid floating-point hardware on edge chips, <i>M</i> is factored into a 32-bit fixed-point integer multiplier <i>M<sub>0</sub> &isin; [0.5, 1.0) &times; 2<sup>31</sup></i> and a right-shift exponent <i>n</i>."
    )
    story.append(Paragraph(p4_details, styles['Body']))
    story.append(Spacer(1, 3))

    # PHASE 5
    story.append(phase_badge(5, "Calibration Methodologies, PTQ & Quantization-Aware Training (QAT)"))
    story.append(Spacer(1, 2))
    
    flow_ptq = flow_box(["FP32 PRETRAINED MODEL", "REPRESENTATIVE CALIBRATION DATA", "OBSERVERS (MinMax / Histogram)", "INT8 ARTIFACT EXPORT"])
    story.append(flow_ptq)
    story.append(Spacer(1, 2))
    flow_qat = flow_box(["FP32 TRAINED MODEL", "INSERT FAKE-QUANT NODES", "FINETUNE (STE BACKPROP)", "INT8 FROZEN EXPORT"])
    story.append(flow_qat)
    story.append(Spacer(1, 2))

    p5_details = (
        "<b>Calibration:</b> The mathematical process of feeding representative unlabelled input data through the model to record activation distributions, "
        "enabling optimal selection of scale <i>S</i> and zero-point <i>Z</i> without modifying weights.<br/>"
        "&bull; <b>Observers:</b> Profiling modules placed at tensor boundaries. <i>MinMaxObserver</i> tracks absolute bounds [min, max] (fast, but susceptible to outliers). "
        "<i>Histogram / Percentile Observer</i> clips the top 0.01% extreme outliers. <i>Entropy / KL-Divergence Observer</i> minimizes the Kullback-Leibler divergence "
        "between the FP32 distribution and the quantized INT8 distribution (widely adopted by TensorRT).<br/>"
        "&bull; <b>Representative Dataset:</b> A curated subset of calibration samples (typically 100&ndash;500 images) matching the target deployment sensor domain.<br/>"
        "&bull; <b>Weight Calibration:</b> Static and deterministic; computed directly from frozen weight tensors without requiring data batches.<br/>"
        "&bull; <b>Activation Calibration:</b> Dynamic and data-dependent; must capture realistic input variance.<br/>"
        "&bull; <b>Fake Quantization in QAT:</b> Nodes inserted during training that discretize activations and weights to simulate INT8 rounding error during the forward pass, "
        "while storing gradients and parameters in FP32: <i>x<sub>fake</sub> = S &times; (clamp(round(x/S) + Z, q<sub>min</sub>, q<sub>max</sub>) - Z)</i>.<br/>"
        "&bull; <b>Straight-Through Estimator (STE):</b> Because the derivative of the step-rounding function is 0 everywhere (and undefined at boundaries), "
        "standard backpropagation cannot train quantized networks. The STE replaces the non-differentiable gradient with an identity pass-through: "
        "<i>&part; round(x) / &part; x &approx; 1</i> for <i>x &isin; [x<sub>min</sub>, x<sub>max</sub>]</i>, and 0 outside, allowing full gradient descent."
    )
    story.append(Paragraph(p5_details, styles['Body']))
    story.append(Spacer(1, 3))

    # PHASE 6
    story.append(phase_badge(6, "PTQ vs QAT in UAQE: Production Workflows & Performance Decoupling"))
    story.append(Spacer(1, 2))
    
    p6_table_data = [
        [Paragraph("<b>Evaluation Dimension</b>", styles['TableHead']), Paragraph("<b>Post-Training Quantization (PTQ)</b>", styles['TableHead']), Paragraph("<b>Quantization-Aware Training (QAT)</b>", styles['TableHead']), Paragraph("<b>UAQE Architectural Role &amp; Findings</b>", styles['TableHead'])],
        [Paragraph("<b>Retraining Required?</b>", styles['TableCellBold']), Paragraph("No &mdash; zero parameter retraining", styles['TableCell']), Paragraph("Yes &mdash; fine-tuning for several epochs", styles['TableCell']), Paragraph("PTQ offers immediate candidate evaluation (seconds vs hours).", styles['TableCell'])],
        [Paragraph("<b>Compute / Time Cost</b>", styles['TableCellBold']), Paragraph("Very low (seconds on CPU)", styles['TableCell']), Paragraph("High (requires GPU training loop)", styles['TableCell']), Paragraph("UAQE autonomous search runs PTQ exploration passes first.", styles['TableCell'])],
        [Paragraph("<b>Accuracy Preservation</b>", styles['TableCellBold']), Paragraph("Good for large models; drops on small CNNs", styles['TableCell']), Paragraph("Superior &mdash; recovers &gt;99% of FP32 baseline", styles['TableCell']), Paragraph("QAT used when PTQ exceeds critical loss threshold (&gt;4.0 pp).", styles['TableCell'])],
        [Paragraph("<b>Verified Project Usage</b>", styles['TableCellBold']), Paragraph("<b>ResNet-50 + CIFAR-10:</b> ONNX QDQ static PTQ", styles['TableCell']), Paragraph("<b>MobileNet:</b> True INT8 TFLite (QAT history)", styles['TableCell']), Paragraph("Verified: ResNet ONNX QDQ achieves robust PTQ accuracy.", styles['TableCell'])],
    ]
    t_p6 = Table(p6_table_data, colWidths=[90, 115, 115, USABLE_WIDTH - 320])
    t_p6.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_LINE),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_p6)
    story.append(Spacer(1, 2))

    story.append(callout_box(
        "CRITICAL DISTINCTION: ACCURACY RECOVERY VS. RUNTIME SPEEDUP",
        "<b>QAT DOES NOT AUTOMATICALLY GUARANTEE FASTER INFERENCE.</b> "
        "QAT only alters weight distributions to tolerate precision loss; it produces the exact same INT8 tensor format as PTQ. "
        "Actual runtime inference speedup depends <i>strictly</i> on execution kernels (e.g. XNNPACK, AVX-512 VNNI, ARM NEON), memory bandwidth, "
        "cache hit rates, and tensor layout transforms (NCHW &harr; NHWC). Quantization accuracy and runtime performance are completely decoupled engineering concerns.",
        "warning"
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3 -- PHASES 7-9
    # =========================================================================
    story.append(phase_badge(7, "Actual UAQE Quantization Engine Architecture & Execution Topology"))
    story.append(Spacer(1, 2))
    
    r1 = ["MODEL UPLOAD", "MODEL ADAPTER", "INSPECTION", "DATASET ADAPTER", "TASK DETECTION", "COMPATIBILITY", "BASELINE VALID"]
    r2 = ["AUTO PLAN", "CANDIDATE GEN", "OPTIMIZATION", "BENCHMARK", "SAFETY GATING", "WINNER SELECTION", "DEPLOY PACKAGE"]
    story.append(flow_box(r1))
    story.append(Spacer(1, 1))
    story.append(flow_box(r2))
    story.append(Spacer(1, 3))

    p7_text = (
        "<b>Architectural Subsystems &amp; Core Engineering Principles:</b><br/>"
        "&bull; <b>Model Adapters:</b> Abstract away framework-specific details (PyTorch, ONNX, TFLite), exposing a uniform graph interface for inspection, node traversal, and parameter extraction.<br/>"
        "&bull; <b>Dataset Adapters:</b> Ingest heterogeneous inputs (ImageFolder directory structures, CIFAR-10 binary pickles, single images), enforcing strict input shape resolution, normalization, and tensor batching.<br/>"
        "&bull; <b>Optimization Planner &amp; Orchestrator:</b> Evaluates target hardware capabilities against model architecture, generating a deterministic matrix of candidate optimization strategies.<br/>"
        "&bull; <b>Candidate Evaluator &amp; Job Isolation:</b> Every candidate executes in a dedicated sandbox directory (<code>output/jobs/&lt;id&gt;</code>). Telemetry and metrics are logged independently, completely preventing state leakage across candidate runs.<br/>"
        "&bull; <b>Baseline Validity Preflight:</b> Before any optimization begins, the unquantized FP32 model is evaluated on the validation dataset. If the baseline fails to produce valid outputs or accuracy, the entire pipeline immediately halts.<br/>"
        "&bull; <b>Capability-Driven Architecture (Universal &ne; Every Architecture):</b> UAQE operates on verified operator capabilities. "
        "It supports verified CNN pipelines (MobileNet, ResNet-50) while cleanly detecting and rejecting unsupported architectures (e.g., Vision Transformers)."
    )
    story.append(Paragraph(p7_text, styles['Body']))
    story.append(Spacer(1, 3))

    # PHASE 8
    story.append(phase_badge(8, "Deep INT8 Inference, Micro-Architecture & Kernel Execution"))
    story.append(Spacer(1, 2))
    story.append(flow_box(["INT8 INPUT TENSOR", "INT8 WEIGHT TENSOR", "INT32 ACCUMULATOR", "REQUANTIZATION (Fixed Mult+Shift)", "INT8 OUTPUT TENSOR"]))
    story.append(Spacer(1, 2))

    p8_text = (
        "<b>Micro-Architectural Mechanics of INT8 Vector Execution:</b><br/>"
        "&bull; <b>Convolution &amp; MAC Loop:</b> An edge CPU processes convolution by unfolding input patches into contiguous memory (im2col) and executing General Matrix Multiply (GEMM). "
        "Four 8-bit integers are packed into a single 32-bit register and multiplied in parallel.<br/>"
        "&bull; <b>SIMD Acceleration (ARM NEON &amp; x86 VNNI):</b> Modern edge cores feature SIMD vector registers (128-bit NEON on Cortex-A76, 256-bit AVX2 / 512-bit AVX-512 on x86). "
        "Instructions such as ARM <code>SDOT</code> (Signed Dot Product) multiply four pairs of INT8 values and accumulate into a 32-bit vector lane in a single clock cycle.<br/>"
        "&bull; <b>XNNPACK Operator Concept:</b> A highly tuned assembly micro-kernel library targeting ARM, x86, and WebAssembly. It mandates NHWC memory layout and packs weights into specialized cache-friendly blocks.<br/>"
        "&bull; <b>Cache Hierarchy &amp; Memory Bandwidth:</b> In FP32 inference, fetching 4-byte weights frequently saturates external DRAM bus bandwidth, causing CPU stalls. INT8 reduces memory traffic by 75%, allowing entire model layer weights to reside within fast L2/L3 cache.<br/>"
        "&bull; <b>Cold-Start vs. Warm Inference:</b> Cold-start latency includes graph loading, memory allocation (<code>allocate_tensors</code>), kernel compilation, and instruction cache warming. "
        "Warm inference measures steady-state execution after memory buffers and caches are fully primed."
    )
    story.append(Paragraph(p8_text, styles['Body']))
    story.append(Spacer(1, 3))

    # PHASE 9
    story.append(phase_badge(9, "Pruning, Sparsity Paradigms & Lossless Artifact Compression"))
    story.append(Spacer(1, 2))
    
    p9_table_data = [
        [Paragraph("<b>Pruning Paradigm</b>", styles['TableHead']), Paragraph("<b>Sparsity Pattern</b>", styles['TableHead']), Paragraph("<b>Hardware Acceleration Feasibility</b>", styles['TableHead']), Paragraph("<b>Storage vs. Runtime Impact</b>", styles['TableHead'])],
        [Paragraph("<b>Unstructured Pruning</b>", styles['TableCellBold']), Paragraph("Fine-grained, random zero weights throughout tensor.", styles['TableCell']), Paragraph("Poor &mdash; standard edge CPUs cannot skip random zeros; requires dense compute.", styles['TableCell']), Paragraph("High storage compression (via gzip/RLE); 0% speedup on standard CPUs.", styles['TableCell'])],
        [Paragraph("<b>Structured Pruning</b>", styles['TableCellBold']), Paragraph("Coarse-grained; removes entire channels or filters.", styles['TableCell']), Paragraph("<b>Excellent</b> &mdash; directly shrinks matrix dimensions; native speedup on any engine.", styles['TableCell']), Paragraph("Reduces both storage size AND runtime latency proportionately.", styles['TableCell'])],
        [Paragraph("<b>Lossless Compression</b>", styles['TableCellBold']), Paragraph("Huffman / Deflate / Run-Length Encoding (RLE).", styles['TableCell']), Paragraph("Applied during on-disk packaging; requires decompression pass before inference.", styles['TableCell']), Paragraph("Reduces flash memory footprint; does NOT alter runtime model RAM.", styles['TableCell'])],
    ]
    t_p9 = Table(p9_table_data, colWidths=[90, 115, 115, USABLE_WIDTH - 320])
    t_p9.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_LINE),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_p9)
    story.append(Spacer(1, 2))

    story.append(callout_box(
        "CARDINAL RULE OF COMPRESSION",
        "<b>SPARSITY % &ne; STORAGE REDUCTION % &ne; RUNTIME LATENCY SPEEDUP %.</b> "
        "A model with 50% unstructured weight sparsity still requires 100% of dense floating-point or integer operations on standard ARM Cortex-A CPU cores. "
        "Furthermore, uncompressed zeros still consume 1 byte each in uncompressed flat file formats. "
        "Speedup is achieved only when structured channel pruning reduces matrix dimensions or when runtime kernels natively support sparse block indexing.",
        "remember"
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4 -- PHASES 10-12
    # =========================================================================
    story.append(phase_badge(10, "Model Compression & Artifact Engineering: Storage vs. Runtime Representation"))
    story.append(Spacer(1, 2))

    p10_text = (
        "<b>Artifact Engineering Principles &amp; Packaging Lifecycle:</b><br/>"
        "&bull; <b>Deployable Artifact:</b> A standalone, self-contained deployment bundle containing the runtime model binary, cryptographic verification hashes, hardware metadata, and execution contracts.<br/>"
        "&bull; <b>Storage Representation vs. Runtime Representation (Verified UAQE Empirical Finding):</b> "
        "An optimized model packaged with sensitivity-aware pruning and lossless stream compression can occupy <b>1.39 MB</b> on disk. "
        "However, when loaded into the edge runtime (e.g. TFLite Interpreter), the flat binary tensor arrays expand into their full uncompressed representation of <b>1.86 MB</b> in RAM. "
        "Engineers must never confuse compressed distribution archive size with active memory footprint.<br/>"
        "&bull; <b>Reconstruction &amp; Decoder Overhead:</b> Unpacking compressed archives introduces a small CPU and RAM overhead during cold boot. UAQE profiles this reconstruction latency to verify that edge startup limits are strictly respected.<br/>"
        "&bull; <b>Cryptographic Provenance (SHA-256):</b> Every raw model file, calibration dataset slice, intermediate candidate, and final deployment package is hashed with SHA-256. "
        "The resulting checksums are embedded in a tamper-proof <code>manifest.json</code> to guarantee auditability and supply chain integrity."
    )
    story.append(Paragraph(p10_text, styles['Body']))
    story.append(Spacer(1, 3))

    # PHASE 11
    story.append(phase_badge(11, "Benchmarking & Performance Engineering: Latency, Throughput & Telemetry Isolation"))
    story.append(Spacer(1, 2))

    bench_formulas = [
        [
            Paragraph("<b>Throughput (Frames Per Second):</b><br/><font color='#1e40af'><b>Throughput = 1000 / Latency<sub>ms</sub></b></font>", styles['TableCellBold']),
            Paragraph("<b>Latency Speedup Percentage:</b><br/><font color='#1e40af'><b>Speedup % = ((Baseline Latency - Optimized Latency) / Baseline Latency) &times; 100</b></font>", styles['TableCellBold'])
        ]
    ]
    t_bf = Table(bench_formulas, colWidths=[USABLE_WIDTH/2.0]*2)
    t_bf.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_LINE),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_bf)
    story.append(Spacer(1, 2))

    p11_text = (
        "<b>Rigorous Performance Measurement Methodology:</b><br/>"
        "&bull; <b>Pure Inference vs. End-to-End Latency:</b> Pure inference measures only model graph compute (<code>interpreter.invoke()</code>). "
        "End-to-end latency includes image decoding, resizing, RGB normalization, tensor copy, and argmax postprocessing. Both must be reported separately.<br/>"
        "&bull; <b>Statistical Rigor:</b> A single timing run is meaningless. UAQE discards initial warmup passes (cache priming), then samples 50&ndash;200 timed iterations to calculate <b>Mean, Median (P50), P95, and P99 tail latency</b>.<br/>"
        "&bull; <b>Memory Profiling:</b> Tracks OS Resident Set Size (RSS) and Peak Allocation using platform system calls, isolating memory consumption from garbage collection spikes.<br/>"
        "&bull; <b>Telemetry Contamination Avoidance:</b> Profiling hooks must not run inside the timing loop. Telemetry collection is isolated to dedicated sampling threads to prevent observer overhead.<br/>"
        "&bull; <b>Verified MobileNet XNNPACK Compatibility Finding:</b> INT8 is NOT automatically faster. If a runtime delegate encounters unsupported tensor layouts or missing kernel patterns, it falls back to slow reference kernels. "
        "In our verified testbed, the MobileNet INT8 model runs on the verified stable reference TFLite path rather than assuming XNNPACK acceleration."
    )
    story.append(Paragraph(p11_text, styles['Body']))
    story.append(Spacer(1, 3))

    # PHASE 12
    story.append(phase_badge(12, "Autonomous Optimization, Candidate Search & Safety Policy Gating"))
    story.append(Spacer(1, 2))

    safety_table_data = [
        [Paragraph("<b>Classification</b>", styles['TableHead']), Paragraph("<b>Accuracy Loss Threshold</b>", styles['TableHead']), Paragraph("<b>UAQE Engine Action &amp; Candidate Status</b>", styles['TableHead'])],
        [Paragraph("<font color='#166534'><b>EXCELLENT</b></font>", styles['TableCellBold']), Paragraph("<b>&le; 1.00 percentage point (pp)</b>", styles['TableCell']), Paragraph("Fully qualified candidate; eligible for Pareto winner selection.", styles['TableCell'])],
        [Paragraph("<font color='#9a3412'><b>ACCEPTABLE</b></font>", styles['TableCellBold']), Paragraph("<b>&gt; 1.00 pp and &le; 4.00 pp</b>", styles['TableCell']), Paragraph("Conditionally qualified; selected only if EXCELLENT candidates are unavailable.", styles['TableCell'])],
        [Paragraph("<font color='#991b1b'><b>CRITICAL</b></font>", styles['TableCellBold']), Paragraph("<b>&gt; 4.00 percentage points (pp)</b>", styles['TableCell']), Paragraph("<b>STRICT REJECTION;</b> barred from selection when any valid safer candidate exists.", styles['TableCell'])],
    ]
    t_safety = Table(safety_table_data, colWidths=[80, 130, USABLE_WIDTH - 210])
    t_safety.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_LINE),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_safety)
    story.append(Spacer(1, 2))

    p12_text = (
        "<b>Autonomous Search Mechanics:</b><br/>"
        "&bull; <b>Candidate Generation:</b> Explores permutations of quantization strategies (dynamic INT8, static per-tensor PTQ, per-channel PTQ, QAT) and structured pruning ratios (10%, 20%, 30%).<br/>"
        "&bull; <b>Pareto Frontier Evaluation:</b> Multi-objective optimization balancing accuracy retention, latency reduction, and memory footprint. A candidate dominates another if it is superior in at least one metric without degrading others.<br/>"
        "&bull; <b>Safety Policy Invariant:</b> A CRITICAL candidate (&gt;4.0 pp accuracy degradation) will NEVER be declared the winner, even if it achieves 10&times; speedup or 90% size reduction.<br/>"
        "&bull; <b>No-Valid-Candidate Fallback:</b> If all explored candidates breach the safety threshold, UAQE flags baseline fallback, logs full diagnostics, and halts deployment packaging."
    )
    story.append(Paragraph(p12_text, styles['Body']))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 5 -- PHASES 13-14
    # =========================================================================
    story.append(phase_badge(13, "Deployment Engineering, Packaging & 4-Level Runtime Verification"))
    story.append(Spacer(1, 2))

    story.append(flow_box(["GENERATE ARTIFACT", "VALIDATE FILE", "SHA-256 HASH", "PACKAGE BUNDLE", "RELOAD ENGINE", "TEST EXECUTE", "VERIFY OUTPUTS"]))
    story.append(Spacer(1, 2))

    p13_text = (
        "<b>The Four Rigorous Validation Levels:</b><br/>"
        "Before any optimization candidate is designated as a production-ready deployable artifact, it must successfully pass four sequential verification gates:<br/>"
        "&bull; <b>Level 1 &mdash; File Integrity Gate:</b> Byte-level inspection ensuring the artifact exists, is non-empty, and its SHA-256 cryptographic digest matches the manifest registry.<br/>"
        "&bull; <b>Level 2 &mdash; Decode &amp; Reconstruction Correctness:</b> Decompresses or unpacks the artifact into memory, verifying that headers, magic numbers, and operator offsets are uncorrupted.<br/>"
        "&bull; <b>Level 3 &mdash; Model Correctness:</b> Loads the model into the target runtime graph engine (TFLite / ONNX Runtime), validating that all quantized operators are supported by the runtime delegate.<br/>"
        "&bull; <b>Level 4 &mdash; Deployment &amp; Runtime Correctness:</b> Executes end-to-end inference using real calibration inputs. Verifies that output tensor dimensions, dtypes, and prediction distributions match baseline expectations without NaN or infinity faults.<br/>"
        "&bull; <b>Deployable Package Contents:</b> The final zip archive contains: (1) <code>optimized_model.onnx</code> or <code>.tflite</code>, (2) <code>manifest.json</code> (provenance, hardware profile, parameters), (3) <code>metrics.json</code> (empirical benchmark results), and (4) <code>input_contract.json</code> (normalization scales, shapes)."
    )
    story.append(Paragraph(p13_text, styles['Body']))
    story.append(Spacer(1, 3))

    # PHASE 14
    story.append(phase_badge(14, "Hardware Awareness, Target Registries & Truth-in-Engineering"))
    story.append(Spacer(1, 2))

    p14_table_data = [
        [Paragraph("<b>Target Hardware Profile</b>", styles['TableHead']), Paragraph("<b>CPU / Architecture</b>", styles['TableHead']), Paragraph("<b>SIMD / Vector Extensions</b>", styles['TableHead']), Paragraph("<b>UAQE Validation Status &amp; Evidence</b>", styles['TableHead'])],
        [Paragraph("<b>Host Development Platform</b>", styles['TableCellBold']), Paragraph("x86_64 / Modern Multi-Core", styles['TableCell']), Paragraph("AVX2 / AVX-512 VNNI", styles['TableCell']), Paragraph("<font color='#166534'><b>VERIFIED &amp; MEASURED:</b></font> Fully benchmarked on local testbed.", styles['TableCell'])],
        [Paragraph("<b>Target Profile: Raspberry Pi 5</b>", styles['TableCellBold']), Paragraph("ARMv8-A (Broadcom BCM2712, Quad-Core Cortex-A76 @ 2.4 GHz)", styles['TableCell']), Paragraph("ARM NEON / Dot Product (ASIMD)", styles['TableCell']), Paragraph("<font color='#9a3412'><b>PENDING PHYSICAL VALIDATION:</b></font> Cross-compiled &amp; profiled for ARM; on-device hardware run pending.", styles['TableCell'])],
        [Paragraph("<b>Unsupported Architecture: Vision Transformer (ViT-Large)</b>", styles['TableCellBold']), Paragraph("Multi-Head Self-Attention on ImageNet", styles['TableCell']), Paragraph("N/A", styles['TableCell']), Paragraph("<font color='#991b1b'><b>REJECTED SAFELY:</b></font> Accurately detected as unsupported by CNN engine; halted cleanly.", styles['TableCell'])],
    ]
    t_p14 = Table(p14_table_data, colWidths=[100, 120, 95, USABLE_WIDTH - 315])
    t_p14.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_LINE),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_p14)
    story.append(Spacer(1, 2))

    p14_text = (
        "<b>Core Hardware Awareness Principles:</b><br/>"
        "&bull; <b>Target Hardware Registry:</b> Maintains hardware parameter models (CPU family, SIMD vector width, L1/L2 cache capacities, thermal throttling ceilings). "
        "The planner utilizes these parameters to prune candidate search spaces that would fail on target chips.<br/>"
        "&bull; <b>Truth in Engineering &amp; Scope Definition:</b> 'Universal' denotes a generalized, capability-driven orchestration framework&mdash;<b>NOT</b> a claim of magical compatibility with every conceivable model architecture. "
        "UAQE strictly isolates supported models from unsupported models.<br/>"
        "&bull; <b>Negative Testing &amp; Vision Transformers:</b> When ViT-Large was ingested, UAQE's compatibility analyzer correctly identified that the attention operators fell outside the verified CNN optimization backend. "
        "Rather than emitting a corrupted model or inventing fake metrics, the engine safely aborted the pipeline with a descriptive error report."
    )
    story.append(Paragraph(p14_text, styles['Body']))
    story.append(Spacer(1, 2))

    story.append(callout_box(
        "VIVA & AUDIT REQUIREMENT: HOST VS. TARGET HARDWARE DISTINCTION",
        "<b>DO NOT CLAIM CURRENT RASPBERRY PI MEASUREMENTS AS COMPLETED.</b> "
        "In UAQE technical documentation and viva defense, engineers must strictly state: "
        "<b>Host validation = MEASURED</b> (benchmarked on host x86 workstation); "
        "<b>Physical Raspberry Pi 5 validation = PENDING</b> (target-specific packaging verified, on-board bench test scheduled). "
        "Academic and industrial trust relies on absolute transparency regarding what has been physically measured versus simulated.",
        "warning"
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 6 -- PHASES 15-16
    # =========================================================================
    story.append(phase_badge(15, "Observability, Telemetry Architecture & Live Pipeline Explainability"))
    story.append(Spacer(1, 2))

    story.append(flow_box(["INITIALIZING", "INGESTING", "INSPECTING", "CALIBRATING", "PROFILING", "SEARCHING"]))
    story.append(Spacer(1, 1))
    story.append(flow_box(["EVALUATING", "SELECTING", "VALIDATING", "PACKAGING", "COMPLETED"]))
    story.append(Spacer(1, 3))

    p15_text = (
        "<b>Canonical Pipeline State Machine &amp; Real-Time Telemetry:</b><br/>"
        "The UAQE optimization lifecycle is governed by an explicit 11-stage state machine. Every stage emits structured events over Server-Sent Events (SSE) to the browser cockpit.<br/>"
        "&bull; <b>Telemetry Dimensions:</b> Live monitoring tracks: <i>System CPU %</i>, <i>Isolated Process CPU %</i>, <i>Current RAM RSS (MB)</i>, <i>Peak RSS (MB)</i>, and active candidate iteration index.<br/>"
        "&bull; <b>Single Source of Truth:</b> Backend telemetry logs (<code>telemetry.json</code>) drive the frontend UI directly. "
        "<b>Zero Synthetic Metrics Policy:</b> The frontend is strictly forbidden from fabricating mock latencies or simulated memory curves. If the backend fails or disconnects, the UI must display error boundaries rather than fake progress.<br/>"
        "&bull; <b>Post-Mortem Replay &amp; Explainability:</b> Every optimization run serializes full timeline event logs, enabling engineers to reconstruct exact candidate execution timelines, memory spikes, and pruning decisions."
    )
    story.append(Paragraph(p15_text, styles['Body']))
    story.append(Spacer(1, 3))

    # PHASE 16
    story.append(phase_badge(16, "Testing, Multi-Tier Verification & Trust Engineering"))
    story.append(Spacer(1, 2))

    p16_table_data = [
        [Paragraph("<b>Testing Tier</b>", styles['TableHead']), Paragraph("<b>Scope &amp; Verification Methodology</b>", styles['TableHead']), Paragraph("<b>Passing Criteria / UAQE Standard</b>", styles['TableHead'])],
        [Paragraph("<b>Unit Tests</b>", styles['TableCellBold']), Paragraph("Quantization formulas, scale computation, clamping, tensor adapters.", styles['TableCell']), Paragraph("100% mathematical precision; zero numerical drift.", styles['TableCell'])],
        [Paragraph("<b>Integration Tests</b>", styles['TableCellBold']), Paragraph("Model ingest &rarr; calibrate &rarr; optimize &rarr; package pipeline loops.", styles['TableCell']), Paragraph("All 267 backend tests pass completely in 120.1s.", styles['TableCell'])],
        [Paragraph("<b>API Contract Tests</b>", styles['TableCellBold']), Paragraph("FastAPI endpoints (<code>/api/jobs/optimize</code>, <code>/api/status</code>).", styles['TableCell']), Paragraph("Strict Pydantic schema validation; zero 500 errors.", styles['TableCell'])],
        [Paragraph("<b>Browser E2E Tests</b>", styles['TableCellBold']), Paragraph("Playwright headless testing of React UI, drag-and-drop, and cockpit charts.", styles['TableCell']), Paragraph("No unhandled exceptions; live SSE streams render cleanly.", styles['TableCell'])],
        [Paragraph("<b>Negative Testing</b>", styles['TableCellBold']), Paragraph("Corrupted ONNX files, invalid image formats, out-of-memory triggers.", styles['TableCell']), Paragraph("System fails gracefully with structured diagnostic logs.", styles['TableCell'])],
    ]
    t_p16 = Table(p16_table_data, colWidths=[90, 160, USABLE_WIDTH - 250])
    t_p16.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_LINE),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_p16)
    story.append(Spacer(1, 2))

    p16_text = (
        "<b>The Cardinal Axioms of Trust Engineering:</b><br/>"
        "&bull; <b>Verification vs. Validation:</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&bull; <i>Verification:</i> <b>'Did we build the system correctly?'</b> &mdash; Ensures code satisfies specifications, types, API schemas, and unit test invariants.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&bull; <i>Validation:</i> <b>'Did we build the right system?'</b> &mdash; Confirms the optimized model accurately performs its real-world inference task on target hardware.<br/>"
        "&bull; <b>Empirical Evidence:</b> Trust is not established by claims or promotional benchmarks, but by reproducible, non-destructive automated test suites, SHA-256 provenance manifests, and zero-mock verification gates."
    )
    story.append(Paragraph(p16_text, styles['Body']))
    story.append(Spacer(1, 2))

    story.append(callout_box(
        "SUMMARY OF VERIFIED PROJECT HEALTH",
        "The core UAQE optimization engine is 100% verified and operational: all 267 backend automated tests pass completely in 120.157 seconds. "
        "The complete optimization pipeline (model ingestion, task detection, CIFAR-10 parsing, ONNX QDQ static quantization, structured pruning, and deployable artifact generation) "
        "executes with deterministic mathematical accuracy.",
        "verified"
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 7 -- ONE COMPLETE MATHEMATICAL PROBLEM
    # =========================================================================
    story.append(Paragraph("<b>INTEGRATED END-TO-END MATHEMATICAL DERIVATION</b>", styles['SectionHeader']))
    story.append(Paragraph("<b>Comprehensive Quantitative Optimization Analysis for Edge CNN Deployment (Parts A through J)</b>", styles['DocSubtitle']))
    story.append(Spacer(1, 2))

    prob_p = Paragraph(
        "<b>PROBLEM STATEMENT:</b> A production Convolutional Neural Network (CNN) is optimized for edge deployment using UAQE.<br/>"
        "&bull; <b>Original FP32 Baseline:</b> Model Size = <b>6.12 MB</b>, Top-1 Accuracy = <b>98.00%</b>, Inference Latency = <b>60.0 ms/image</b>, Peak Execution RAM = <b>500 MB</b>.<br/>"
        "&bull; <b>After Quantization Pass:</b> INT8 Model Size = <b>1.86 MB</b>, Top-1 Accuracy = <b>97.80%</b>, Inference Latency = <b>40.0 ms/image</b>, Peak Execution RAM = <b>360 MB</b>.<br/>"
        "&bull; <b>After Sensitivity-Aware Pruning + Lossless Compression:</b> Storage Package = <b>1.39 MB</b>, Runtime Model in Memory = <b>1.86 MB</b>, Accuracy = <b>97.80%</b>, Runtime Latency = <b>40.0 ms</b>.<br/>"
        "<i>Execute the complete 10-part mathematical derivation below to verify all precision steps, efficiency gains, and safety compliance.</i>",
        styles['Body']
    )
    t_prob = Table([[prob_p]], colWidths=[USABLE_WIDTH])
    t_prob.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 0.75, C_PRIMARY_BORDER),
        ('LINELEFT', (0,0), (-1,-1), 3, C_PRIMARY),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_prob)
    story.append(Spacer(1, 3))

    col_w = (USABLE_WIDTH - 6) / 2.0

    left_math = [
        Paragraph("<b>A. INT8 Quantization &amp; Reconstruction Math</b><br/>"
                  "Given input float <i>x = 0.80</i>, scale <i>S = 0.01</i>, zero point <i>Z = 0</i>:<br/>"
                  "&bull; <i>q = round(x / S) + Z = round(0.80 / 0.01) + 0 = round(80.0) =</i> <b>80</b><br/>"
                  "&bull; Reconstruct: <i>x&#770; = (q - Z) &times; S = (80 - 0) &times; 0.01 =</i> <b>0.80</b><br/>"
                  "&bull; Quantization Error: <i>&epsilon; = |x - x&#770;| = |0.80 - 0.80| =</i> <b>0.00</b> (exact representation).<br/>"
                  "<b>UAQE Meaning:</b> Quantizes continuous weights into discrete INT8 levels without loss when values match the step grid.", styles['TableCell']),
        
        Paragraph("<b>B. INT8 Convolution Arithmetic &amp; Accumulator Width</b><br/>"
                  "Given quantized input <i>x<sub>q</sub> = 20</i>, weight <i>w<sub>q</sub> = 3</i>, bias <i>b<sub>q</sub> = 5</i>:<br/>"
                  "&bull; <i>Accumulator = (x<sub>q</sub> &times; w<sub>q</sub>) + b<sub>q</sub> = (20 &times; 3) + 5 =</i> <b>65</b>.<br/>"
                  "&bull; <b>Why INT32 accumulation is mandatory:</b> An 8-bit product reaches 127 &times; 127 = 16,129. Accumulating over <i>C<sub>in</sub> = 512</i> channels with 3&times;3 kernels (4,608 MACs) can total &sim;7.4e+7, easily exceeding an INT16 limit (32,767). A 32-bit register (&plusmn;2.14e+9) guarantees zero overflow.<br/>"
                  "<b>UAQE Meaning:</b> SIMD vector engines safely accumulate integer dot-products before down-scaling.", styles['TableCell']),

        Paragraph("<b>C. Requantization Scale Multiplier</b><br/>"
                  "Given <i>S<sub>x</sub> = 0.02</i>, <i>S<sub>w</sub> = 0.01</i>, target output scale <i>S<sub>y</sub> = 0.05</i>:<br/>"
                  "&bull; <i>M = (S<sub>x</sub> &times; S<sub>w</sub>) / S<sub>y</sub> = (0.02 &times; 0.01) / 0.05 = 0.0002 / 0.05 =</i> <b>0.004</b><br/>"
                  "&bull; Output mapping: <i>y<sub>q</sub> = round(65 &times; 0.004) = round(0.26) =</i> <b>0</b>.<br/>"
                  "In hardware, <i>M</i> is mapped via integer fixed-point multiplication and bit-shift.<br/>"
                  "<b>UAQE Meaning:</b> Eliminates costly floating-point conversion at runtime by scaling directly into the next layer's INT8 format.", styles['TableCell']),

        Paragraph("<b>D. Accuracy Degradation Analysis</b><br/>"
                  "Baseline = 98.00%, Optimized = 97.80%:<br/>"
                  "&bull; <i>Accuracy Delta = Optimized - Baseline = 97.80% - 98.00% =</i> <b>-0.20 pp</b><br/>"
                  "&bull; <i>Accuracy Loss = Baseline - Optimized = 98.00% - 97.80% =</i> <b>0.20 pp</b><br/>"
                  "&bull; <b>UAQE Safety Classification:</b> Because 0.20 pp &le; 1.00 pp, this candidate is officially classified as <b>EXCELLENT</b>.<br/>"
                  "<b>UAQE Meaning:</b> Qualifies the candidate as fully compliant with production quality thresholds.", styles['TableCell']),

        Paragraph("<b>E. Model Size Reduction</b><br/>"
                  "&bull; Runtime Model: <i>((6.12 - 1.86) / 6.12) &times; 100 = (4.26 / 6.12) &times; 100 =</i> <b>69.61%</b><br/>"
                  "&bull; Compressed Storage: <i>((6.12 - 1.39) / 6.12) &times; 100 = (4.73 / 6.12) &times; 100 =</i> <b>77.29%</b><br/>"
                  "&bull; <b>Storage vs. Runtime Insight:</b> The 1.39 MB archive reduces distribution flash; the 1.86 MB uncompressed binary maps to active RAM.<br/>"
                  "<b>UAQE Meaning:</b> Distinguishes network OTA transport footprint from resident DRAM usage.", styles['TableCell'])
    ]

    right_math = [
        Paragraph("<b>F. Pure Inference Latency Speedup</b><br/>"
                  "Baseline Latency = 60.0 ms, Optimized Latency = 40.0 ms:<br/>"
                  "&bull; <i>Speedup % = ((60.0 - 40.0) / 60.0) &times; 100 = (20.0 / 60.0) &times; 100 =</i> <b>33.33%</b>.<br/>"
                  "The model processes batches 1.5&times; faster due to INT8 SIMD vector throughput and reduced memory cache pressure.<br/>"
                  "<b>UAQE Meaning:</b> Confirms that INT8 vector kernels successfully overcome tensor conversion overhead.", styles['TableCell']),

        Paragraph("<b>G. Inference Throughput (FPS)</b><br/>"
                  "Using standard formula <i>Throughput = 1000 / Latency<sub>ms</sub></i>:<br/>"
                  "&bull; FP32 Throughput: <i>1000 / 60.0 =</i> <b>16.67 FPS</b><br/>"
                  "&bull; INT8 Throughput: <i>1000 / 40.0 =</i> <b>25.00 FPS</b><br/>"
                  "&bull; Net Throughput Gain: <i>25.00 - 16.67 =</i> <b>+8.33 FPS (+50.0% gain)</b>.<br/>"
                  "<b>UAQE Meaning:</b> Allows real-time video stream processing at edge line rates.", styles['TableCell']),

        Paragraph("<b>H. Process RAM Reduction</b><br/>"
                  "FP32 Execution RAM = 500 MB, INT8 Execution RAM = 360 MB:<br/>"
                  "&bull; <i>RAM Reduction % = ((500 - 360) / 500) &times; 100 = (140 / 500) &times; 100 =</i> <b>28.00%</b>.<br/>"
                  "Frees 140 MB of system RAM for concurrency on constrained edge Linux boards.<br/>"
                  "<b>UAQE Meaning:</b> Prevents Out-Of-Memory (OOM) kernel kills on 1GB/2GB embedded devices.", styles['TableCell']),

        Paragraph("<b>I. Final UAQE Gating Decision</b><br/>"
                  "&bull; Accuracy Loss: 0.20 pp (&le; 1.00 pp &rarr; <b>EXCELLENT</b>)<br/>"
                  "&bull; Latency &amp; FPS: 33.33% speedup (+50% throughput)<br/>"
                  "&bull; Footprint: 69.61% runtime size cut, 28.00% RAM reduction<br/>"
                  "&bull; <b>Verdict:</b> <b>APPROVED AS PARETO WINNER.</b> The candidate strictly satisfies safety constraints and Pareto-dominates the baseline across all axes.<br/>"
                  "<b>UAQE Meaning:</b> Automatically triggers deployment packaging and artifact sealing.", styles['TableCell']),

        Paragraph("<b>J. Complete Pipeline Lifecycle Trace</b><br/>"
                  "Connects mathematical results across all 11 phases:<br/>"
                  "<i>UPLOAD &rarr; INSPECT &rarr; BASELINE (98%) &rarr; QUANTIZE &rarr; PRUNE &rarr; COMPRESS (1.39MB) &rarr; BENCHMARK (40ms) &rarr; VALIDATE &rarr; SAFETY (Pass) &rarr; PACKAGE &rarr; DEPLOY</i><br/>"
                  "<b>UAQE Meaning:</b> Illustrates the unified execution lifecycle converting raw models into validated edge solutions.", styles['TableCell'])
    ]

    math_grid_data = []
    for l_item, r_item in zip(left_math, right_math):
        math_grid_data.append([l_item, r_item])

    t_grid = Table(math_grid_data, colWidths=[col_w, col_w])
    t_grid.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#ffffff')),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_LINE),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_grid)
    story.append(Spacer(1, 3))

    story.append(callout_box(
        "CANONICAL SUMMARY",
        "<b>THIS ONE PROBLEM REPRESENTS THE ENTIRE CORE MATHEMATICAL AND LOGICAL WORKFLOW OF UAQE.</b> "
        "Every phase&mdash;from tensor discretization to SIMD accumulator sizing, fixed-point requantization, multi-metric benchmarking, "
        "safety gating, and deployment packaging&mdash;interlocks seamlessly to produce verified, production-ready edge AI models.",
        "remember"
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 8 -- FINAL MASTER CHEAT SHEET
    # =========================================================================
    story.append(Paragraph("<b>UAQE MASTER CHEAT SHEET -- 16 PHASES IN ONE VIEW</b>", styles['SectionHeader']))
    story.append(Paragraph("<b>High-Density Quick Revision, Formula Compilation &amp; Architectural Axioms for Viva Preparation</b>", styles['DocSubtitle']))
    story.append(Spacer(1, 2))

    grid_cells = [
        ("01. NN Fundamentals", "y = wx + b; MAC = 1 Mult + 1 Add = 2 FLOPs. NCHW vs NHWC layouts."),
        ("02. Model Storage", "Model = Graph + Weights + Meta. SafeTensors, ONNX QDQ, TFLite FlatBuffers."),
        ("03. Precision", "FP32(4B), FP16(2B), INT8(1B). 4x memory savings. Accumulator requires INT32."),
        ("04. Quant Math", "q = round(x/S)+Z; M = (Sx*Sw)/Sy. Outliers handled via per-channel scaling."),
        ("05. Calibration", "PTQ: Calibrate with 100-500 images. QAT: FakeQuant + Straight-Through Estimator."),
        ("06. Project Usage", "ResNet-50: ONNX QDQ PTQ. MobileNet: INT8 TFLite. Speedup depends on kernels."),
        ("07. Engine Arch", "11-stage pipeline; sandboxed job isolation; capability-driven model adapters."),
        ("08. INT8 Inference", "SIMD vector dot-products (ARM NEON SDOT, VNNI). L2/L3 cache residency."),
        ("09. Pruning & Sparsity", "Structured pruning yields speedup; unstructured yields 0% CPU gain. Sparsity!=Speed."),
        ("10. Artifacts", "Storage archive (1.39MB) != Runtime memory (1.86MB). SHA-256 provenance."),
        ("11. Benchmarking", "Pure inference != End-to-End latency. FPS = 1000/ms. Isolated telemetry."),
        ("12. Candidate Search", "Safety: <=1pp EXCELLENT, <=4pp ACCEPTABLE, >4pp CRITICAL. Pareto sorting."),
        ("13. Deployment", "4 Validation Levels: File -> Decompress -> Model -> Runtime inference pass."),
        ("14. Hardware Aware", "Host x86 = MEASURED. Raspberry Pi 5 = PENDING. ViT unsupported rejected."),
        ("15. Observability", "Real-time SSE events; process vs system CPU/RAM; zero synthetic metrics."),
        ("16. Trust Engineering", "Verification != Validation. 267/267 automated tests passing. Zero mock evidence.")
    ]

    grid_4x4_data = []
    row = []
    w4 = USABLE_WIDTH / 4.0
    for idx, (title, desc) in enumerate(grid_cells):
        cell_p = Paragraph(f"<b><font color='{C_PRIMARY.hexval()}'>{title}</font></b><br/>{desc}", styles['TableCell'])
        row.append(cell_p)
        if len(row) == 4:
            grid_4x4_data.append(row)
            row = []

    t_4x4 = Table(grid_4x4_data, colWidths=[w4]*4)
    t_4x4.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 0.5, C_PRIMARY_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_LINE),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_4x4)
    story.append(Spacer(1, 3))

    # MUST REMEMBER FORMULAS & PRINCIPLES (15 Core Items)
    story.append(Paragraph("<b>CORE FORMULAS &amp; ARCHITECTURAL AXIOMS (MUST REMEMBER)</b>", styles['SubHeader']))
    story.append(Spacer(1, 1))

    must_remember_items = [
        [
            Paragraph("<b>1. Neuron MAC:</b> <i>y = w &times; x + b</i>", styles['TableCell']),
            Paragraph("<b>6. Throughput:</b> <i>FPS = 1000 / Latency<sub>ms</sub></i>", styles['TableCell']),
            Paragraph("<b>11. Cardinal Rule:</b> Smaller &ne; Faster", styles['TableCell'])
        ],
        [
            Paragraph("<b>2. Quantization:</b> <i>q = round(x / S) + Z</i>", styles['TableCell']),
            Paragraph("<b>7. Speedup:</b> <i>((B - O) / B) &times; 100</i>", styles['TableCell']),
            Paragraph("<b>12. Scope:</b> Universal &ne; Every Architecture", styles['TableCell'])
        ],
        [
            Paragraph("<b>3. Dequantization:</b> <i>x&#770; = (q - Z) &times; S</i>", styles['TableCell']),
            Paragraph("<b>8. Accuracy Loss:</b> <i>Baseline - Optimized</i>", styles['TableCell']),
            Paragraph("<b>13. Honesty:</b> Host Measured &ne; RPi Pending", styles['TableCell'])
        ],
        [
            Paragraph("<b>4. Safe MAC:</b> <i>INT8 &times; INT8 &rarr; INT32 Acc</i>", styles['TableCell']),
            Paragraph("<b>9. Size Reduction:</b> <i>((Orig - Opt) / Orig) &times; 100</i>", styles['TableCell']),
            Paragraph("<b>14. Trust:</b> Zero Synthetic / Mock Metrics", styles['TableCell'])
        ],
        [
            Paragraph("<b>5. Requantization:</b> <i>M = (S<sub>x</sub> &times; S<sub>w</sub>) / S<sub>y</sub></i>", styles['TableCell']),
            Paragraph("<b>10. Philosophy:</b> Verification &ne; Validation", styles['TableCell']),
            Paragraph("<b>15. Safety:</b> CRITICAL Candidates Never Selected", styles['TableCell'])
        ],
    ]
    t_mr = Table(must_remember_items, colWidths=[USABLE_WIDTH/3.0]*3)
    t_mr.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#ffffff')),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_LINE),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_mr)
    story.append(Spacer(1, 3))

    # Final Complete Flow Pipeline
    story.append(Paragraph("<b>CANONICAL END-TO-END EXECUTION FLOW</b>", styles['SubHeader']))
    story.append(Spacer(1, 1))
    flow_final_1 = ["MODEL INGEST", "DATASET INGEST", "TARGET PROFILE", "GRAPH INSPECT", "VALIDATE BASELINE", "AUTO PLAN"]
    flow_final_2 = ["GENERATE CANDIDATES", "OPTIMIZE (INT8/Prune)", "BENCHMARK METRICS", "SAFETY GATING", "SELECT WINNER", "PACKAGE & DEPLOY"]
    story.append(flow_box(flow_final_1))
    story.append(Spacer(1, 1))
    story.append(flow_box(flow_final_2))
    story.append(Spacer(1, 3))

    # Final Project Definition Banner
    final_def_p = Paragraph(
        "<b>FINAL DEFINITIVE PROJECT FORMULATION:</b><br/>"
        "<i>\"UAQE is a capability-driven AI optimization platform that automatically explores supported optimization strategies, "
        "measures real accuracy and efficiency, enforces safety constraints, and produces a verified deployment artifact.\"</i>",
        styles['CalloutBody']
    )
    t_fdef = Table([[final_def_p]], colWidths=[USABLE_WIDTH])
    t_fdef.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_PRIMARY_LIGHT),
        ('BOX', (0,0), (-1,-1), 0.75, C_PRIMARY_BORDER),
        ('LINELEFT', (0,0), (-1,-1), 3, C_PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_fdef)

    # Build Document with Document Metadata
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=MARGIN_X,
        rightMargin=MARGIN_X,
        topMargin=MARGIN_TOP,
        bottomMargin=MARGIN_BOTTOM,
        title="Universal AI Quantization Engine (UAQE) - 16-Phase Technical Learning Guide",
        author="UAQE Systems & ML Engineering",
        subject="AI Quantization, Model Optimization, Embedded Edge Deployment"
    )

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF generated successfully: {filename}")


if __name__ == "__main__":
    out_pdf = "UAQE_16_Phase_Complete_Guide.pdf"
    build_pdf(out_pdf)
    
    with open(out_pdf, "rb") as f:
        pdf_bytes = f.read()
    page_count = len(re.findall(rb"/Type\s*/Page\b", pdf_bytes))
    print(f"Total Pages in {out_pdf}: {page_count}")
    if page_count == 8:
        print("PERFECT: Document is EXACTLY 8 pages!")
    else:
        print(f"WARNING: Page count is {page_count}, expected 8.")
