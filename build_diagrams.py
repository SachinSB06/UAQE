"""
build_diagrams.py
Generates 13 professional, high-resolution vector diagrams for the
Quantization Engine Technical Documentation using ReportLab Graphics.
Fully dynamic coordinate calculations to prevent overlap and text collisions.
"""

from reportlab.graphics.shapes import (
    Drawing, Rect, String, Line, Polygon, Group, Circle
)
from reportlab.lib import colors

# Color Palette
NAVY = colors.HexColor('#0f2b5c')       # Primary dark navy
DEEP_BLUE = colors.HexColor('#1e3a8a')  # Primary blue
BLUE = colors.HexColor('#2563eb')       # Secondary blue
CYAN = colors.HexColor('#0284c7')       # Accent cyan
TEAL = colors.HexColor('#0d9488')       # Accent teal
DARK_GRAY = colors.HexColor('#1e293b')   # Dark text / outline
SLATE = colors.HexColor('#334155')       # Neutral slate
LIGHT_BG = colors.HexColor('#f8fafc')    # Very light gray
CARD_BG = colors.HexColor('#ffffff')     # Pure white card
BORDER = colors.HexColor('#cbd5e1')      # Subtle border
AMBER = colors.HexColor('#b45309')       # Warning / note
GREEN = colors.HexColor('#047857')       # Success / validated


def draw_arrow(d, x1, y1, x2, y2, color=CYAN, stroke_width=1.5, head_size=5):
    """Draw a directed arrow from (x1, y1) to (x2, y2)."""
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=stroke_width))
    import math
    angle = math.atan2(y2 - y1, x2 - x1)
    sin_a = math.sin(angle)
    cos_a = math.cos(angle)
    p1_x = x2 - head_size * cos_a + (head_size * 0.5) * sin_a
    p1_y = y2 - head_size * sin_a - (head_size * 0.5) * cos_a
    p2_x = x2 - head_size * cos_a - (head_size * 0.5) * sin_a
    p2_y = y2 - head_size * sin_a + (head_size * 0.5) * cos_a
    arrowhead = Polygon([x2, y2, p1_x, p1_y, p2_x, p2_y],
                        fillColor=color, strokeColor=color, strokeWidth=0.5)
    d.add(arrowhead)


def draw_block(d, x, y, w, h, title, subtitle="", bg_color=CARD_BG, border_color=BORDER, text_color=DARK_GRAY, sub_color=SLATE, rx=4, ry=4):
    """
    Draw a rounded rectangular block with title and optional multiline subtitle.
    Splits both title and subtitle by newlines and distributes lines cleanly inside the box.
    """
    d.add(Rect(x, y, w, h, rx=rx, ry=ry, fillColor=bg_color, strokeColor=border_color, strokeWidth=1.2))
    
    title_lines = [line.strip() for line in title.split('\n') if line.strip()]
    sub_lines = [line.strip() for line in subtitle.split('\n') if line.strip()]
    
    # Choose base font size and line height based on number of lines and box height
    total_lines = len(title_lines) + len(sub_lines)
    if total_lines >= 4 or h < 45:
        title_font = 7.5
        title_lead = 9.0
        sub_font = 6.2
        sub_lead = 7.5
        gap = 2.5
    elif total_lines >= 3 or h < 55:
        title_font = 8.0
        title_lead = 9.5
        sub_font = 6.6
        sub_lead = 8.0
        gap = 3.0
    else:
        title_font = 8.5
        title_lead = 10.5
        sub_font = 7.0
        sub_lead = 8.5
        gap = 3.5

    total_h = len(title_lines) * title_lead + (gap if (title_lines and sub_lines) else 0) + len(sub_lines) * sub_lead
    # If text exceeds available height, scale font sizes down
    if total_h > (h - 4):
        scale = max(0.5, (h - 4) / max(total_h, 1))
        title_font = max(5.5, title_font * scale)
        title_lead = title_font + 1.2
        sub_font = max(5.0, sub_font * scale)
        sub_lead = sub_font + 1.0
        gap = max(1.0, gap * scale)
        total_h = len(title_lines) * title_lead + (gap if (title_lines and sub_lines) else 0) + len(sub_lines) * sub_lead

    # Center vertically inside the block
    top_y = y + (h + total_h) / 2.0
    curr_y = top_y - title_lead * 0.75
    
    for tl in title_lines:
        d.add(String(x + w / 2.0, curr_y, tl, textAnchor='middle', fontName='Helvetica-Bold', fontSize=title_font, fillColor=text_color))
        curr_y -= title_lead
        
    if title_lines and sub_lines:
        curr_y -= gap
        
    for sl in sub_lines:
        d.add(String(x + w / 2.0, curr_y, sl, textAnchor='middle', fontName='Helvetica', fontSize=sub_font, fillColor=sub_color))
        curr_y -= sub_lead


# =========================================================================
# DIAGRAM 1: Overall System Architecture
# =========================================================================
def create_diagram_1_overall_architecture(w=480, h=180):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, rx=6, ry=6, fillColor=LIGHT_BG, strokeColor=BORDER, strokeWidth=1))
    d.add(String(w / 2.0, h - 13, "OVERALL QUANTIZATION ENGINE SYSTEM ARCHITECTURE", textAnchor='middle', fontName='Helvetica-Bold', fontSize=9.5, fillColor=NAVY))
    
    bh = 34
    # Row 1 (y = 118, top = 152)
    r1_y = 118
    draw_block(d, 15, r1_y, 125, bh, "TRAINED AI MODEL", "FP32 PyTorch / ONNX", bg_color=CARD_BG, border_color=BLUE, text_color=NAVY)
    draw_arrow(d, 140, r1_y + bh/2, 170, r1_y + bh/2)
    
    draw_block(d, 170, r1_y, 135, bh, "DATASET & CALIB", "Representative Subset", bg_color=CARD_BG, border_color=CYAN, text_color=NAVY)
    draw_arrow(d, 305, r1_y + bh/2, 335, r1_y + bh/2)
    
    draw_block(d, 335, r1_y, 130, bh, "INT8 QUANTIZATION", "Scale & Zero-Point", bg_color=CARD_BG, border_color=DEEP_BLUE, text_color=NAVY)
    draw_arrow(d, 400, r1_y, 400, 100)
    
    # Row 2 (y = 66, top = 100)
    r2_y = 66
    draw_block(d, 335, r2_y, 130, bh, "FIXED-POINT CONV", "INT8 Ops / INT32 Accum", bg_color=CARD_BG, border_color=TEAL, text_color=NAVY)
    draw_arrow(d, 335, r2_y + bh/2, 305, r2_y + bh/2)
    
    draw_block(d, 170, r2_y, 135, bh, "MAGNITUDE PRUNING", "Sparsity Thresholding", bg_color=CARD_BG, border_color=AMBER, text_color=NAVY)
    draw_arrow(d, 170, r2_y + bh/2, 140, r2_y + bh/2)
    
    draw_block(d, 15, r2_y, 125, bh, "RLE COMPRESSION", "Bitmask & Run Encoding", bg_color=CARD_BG, border_color=CYAN, text_color=NAVY)
    draw_arrow(d, 77, r2_y, 77, 48)
    
    # Row 3 (y = 14, top = 48)
    r3_y = 14
    draw_block(d, 15, r3_y, 125, bh, "MEMORY EXPORT", ".mem | .hex | .bin", bg_color=colors.HexColor('#eff6ff'), border_color=BLUE, text_color=NAVY)
    draw_arrow(d, 140, r3_y + bh/2, 170, r3_y + bh/2)
    
    draw_block(d, 170, r3_y, 155, bh, "HARDWARE DEPLOY", "Artix-7 FPGA / RPi 5", bg_color=CARD_BG, border_color=TEAL, text_color=NAVY)
    draw_arrow(d, 325, r3_y + bh/2, 355, r3_y + bh/2)
    
    draw_block(d, 355, r3_y, 110, bh, "PERFORMANCE", "Telemetry & Report", bg_color=colors.HexColor('#f0fdf4'), border_color=GREEN, text_color=GREEN)
    
    return d


# =========================================================================
# DIAGRAM 2: Quantization Pipeline (12 Steps)
# =========================================================================
def create_diagram_2_quantization_pipeline(w=480, h=175):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, rx=6, ry=6, fillColor=LIGHT_BG, strokeColor=BORDER, strokeWidth=1))
    d.add(String(w / 2.0, h - 13, "END-TO-END QUANTIZATION & DEPLOYMENT PIPELINE", textAnchor='middle', fontName='Helvetica-Bold', fontSize=9.5, fillColor=NAVY))
    
    steps = [
        ("1. Input Model", "PyTorch/ONNX"),
        ("2. Model Ingest", "Shape & Dtype"),
        ("3. FP32 Eval", "Baseline Acc"),
        ("4. Calibration", "Histogram/MinMax"),
        ("5. INT8 Quant", "Uniform Affine"),
        ("6. INT8 Eval", "Accuracy Guard"),
        ("7. Fixed-Point", "INT32 Accumulator"),
        ("8. Magnitude Prune", "Sparse Matrix"),
        ("9. RLE Encoding", "Zero Compression"),
        ("10. Mem Gen", ".mem / .hex / .bin"),
        ("11. RTL Sim", "Testbench Check"),
        ("12. Edge Deploy", "RPi5 / Artix-7"),
    ]
    
    bw, bh = 100, 32
    xs = [15, 130, 245, 360]
    ys = [114, 64, 14]
    
    # Row 0: Left to Right
    for i in range(4):
        draw_block(d, xs[i], ys[0], bw, bh, steps[i][0], steps[i][1], bg_color=CARD_BG, border_color=BLUE, text_color=NAVY)
        if i < 3:
            draw_arrow(d, xs[i] + bw, ys[0] + bh/2, xs[i+1], ys[0] + bh/2)
    draw_arrow(d, xs[3] + bw/2, ys[0], xs[3] + bw/2, ys[1] + bh)
    
    # Row 1: Right to Left
    for idx, col in enumerate(reversed(range(4))):
        step_idx = 4 + idx
        draw_block(d, xs[col], ys[1], bw, bh, steps[step_idx][0], steps[step_idx][1], bg_color=CARD_BG, border_color=CYAN, text_color=NAVY)
        if col > 0:
            draw_arrow(d, xs[col], ys[1] + bh/2, xs[col-1] + bw, ys[1] + bh/2)
    draw_arrow(d, xs[0] + bw/2, ys[1], xs[0] + bw/2, ys[2] + bh)
    
    # Row 2: Left to Right
    for idx, col in enumerate(range(4)):
        step_idx = 8 + idx
        border_c = GREEN if step_idx == 11 else TEAL
        draw_block(d, xs[col], ys[2], bw, bh, steps[step_idx][0], steps[step_idx][1], bg_color=CARD_BG, border_color=border_c, text_color=NAVY)
        if col < 3:
            draw_arrow(d, xs[col] + bw, ys[2] + bh/2, xs[col+1], ys[2] + bh/2)
            
    return d


# =========================================================================
# DIAGRAM 3: FP32 to INT8 Mapping (FIGURE 6.1)
# =========================================================================
def create_diagram_3_fp32_to_int8(w=480, h=130):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, rx=6, ry=6, fillColor=LIGHT_BG, strokeColor=BORDER, strokeWidth=1))
    d.add(String(w / 2.0, h - 14, "NUMERICAL PRECISION TRANSFORMATION: FP32 TO INT8", textAnchor='middle', fontName='Helvetica-Bold', fontSize=9.5, fillColor=NAVY))
    
    bh = 78
    by = 24
    
    # Box 1: FP32 Format (Left)
    bw1 = 132
    bx1 = 14
    sub1 = "IEEE-754 Single Precision\n32 Bits per Weight\nRange: [-3.4e38, 3.4e38]\nDynamic Storage: 4 Bytes"
    draw_block(d, bx1, by, bw1, bh, "FP32 FORMAT", sub1, bg_color=CARD_BG, border_color=BLUE, text_color=NAVY)
    
    # Box 2: Quantization Transformation (Middle)
    bw2 = 140
    bx2 = 170
    sub2 = "q = round( x / S ) + Z\nS = (β − α) / 255\nx_approx = (q − Z) · S"
    draw_block(d, bx2, by, bw2, bh, "QUANTIZATION", sub2, bg_color=colors.HexColor('#eff6ff'), border_color=CYAN, text_color=NAVY)
    
    # Box 3: INT8 Format (Right)
    bw3 = 132
    bx3 = 334
    sub3 = "Signed 8-Bit Two's Comp\n8 Bits per Weight\nRange: [-128, +127]\nDynamic Storage: 1 Byte"
    draw_block(d, bx3, by, bw3, bh, "INT8 FORMAT", sub3, bg_color=colors.HexColor('#f0fdf4'), border_color=GREEN, text_color=GREEN)
    
    # Connecting Arrows
    draw_arrow(d, bx1 + bw1, by + bh/2, bx2, by + bh/2, color=CYAN, stroke_width=1.5, head_size=5)
    draw_arrow(d, bx2 + bw2, by + bh/2, bx3, by + bh/2, color=CYAN, stroke_width=1.5, head_size=5)
    
    # Bottom Callout Text
    d.add(String(w / 2.0, 10, "Storage Footprint Reduced by Exactly 75.0% (4x Memory Compression Factor)", textAnchor='middle', fontName='Helvetica-Bold', fontSize=8.0, fillColor=DEEP_BLUE))
    return d


# =========================================================================
# DIAGRAM 4: Calibration Flow
# =========================================================================
def create_diagram_4_calibration_flow(w=480, h=140):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, rx=6, ry=6, fillColor=LIGHT_BG, strokeColor=BORDER, strokeWidth=1))
    d.add(String(w / 2.0, h - 14, "CALIBRATION AND ACTIVATION RANGE OBSERVER PIPELINE", textAnchor='middle', fontName='Helvetica-Bold', fontSize=9.5, fillColor=NAVY))
    
    bw, bh = 100, 68
    by = 30
    draw_block(d, 14, by, bw, bh, "REPRESENTATIVE\nDATASET", "100-500 Unlabeled\nDomain Images", bg_color=CARD_BG, border_color=BLUE, text_color=NAVY)
    draw_arrow(d, 114, by + bh/2, 132, by + bh/2)
    
    draw_block(d, 132, by, bw, bh, "FORWARD\nPROPAGATION", "FP32 Activation\nTensor Tapping", bg_color=CARD_BG, border_color=CYAN, text_color=NAVY)
    draw_arrow(d, 232, by + bh/2, 250, by + bh/2)
    
    draw_block(d, 250, by, bw, bh, "OBSERVER\nANALYSIS", "Histogram / MinMax\nKL Divergence", bg_color=CARD_BG, border_color=TEAL, text_color=NAVY)
    draw_arrow(d, 350, by + bh/2, 368, by + bh/2)
    
    draw_block(d, 368, by, bw, bh, "QUANTIZATION\nPARAMETERS", "Scale (S) & Offset (Z)\nPer-Tensor Clipping", bg_color=colors.HexColor('#f0fdf4'), border_color=GREEN, text_color=GREEN)
    
    d.add(String(w / 2.0, 11, "Calibration establishes activation boundaries to eliminate overflow without label supervision.", textAnchor='middle', fontName='Helvetica-Oblique', fontSize=7.5, fillColor=SLATE))
    return d


# =========================================================================
# DIAGRAM 5: PTQ vs QAT Comparison
# =========================================================================
def create_diagram_5_ptq_vs_qat(w=480, h=140):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, rx=6, ry=6, fillColor=LIGHT_BG, strokeColor=BORDER, strokeWidth=1))
    d.add(String(w / 2.0, h - 13, "COMPARISON OF QUANTIZATION PARADIGMS: PTQ VS QAT", textAnchor='middle', fontName='Helvetica-Bold', fontSize=9.5, fillColor=NAVY))
    
    card_h = 106
    card_y = 12
    card_w = 220
    
    # Left Card: PTQ
    d.add(Rect(14, card_y, card_w, card_h, rx=4, ry=4, fillColor=CARD_BG, strokeColor=BLUE, strokeWidth=1))
    d.add(String(124, card_y + card_h - 13, "Post-Training Quantization (PTQ)", textAnchor='middle', fontName='Helvetica-Bold', fontSize=8.0, fillColor=NAVY))
    ptq_blocks = [
        (24, card_y + 64, 200, 20, "1. Pretrained FP32 Weights"),
        (24, card_y + 38, 200, 20, "2. Calibration Pass (No Retraining)"),
        (24, card_y + 12, 200, 20, "3. Discretize Weights & Activations"),
    ]
    for bx, by, bw, bh, text in ptq_blocks:
        d.add(Rect(bx, by, bw, bh, rx=3, ry=3, fillColor=colors.HexColor('#eff6ff'), strokeColor=BORDER, strokeWidth=0.8))
        d.add(String(bx + bw/2, by + 6, text, textAnchor='middle', fontName='Helvetica', fontSize=7.0, fillColor=DARK_GRAY))
    draw_arrow(d, 124, card_y + 64, 124, card_y + 58, head_size=4)
    draw_arrow(d, 124, card_y + 38, 124, card_y + 32, head_size=4)
    
    # Right Card: QAT
    d.add(Rect(246, card_y, card_w, card_h, rx=4, ry=4, fillColor=CARD_BG, strokeColor=TEAL, strokeWidth=1))
    d.add(String(356, card_y + card_h - 13, "Quantization-Aware Training (QAT)", textAnchor='middle', fontName='Helvetica-Bold', fontSize=8.0, fillColor=NAVY))
    qat_blocks = [
        (256, card_y + 64, 200, 20, "1. Insert FakeQuant Simulation Nodes"),
        (256, card_y + 38, 200, 20, "2. Fine-tune with Straight-Through (STE)"),
        (256, card_y + 12, 200, 20, "3. Deploy True INT8 with Learned Weights"),
    ]
    for bx, by, bw, bh, text in qat_blocks:
        d.add(Rect(bx, by, bw, bh, rx=3, ry=3, fillColor=colors.HexColor('#f0fdf4'), strokeColor=BORDER, strokeWidth=0.8))
        d.add(String(bx + bw/2, by + 6, text, textAnchor='middle', fontName='Helvetica', fontSize=7.0, fillColor=DARK_GRAY))
    draw_arrow(d, 356, card_y + 64, 356, card_y + 58, head_size=4)
    draw_arrow(d, 356, card_y + 38, 356, card_y + 32, head_size=4)
    
    return d


# =========================================================================
# DIAGRAM 6: Fixed-Point Computation & INT32 Accumulation
# =========================================================================
def create_diagram_6_fixed_point(w=480, h=140):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, rx=6, ry=6, fillColor=LIGHT_BG, strokeColor=BORDER, strokeWidth=1))
    d.add(String(w / 2.0, h - 13, "FIXED-POINT ARITHMETIC & INT32 OVERFLOW PROTECTION", textAnchor='middle', fontName='Helvetica-Bold', fontSize=9.5, fillColor=NAVY))
    
    draw_block(d, 15, 78, 95, 32, "INT8 ACTIVATION", "8-Bit Operand X", bg_color=CARD_BG, border_color=BLUE, text_color=NAVY)
    draw_block(d, 15, 32, 95, 32, "INT8 WEIGHT", "8-Bit Operand W", bg_color=CARD_BG, border_color=BLUE, text_color=NAVY)
    
    draw_block(d, 138, 48, 75, 42, "MAC MULT", "8b x 8b -> 16b", bg_color=colors.HexColor('#eff6ff'), border_color=CYAN, text_color=NAVY)
    draw_arrow(d, 110, 94, 138, 75)
    draw_arrow(d, 110, 48, 138, 65)
    
    draw_block(d, 238, 42, 115, 52, "INT32 ACCUM", "32-Bit Summation\nPrevents Overflow\nUp to 65,536 MACs", bg_color=CARD_BG, border_color=AMBER, text_color=NAVY)
    draw_arrow(d, 213, 69, 238, 69)
    
    draw_block(d, 378, 48, 85, 42, "INT8 OUT", "Scale & Sat\n[-128, 127]", bg_color=colors.HexColor('#f0fdf4'), border_color=GREEN, text_color=GREEN)
    draw_arrow(d, 353, 69, 378, 69)
    
    d.add(String(w / 2.0, 11, "Bit Growth: 8b x 8b = 16b Product; 32b accumulator provides 16 guard bits for N=65,536 additions.", textAnchor='middle', fontName='Helvetica', fontSize=7.0, fillColor=SLATE))
    return d


# =========================================================================
# DIAGRAM 7: Compression Pipeline
# =========================================================================
def create_diagram_7_compression_pipeline(w=480, h=110):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, rx=6, ry=6, fillColor=LIGHT_BG, strokeColor=BORDER, strokeWidth=1))
    d.add(String(w / 2.0, h - 13, "TWO-STAGE COMPRESSION ENGINE: PRUNING + RUN-LENGTH ENCODING", textAnchor='middle', fontName='Helvetica-Bold', fontSize=9.5, fillColor=NAVY))
    
    bw, bh = 98, 46
    by = 28
    xs = [14, 130, 246, 362]
    
    draw_block(d, xs[0], by, bw, bh, "INT8 WEIGHTS", "Dense Tensor Array", bg_color=CARD_BG, border_color=BLUE, text_color=NAVY)
    draw_arrow(d, xs[0] + bw, by + bh/2, xs[1], by + bh/2)
    
    draw_block(d, xs[1], by, bw, bh, "MAGNITUDE PRUNE", "|w| < τ → 0\n20%-30% Sparsity", bg_color=CARD_BG, border_color=AMBER, text_color=NAVY)
    draw_arrow(d, xs[1] + bw, by + bh/2, xs[2], by + bh/2)
    
    draw_block(d, xs[2], by, bw, bh, "SPARSE BITMASK", "1-Bit Presence Map\nContiguous Non-Zeros", bg_color=CARD_BG, border_color=CYAN, text_color=NAVY)
    draw_arrow(d, xs[2] + bw, by + bh/2, xs[3], by + bh/2)
    
    draw_block(d, xs[3], by, bw, bh, "RLE ENCODER", "Run of Zeros Encoded\nUp to 32.9% Net Gain", bg_color=colors.HexColor('#f0fdf4'), border_color=GREEN, text_color=GREEN)
    
    d.add(String(w / 2.0, 10, "Bit-for-Bit Lossless Reconstruction: Verified MAE = 0.0000 across all 84 weight tensors.", textAnchor='middle', fontName='Helvetica-Bold', fontSize=7.5, fillColor=DEEP_BLUE))
    return d


# =========================================================================
# DIAGRAM 8: Run-Length Encoding Example
# =========================================================================
def create_diagram_8_rle_example(w=480, h=100):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, rx=6, ry=6, fillColor=LIGHT_BG, strokeColor=BORDER, strokeWidth=1))
    d.add(String(w / 2.0, h - 13, "WORKED RUN-LENGTH ENCODING (RLE) SPARSITY COMPRESSION", textAnchor='middle', fontName='Helvetica-Bold', fontSize=9.5, fillColor=NAVY))
    
    # Uncompressed Stream
    d.add(String(20, 68, "Uncompressed Stream (10 Bytes):", fontName='Helvetica-Bold', fontSize=7.5, fillColor=NAVY))
    cells_raw = ["+14", "0", "0", "0", "0", "-25", "0", "0", "+8", "+3"]
    cw = 42
    for i, val in enumerate(cells_raw):
        bg = colors.HexColor('#fee2e2') if val != "0" else colors.HexColor('#f1f5f9')
        d.add(Rect(20 + i * cw, 48, cw - 4, 16, rx=2, ry=2, fillColor=bg, strokeColor=BORDER, strokeWidth=0.8))
        d.add(String(20 + i * cw + (cw-4)/2, 53, val, textAnchor='middle', fontName='Helvetica-Bold', fontSize=7.5, fillColor=DARK_GRAY))
        
    # RLE Stream
    d.add(String(20, 31, "RLE Encoded Stream (Zero-Run Escape Representation - 6 Bytes):", fontName='Helvetica-Bold', fontSize=7.5, fillColor=NAVY))
    cells_rle = [("+14", "Val"), ("ESC, 4", "4 Zeros"), ("-25", "Val"), ("ESC, 2", "2 Zeros"), ("+8", "Val"), ("+3", "Val")]
    rw = 70
    for i, (val, desc) in enumerate(cells_rle):
        d.add(Rect(20 + i * rw, 10, rw - 5, 17, rx=2, ry=2, fillColor=colors.HexColor('#f0fdf4'), strokeColor=GREEN, strokeWidth=0.8))
        d.add(String(20 + i * rw + (rw-5)/2, 19, val, textAnchor='middle', fontName='Helvetica-Bold', fontSize=7.0, fillColor=GREEN))
        d.add(String(20 + i * rw + (rw-5)/2, 12, desc, textAnchor='middle', fontName='Helvetica', fontSize=5.8, fillColor=SLATE))
        
    return d


# =========================================================================
# DIAGRAM 9: Hardware Memory Generation
# =========================================================================
def create_diagram_9_memory_generation(w=480, h=115):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, rx=6, ry=6, fillColor=LIGHT_BG, strokeColor=BORDER, strokeWidth=1))
    d.add(String(w / 2.0, h - 13, "HARDWARE MEMORY INITIALIZATION ARTIFACT GENERATION", textAnchor='middle', fontName='Helvetica-Bold', fontSize=9.5, fillColor=NAVY))
    
    # Left Block
    draw_block(d, 15, 20, 115, 68, "QUANTIZED & PRUNED\nMODEL PARAMETERS", "Serialized FlatBuffer", bg_color=CARD_BG, border_color=BLUE, text_color=NAVY)
    
    # Center Generator Block
    draw_block(d, 160, 20, 85, 68, "MEMORY\nGENERATOR", "Format Exporter\nFactory", bg_color=colors.HexColor('#eff6ff'), border_color=CYAN, text_color=NAVY)
    draw_arrow(d, 130, 54, 160, 54)
    
    # 3 Target Output Formats
    draw_block(d, 275, 66, 190, 24, ".MEM ($readmemh)", "Verilog/VHDL Block RAM Words", bg_color=CARD_BG, border_color=TEAL, text_color=NAVY)
    draw_arrow(d, 245, 54, 275, 78)
    
    draw_block(d, 275, 38, 190, 24, ".HEX (Intel HEX)", "MCU Flash & EEPROM Format", bg_color=CARD_BG, border_color=BLUE, text_color=NAVY)
    draw_arrow(d, 245, 54, 275, 50)
    
    draw_block(d, 275, 10, 190, 24, ".BIN (Raw Binary)", "DMA Transfer & DRAM Blob", bg_color=CARD_BG, border_color=GREEN, text_color=GREEN)
    draw_arrow(d, 245, 54, 275, 22)
    
    return d


# =========================================================================
# DIAGRAM 10: Hardware Architecture (SystemVerilog RTL)
# =========================================================================
def create_diagram_10_hardware_architecture(w=480, h=160):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, rx=6, ry=6, fillColor=LIGHT_BG, strokeColor=BORDER, strokeWidth=1))
    d.add(String(w / 2.0, h - 13, "SYSTEMVERILOG HARDWARE ACCELERATOR CORE ARCHITECTURE", textAnchor='middle', fontName='Helvetica-Bold', fontSize=9.5, fillColor=NAVY))
    
    # Top Control Unit
    draw_block(d, 165, 114, 150, 26, "CONTROL UNIT (FSM)", "State Machine & Handshake", bg_color=CARD_BG, border_color=BLUE, text_color=NAVY)
    
    # Address Generator
    draw_block(d, 165, 76, 150, 24, "ADDRESS GENERATOR", "BRAM Address Sequencer", bg_color=CARD_BG, border_color=CYAN, text_color=NAVY)
    draw_arrow(d, 240, 114, 240, 100)
    
    # Left & Right Memory BRAMs
    draw_block(d, 15, 68, 120, 36, "WEIGHT BRAM", "Quantized INT8 Weights\nInitialized via .mem", bg_color=CARD_BG, border_color=TEAL, text_color=NAVY)
    draw_block(d, 345, 68, 120, 36, "INPUT BRAM", "8-Bit Image Activations\nDual-Port Block RAM", bg_color=CARD_BG, border_color=TEAL, text_color=NAVY)
    draw_arrow(d, 165, 88, 135, 88)
    draw_arrow(d, 315, 88, 345, 88)
    
    # MAC Engine
    draw_block(d, 165, 40, 150, 26, "MAC ENGINE", "8b x 8b Signed Multiplier", bg_color=colors.HexColor('#eff6ff'), border_color=AMBER, text_color=NAVY)
    draw_arrow(d, 75, 68, 165, 53)
    draw_arrow(d, 405, 68, 315, 53)
    
    # Bottom Row: Accumulator -> ReLU -> Output
    draw_block(d, 15, 10, 130, 22, "INT32 ACCUMULATOR", "32-Bit Register Sum", bg_color=CARD_BG, border_color=DEEP_BLUE, text_color=NAVY)
    draw_arrow(d, 165, 45, 120, 32)
    
    draw_block(d, 165, 10, 130, 22, "ReLU ACTIVATION", "max(0, x) Piecewise", bg_color=CARD_BG, border_color=CYAN, text_color=NAVY)
    draw_arrow(d, 145, 21, 165, 21)
    
    draw_block(d, 315, 10, 150, 22, "OUTPUT BUFFER", "Result Valid Handshake", bg_color=colors.HexColor('#f0fdf4'), border_color=GREEN, text_color=GREEN)
    draw_arrow(d, 295, 21, 315, 21)
    
    return d


# =========================================================================
# DIAGRAM 11: Software Architecture
# =========================================================================
def create_diagram_11_software_architecture(w=480, h=170):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, rx=6, ry=6, fillColor=LIGHT_BG, strokeColor=BORDER, strokeWidth=1))
    d.add(String(w / 2.0, h - 13, "MODULAR SOFTWARE SUBSYSTEM HIERARCHY", textAnchor='middle', fontName='Helvetica-Bold', fontSize=9.5, fillColor=NAVY))
    
    draw_block(d, 135, 126, 210, 24, "CLI & REST API ENTRYPOINTS", "main.py | uaqe.py | FastAPI Server", bg_color=CARD_BG, border_color=BLUE, text_color=NAVY)
    
    bw, bh = 100, 44
    xs = [15, 130, 245, 360]
    y_mid = 62
    draw_block(d, xs[0], y_mid, bw, bh, "INGESTION", "Universal Ingestor\nModel & Dataset Adapters", bg_color=CARD_BG, border_color=CYAN, text_color=NAVY)
    draw_block(d, xs[1], y_mid, bw, bh, "QUANTIZATION", "INT8 / Mixed Precision\nCalibration & Sensitivity", bg_color=CARD_BG, border_color=DEEP_BLUE, text_color=NAVY)
    draw_block(d, xs[2], y_mid, bw, bh, "COMPRESSION", "Sensitivity Pruning\nRLE & Clustering", bg_color=CARD_BG, border_color=AMBER, text_color=NAVY)
    draw_block(d, xs[3], y_mid, bw, bh, "EXPORTERS", ".mem / .hex / .bin\nONNX & TFLite", bg_color=CARD_BG, border_color=GREEN, text_color=GREEN)
    
    draw_arrow(d, 180, 126, xs[0] + bw/2, y_mid + bh)
    draw_arrow(d, 210, 126, xs[1] + bw/2, y_mid + bh)
    draw_arrow(d, 270, 126, xs[2] + bw/2, y_mid + bh)
    draw_arrow(d, 300, 126, xs[3] + bw/2, y_mid + bh)
    
    draw_block(d, 15, 12, 445, 32, "COMMON REPOSITORIES, TELEMETRY & HARDWARE PROFILES", "HardwareTargetRegistry | Benchmark | ProcessMetrics | BaselineGuard", bg_color=colors.HexColor('#eff6ff'), border_color=BORDER, text_color=NAVY)
    
    return d


# =========================================================================
# DIAGRAM 12: End-to-End Flowchart
# =========================================================================
def create_diagram_12_flowchart(w=480, h=220):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, rx=6, ry=6, fillColor=LIGHT_BG, strokeColor=BORDER, strokeWidth=1))
    d.add(String(w / 2.0, h - 13, "END-TO-END EXECUTION WORKFLOW FLOWCHART", textAnchor='middle', fontName='Helvetica-Bold', fontSize=9.5, fillColor=NAVY))
    
    # Start button
    d.add(Rect(20, 168, 55, 22, rx=11, ry=11, fillColor=GREEN, strokeColor=GREEN))
    d.add(String(47, 175, "START", textAnchor='middle', fontName='Helvetica-Bold', fontSize=7.5, fillColor=colors.white))
    draw_arrow(d, 75, 179, 95, 179)
    
    draw_block(d, 95, 165, 105, 28, "Load Config & Data", "Parse JSON & Normalize", bg_color=CARD_BG, border_color=BLUE, text_color=NAVY)
    draw_arrow(d, 200, 179, 220, 179)
    
    draw_block(d, 220, 165, 95, 28, "FP32 Evaluation", "Measure Baseline Acc", bg_color=CARD_BG, border_color=BLUE, text_color=NAVY)
    draw_arrow(d, 315, 179, 335, 179)
    
    cx, cy = 385, 179
    diamond = Polygon([cx, cy + 15, cx + 45, cy, cx, cy - 15, cx - 45, cy],
                      fillColor=colors.HexColor('#fef3c7'), strokeColor=AMBER, strokeWidth=1)
    d.add(diamond)
    d.add(String(cx, cy - 3, "Baseline Valid?", textAnchor='middle', fontName='Helvetica-Bold', fontSize=6.5, fillColor=NAVY))
    
    # NO -> HALT
    draw_arrow(d, cx + 45, cy, 450, cy)
    d.add(String(455, cy - 3, "NO", fontName='Helvetica-Bold', fontSize=6.5, fillColor=colors.HexColor('#b91c1c')))
    d.add(Rect(438, 122, 28, 18, rx=3, ry=3, fillColor=colors.HexColor('#fee2e2'), strokeColor=colors.HexColor('#b91c1c'), strokeWidth=0.8))
    d.add(String(452, 128, "HALT", textAnchor='middle', fontName='Helvetica-Bold', fontSize=6.0, fillColor=colors.HexColor('#b91c1c')))
    draw_arrow(d, 452, cy - 8, 452, 140)
    
    # YES -> Down
    draw_arrow(d, cx, cy - 15, cx, 130)
    d.add(String(cx + 4, 140, "YES", fontName='Helvetica-Bold', fontSize=6.5, fillColor=GREEN))
    
    draw_block(d, 330, 95, 110, 28, "Calibration Phase", "Observer Collection", bg_color=CARD_BG, border_color=CYAN, text_color=NAVY)
    draw_arrow(d, 330, 109, 300, 109)
    
    draw_block(d, 190, 95, 110, 28, "INT8 Discretization", "Compute Scales & Zero-Pts", bg_color=CARD_BG, border_color=DEEP_BLUE, text_color=NAVY)
    draw_arrow(d, 190, 109, 160, 109)
    
    draw_block(d, 40, 95, 120, 28, "Fixed-Point Mapping", "INT32 Accumulator Scaling", bg_color=CARD_BG, border_color=TEAL, text_color=NAVY)
    draw_arrow(d, 100, 95, 100, 68)
    
    draw_block(d, 30, 40, 120, 28, "Pruning & RLE", "Zero-Weight Compression", bg_color=CARD_BG, border_color=AMBER, text_color=NAVY)
    draw_arrow(d, 150, 54, 175, 54)
    
    draw_block(d, 175, 40, 110, 28, "Memory Gen", ".mem | .hex | .bin Export", bg_color=CARD_BG, border_color=BLUE, text_color=NAVY)
    draw_arrow(d, 285, 54, 310, 54)
    
    draw_block(d, 310, 40, 85, 28, "Telemetry", "Accuracy & Latency", bg_color=CARD_BG, border_color=GREEN, text_color=GREEN)
    draw_arrow(d, 395, 54, 420, 54)
    
    d.add(Rect(420, 43, 42, 22, rx=11, ry=11, fillColor=NAVY, strokeColor=NAVY))
    d.add(String(441, 50, "END", textAnchor='middle', fontName='Helvetica-Bold', fontSize=7.5, fillColor=colors.white))
    
    return d


# =========================================================================
# DIAGRAM 13: Raspberry Pi 5 Prototype Architecture
# =========================================================================
def create_diagram_13_raspberry_pi(w=480, h=120):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, rx=6, ry=6, fillColor=LIGHT_BG, strokeColor=BORDER, strokeWidth=1))
    d.add(String(w / 2.0, h - 13, "RASPBERRY PI 5 EMBEDDED EDGE DEPLOYMENT ARCHITECTURE", textAnchor='middle', fontName='Helvetica-Bold', fontSize=9.5, fillColor=NAVY))
    
    # Left Block: Artifact
    draw_block(d, 14, 22, 112, 68, "QUANTIZED ARTIFACT", "INT8 FlatBuffer (.tflite)\nor ONNX Runtime\n(Size: 1.77 - 2.17 MB)", bg_color=CARD_BG, border_color=BLUE, text_color=NAVY)
    draw_arrow(d, 126, 56, 148, 56)
    
    # Center Host Card
    d.add(Rect(148, 20, 175, 72, rx=5, ry=5, fillColor=colors.HexColor('#eff6ff'), strokeColor=DEEP_BLUE, strokeWidth=1.2))
    d.add(String(235, 79, "Raspberry Pi 5 Edge Host", textAnchor='middle', fontName='Helvetica-Bold', fontSize=8.0, fillColor=NAVY))
    
    draw_block(d, 156, 48, 158, 22, "Inference Runtime", "TFLite with XNNPACK", bg_color=CARD_BG, border_color=TEAL, text_color=NAVY)
    draw_block(d, 156, 24, 158, 22, "Hardware Telemetry", "CPU % | RAM RSS | Temp C", bg_color=CARD_BG, border_color=CYAN, text_color=NAVY)
    
    # Connecting Arrows to Right Outputs
    draw_arrow(d, 323, 59, 348, 59)
    draw_arrow(d, 323, 35, 348, 35)
    
    # Right Column Blocks
    draw_block(d, 348, 48, 118, 22, "Predictions", "Class & Confidence", bg_color=colors.HexColor('#f0fdf4'), border_color=GREEN, text_color=GREEN)
    draw_block(d, 348, 24, 118, 22, "Performance Log", "Mean: 2.31 ms / 433 FPS", bg_color=CARD_BG, border_color=BORDER, text_color=DARK_GRAY)
    
    d.add(String(w / 2.0, 8, "Broadcom BCM2712 (4x Cortex-A76 @ 2.4 GHz) with 8GB LPDDR4X executing single-threaded edge evaluation.", textAnchor='middle', fontName='Helvetica', fontSize=6.8, fillColor=SLATE))
    return d
