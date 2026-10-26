import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import os

def generate_exact_vector_figure1():
    # Canvas: 15.0 x 10.8 inches at 300 DPI
    fig = plt.figure(figsize=(15.0, 10.8), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 15.0)
    ax.set_ylim(0, 10.8)
    ax.axis('off')
    fig.patch.set_facecolor('#FFFFFF')

    # Color Palette
    c_p1_border, c_p1_bg, c_p1_txt = '#1D4ED8', '#EFF6FF', '#1E40AF'
    c_p2_border, c_p2_bg, c_p2_txt = '#0D9488', '#F0FDFA', '#0F766E'
    c_p3_border, c_p3_bg, c_p3_txt = '#7C3AED', '#FAF5FF', '#5B21B6'

    # Phase 1: y = 6.10 to 10.35 (h = 4.25)
    ax.add_patch(FancyBboxPatch((0.40, 6.10), 4.10, 4.25, boxstyle="round,pad=0.12",
                                edgecolor=c_p1_border, facecolor=c_p1_bg, linewidth=1.8, linestyle='--'))
    ax.text(0.65, 9.98, "PHASE 1: OFFLINE TRAINING", fontsize=12.8, fontweight='bold', color=c_p1_txt)

    # Phase 2: y = 6.10 to 10.35 (h = 4.25)
    ax.add_patch(FancyBboxPatch((4.80, 6.10), 9.80, 4.25, boxstyle="round,pad=0.12",
                                edgecolor=c_p2_border, facecolor=c_p2_bg, linewidth=1.8, linestyle='--'))
    ax.text(5.05, 9.98, "PHASE 2: VALIDATION & CALIBRATION", fontsize=12.8, fontweight='bold', color=c_p2_txt)

    # --- PRODUCTION DEPLOYMENT BUS: y = 4.85 to 5.75 (h = 0.90) ---
    bus_box = FancyBboxPatch((0.40, 4.85), 14.20, 0.90, boxstyle="round,pad=0.08",
                             edgecolor='#334155', facecolor='#F8FAFC', linewidth=2.0, linestyle='-')
    ax.add_patch(bus_box)
    ax.text(7.50, 5.30, r"$\mathbf{PRODUCTION \; DEPLOYMENT \; PIPELINE \; (Algorithm \; 1):}$  " +
                        r"Deploy $\left\{ f_{\mathrm{core}}, \; \tau^*, \; g^{(j)}, \; \{\tau_{\mathrm{meta},k}\} \right\} \longrightarrow$ Streaming Runtime Engine",
            fontsize=11.4, fontweight='bold', ha='center', va='center', color='#0F172A')

    # Phase 3: y = 0.30 to 4.20 (h = 3.90)
    ax.add_patch(FancyBboxPatch((0.40, 0.30), 14.20, 3.90, boxstyle="round,pad=0.12",
                                edgecolor=c_p3_border, facecolor=c_p3_bg, linewidth=1.8, linestyle='--'))
    ax.text(0.65, 3.88, "PHASE 3: STREAMING PRODUCTION DEPLOYMENT WITH SAFETY GUARD", fontsize=12.6, fontweight='bold', color=c_p3_txt)

    def draw_box(x, y, w, h, title, subtitle="", facecolor='#FFFFFF', edgecolor='#CBD5E1', lw=1.5, textcolor='#0F172A', title_size=12.0, sub_size=10.2):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                             edgecolor=edgecolor, facecolor=facecolor, linewidth=lw)
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w/2.0, y + h*0.68, title, fontsize=title_size, fontweight='bold', ha='center', va='center', color=textcolor)
            ax.text(x + w/2.0, y + h*0.30, subtitle, fontsize=sub_size, ha='center', va='center', color='#334155', multialignment='center', linespacing=1.28)
        else:
            ax.text(x + w/2.0, y + h/2.0, title, fontsize=title_size, fontweight='bold', ha='center', va='center', color=textcolor, multialignment='center')
        return box

    def draw_arrow(x1, y1, x2, y2, label="", color='#334155', style='->', lw=1.9, label_offset=(0, 0), dashed=False, label_size=10.2):
        ls = '--' if dashed else '-'
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls, shrinkA=3, shrinkB=3))
        if label:
            lx = (x1 + x2) / 2.0 + label_offset[0]
            ly = (y1 + y2) / 2.0 + label_offset[1]
            ax.text(lx, ly, label, fontsize=label_size, fontweight='bold', ha='center', va='center',
                    color=color, bbox=dict(boxstyle='round,pad=0.22', facecolor='#FFFFFF', edgecolor=color, lw=1.1, alpha=0.98))

    # --- PHASE 1 BOXES ---
    draw_box(0.65, 8.35, 3.60, 1.35, r"Training Partition $\mathcal{D}_{\mathrm{train}}$ (80%)",
             "Chronological / Stratified Splits\nBenchmark Cross-Evaluation", 
             edgecolor='#93C5FD', title_size=12.2, sub_size=10.2)
    draw_box(0.65, 6.30, 3.60, 1.35, r"Stage 1: Primary GBDT Core",
             r"Cost-Weighted Ensemble $f_{\mathrm{core}}(\mathbf{x})$" + "\n" + r"Balanced Weighting ($w_{\mathrm{pos}}$)", 
             edgecolor='#2563EB', lw=1.9, title_size=12.2, sub_size=10.2)
    draw_arrow(2.45, 8.35, 2.45, 7.65)
    draw_arrow(4.25, 7.00, 5.05, 7.00, label=r"Validation Score: $\hat{p} \geq \tau^*$", color='#0D9488', lw=1.9, label_offset=(0, 0.38), label_size=10.5)

    # --- PHASE 2 BOXES ---
    draw_box(5.05, 8.35, 4.30, 1.35, r"Validation Partition $\mathcal{D}_{\mathrm{val}}$ (10%)",
             "Chronological / Stratified Splits\nUnseen Threshold Optimization",
             edgecolor='#99F6E4', title_size=12.2, sub_size=10.2)
    draw_box(5.05, 6.30, 4.30, 1.35, r"Stage 2: Optimal Threshold $\tau^*$ Search",
             r"Operational Cost Surface Search" + "\n" + r"$\tau^* = \arg\min_{\tau} \mathrm{Cost}(\tau)$",
             edgecolor='#0D9488', lw=1.9, title_size=12.2, sub_size=10.2)
    draw_arrow(7.20, 8.35, 7.20, 7.65)
    draw_arrow(9.35, 9.00, 9.95, 9.00)

    draw_box(9.95, 8.35, 4.40, 1.35, r"5-Fold OOF Cross-Fitting",
             r"Candidate Alerts $\mathcal{D}_{\mathrm{cand}}$ ($\hat{p} \geq \tau^*$)" + "\n" + r"TreeSHAP $\phi(\mathbf{x})$ + Meta-Model $g^{(j)}$",
             edgecolor='#0D9488', title_size=12.2, sub_size=10.2)
    draw_box(9.95, 6.30, 4.40, 1.35, r"Amount-Tiered Safety Calibration",
             r"Quantile Stratification ($K = 3$ Tiers)" + "\n" + r"Safety Thresholds $\tau_{\mathrm{meta},k}$ ($\Delta\mathrm{FN} = 0$)",
             edgecolor='#0D9488', lw=1.9, title_size=12.2, sub_size=10.2)
    
    # 0.70" vertical gap with centered pill box
    draw_arrow(12.15, 8.35, 12.15, 7.65, label=r"OOF Meta-Risk $\hat{q}$", color='#0D9488', lw=2.1, label_offset=(0, 0), label_size=10.8)

    # FEED INTO BUS
    draw_arrow(2.45, 6.30, 2.45, 5.75, color='#1D4ED8', lw=1.9)
    draw_arrow(7.20, 6.30, 7.20, 5.75, color='#0D9488', lw=1.9)
    draw_arrow(12.15, 6.30, 12.15, 5.75, color='#7C3AED', lw=1.9)

    # --- PHASE 3: HARMONIOUS SPACING WITH IDENTICAL RIGHT BORDER AT 14.35 ---
    # 1. Live Transaction: 0.65 to 2.50 (w = 1.85)
    draw_box(0.65, 1.50, 1.85, 1.90, r"Live Transaction",
             r"Feature Vector $\mathbf{x} \in \mathbb{R}^d$" + "\nReal-Time Ingestion\n" + r"Single-Row ($B = 1$)",
             edgecolor='#C084FC', title_size=11.8, sub_size=10.0)
    draw_arrow(2.50, 2.45, 2.85, 2.45)

    # 2. Stage 1 Fast-Path: 2.85 to 5.05 (w = 2.20)
    draw_box(2.85, 1.50, 2.20, 1.90, r"Stage 1 Fast-Path",
             r"Score $\hat{p} = f_{\mathrm{core}}(\mathbf{x})$" + "\nSub-millisecond Scoring\n" + r"Gate: $\hat{p} \geq \tau^*$ ?",
             edgecolor='#7C3AED', lw=1.9, title_size=11.8, sub_size=10.0)
    
    # Benign branch down:
    draw_arrow(3.95, 1.50, 3.95, 1.15, label=r"$\hat{p} < \tau^*$", color='#16A34A', label_offset=(-0.45, 0), label_size=10.2)
    draw_box(2.85, 0.45, 2.20, 0.70, r"Auto-Cleared Legit",
             r"Direct Approval ($\hat{y} = 0$)",
             facecolor='#DCFCE7', edgecolor='#16A34A', lw=1.5, title_size=10.6, sub_size=9.4)

    # Feed from Bus to Stage 1
    draw_arrow(3.95, 4.85, 3.95, 3.40, label=r"Deploy $f_{\mathrm{core}}$ & $\tau^*$", color='#1E40AF', lw=1.9, label_offset=(0, 0.42), label_size=10.5)

    # Alert branch right:
    draw_arrow(5.05, 2.45, 5.65, 2.45, label=r"$\hat{p} \geq \tau^*$", color='#DC2626', label_offset=(0, 0.30), label_size=10.4)

    # 3. Stage 3 TreeSHAP: 5.65 to 7.80 (w = 2.15)
    draw_box(5.65, 1.50, 2.15, 1.90, r"Stage 3 TreeSHAP",
             r"Local Attribution $\phi(\mathbf{x})$" + "\nExact TreeExplainer\nSelective Trigger",
             edgecolor='#7C3AED', lw=1.9, title_size=11.8, sub_size=10.0)
    draw_arrow(7.80, 2.45, 8.25, 2.45)

    # 4. Meta-Classification: 8.25 to 10.45 (w = 2.20)
    draw_box(8.25, 1.50, 2.20, 1.90, r"Meta-Classification",
             r"Joint Input $[\mathbf{x}, \phi(\mathbf{x})]$" + "\n" + r"Ensemble $g(\mathbf{u}) \to \bar{q}$" + "\nSecondary Filter",
             edgecolor='#7C3AED', lw=1.9, title_size=11.8, sub_size=10.0)
    draw_arrow(10.45, 2.45, 10.85, 2.45)

    # Feed from Bus to Meta-Classification & Safety Guard
    draw_arrow(9.35, 4.85, 9.35, 3.40, label=r"Deploy $g$ & $\{\tau_{\mathrm{meta},k}\}$", color='#5B21B6', lw=1.9, label_offset=(0, 0.42), label_size=10.5)

    # 5. Safety Guard: 10.85 to 12.20 (w = 1.35)
    draw_box(10.85, 1.50, 1.35, 1.90, r"Safety Guard",
             r"Tier $k(\mathrm{Amount})$" + "\n" + r"$\bar{q} \geq \tau_{\mathrm{meta},k}$ ?",
             facecolor='#FEF3C7', edgecolor='#D97706', lw=1.9, title_size=11.6, sub_size=9.8)

    # Upper Output: Auto-Cleared False Alarm
    draw_box(12.90, 2.15, 1.45, 1.30, "Auto-Cleared\nFalse Alarm\n" + r"($\hat{y} = 0$)",
             facecolor='#DCFCE7', edgecolor='#16A34A', lw=1.7, title_size=10.5)
    draw_arrow(12.20, 2.80, 12.90, 2.80, label=r"$\bar{q} \geq \tau_{\mathrm{meta},k}$", color='#16A34A', label_offset=(0, 0.22), label_size=10.0)

    # Lower Output: Escalated Candidate Alert
    draw_box(12.90, 0.65, 1.45, 1.30, "Escalated\nCandidate Alert\n" + r"($\hat{y} = 1$)",
             facecolor='#FEE2E2', edgecolor='#DC2626', lw=1.7, title_size=10.5)
    draw_arrow(12.20, 2.10, 12.90, 1.30, label=r"$\bar{q} < \tau_{\mathrm{meta},k}$", color='#DC2626', label_offset=(0, 0.22), label_size=10.0)

    out_pdf = '/tmp/test_figure1_vector.pdf'
    out_svg = '/tmp/test_figure1_vector.svg'
    out_png = '/tmp/test_figure1_vector.png'

    plt.savefig(out_pdf, format='pdf')
    plt.savefig(out_svg, format='svg')
    plt.savefig(out_png, format='png', dpi=300)
    plt.close()

    print("Generated:")
    print("  PDF:", out_pdf, os.path.getsize(out_pdf), "bytes")
    print("  SVG:", out_svg, os.path.getsize(out_svg), "bytes")
    print("  PNG:", out_png, os.path.getsize(out_png), "bytes")

if __name__ == '__main__':
    generate_exact_vector_figure1()
