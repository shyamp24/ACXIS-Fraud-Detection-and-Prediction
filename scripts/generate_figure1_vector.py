import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import os

def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(16, 9.5), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.5)
    ax.axis('off')
    
    # Background and Color Palette
    fig.patch.set_facecolor('#FFFFFF')
    
    # Colors
    c_blue_light = '#EFF6FF'   # Phase 1 background
    c_teal_light = '#F0FDFA'   # Phase 2 background
    c_purple_light = '#FAF5FF' # Phase 3 background
    
    # -------------------------------------------------------------------------
    # Phase Background Panels
    # -------------------------------------------------------------------------
    # Phase 1 Panel: Stage 1 Predictive Core (Offline Training)
    rect1 = FancyBboxPatch((0.4, 4.8), 4.6, 4.2, boxstyle="round,pad=0.2",
                           edgecolor='#3B82F6', facecolor=c_blue_light, linewidth=1.5, linestyle='--')
    ax.add_patch(rect1)
    ax.text(0.6, 8.7, "PHASE 1: STAGE 1 PREDICTIVE CORE (OFFLINE TRAINING)", 
            fontsize=10.5, fontweight='bold', color='#1D4ED8')

    # Phase 2 Panel: Stage 2 Validation & OOF Calibration (enclosing all of Phase 2)
    rect2 = FancyBboxPatch((5.4, 4.8), 10.2, 4.2, boxstyle="round,pad=0.2",
                           edgecolor='#0D9488', facecolor=c_teal_light, linewidth=1.5, linestyle='--')
    ax.add_patch(rect2)
    ax.text(5.6, 8.7, r"PHASE 2: STAGE 2 VALIDATION & OOF CALIBRATION", 
            fontsize=10.5, fontweight='bold', color='#0F766E')

    # Phase 3 Panel: Stage 3 Production Streaming Inference
    rect3 = FancyBboxPatch((0.4, 0.4), 15.2, 3.8, boxstyle="round,pad=0.2",
                           edgecolor='#7C3AED', facecolor=c_purple_light, linewidth=1.5, linestyle='--')
    ax.add_patch(rect3)
    ax.text(0.6, 3.85, "PHASE 3: STAGE 3 SELECTIVE PRODUCTION STREAMING INFERENCE", 
            fontsize=10.5, fontweight='bold', color='#6D28D9')

    # -------------------------------------------------------------------------
    # Helper to draw styled boxes
    # -------------------------------------------------------------------------
    def draw_box(x, y, w, h, title, subtitle="", facecolor='#FFFFFF', edgecolor='#CBD5E1', lw=1.2, textcolor='#1E293B'):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             edgecolor=edgecolor, facecolor=facecolor, linewidth=lw)
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w/2.0, y + h*0.62, title, fontsize=9.5, fontweight='bold',
                    ha='center', va='center', color=textcolor)
            ax.text(x + w/2.0, y + h*0.28, subtitle, fontsize=8.0,
                    ha='center', va='center', color='#475569')
        else:
            ax.text(x + w/2.0, y + h/2.0, title, fontsize=9.0, fontweight='bold',
                    ha='center', va='center', color=textcolor)
        return box

    def draw_arrow(x1, y1, x2, y2, label="", color='#64748B', style='->', lw=1.5, label_pos=0.5, label_offset=(0, 0.15)):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=3, shrinkB=3))
        if label:
            lx = x1 + (x2 - x1) * label_pos + label_offset[0]
            ly = y1 + (y2 - y1) * label_pos + label_offset[1]
            ax.text(lx, ly, label, fontsize=8.0, fontweight='bold', ha='center', va='center',
                    color=color, bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFFFFF', edgecolor=color, lw=0.8, alpha=0.9))

    # -------------------------------------------------------------------------
    # Phase 1 Components
    # -------------------------------------------------------------------------
    draw_box(0.7, 7.3, 4.0, 1.0, r"Dataset Partition $\mathcal{D}_{\mathrm{train}}$ (80%)", 
             "PaySim / IEEE-CIS / ULB:\nStandardized Stratified Split (80/10/10)", facecolor='#FFFFFF', edgecolor='#93C5FD')
    
    draw_box(0.7, 5.2, 4.0, 1.2, "Stage 1 XGBoost Predictive Core", 
             r"Cost-Weighted Tree Ensemble $f_{\mathrm{core}}(\mathbf{x})$" + "\n" + r"Deterministic Objective Optimization", facecolor='#FFFFFF', edgecolor='#3B82F6', lw=1.5)

    draw_arrow(2.7, 7.3, 2.7, 6.4, label="")

    # -------------------------------------------------------------------------
    # Phase 2 Components
    # -------------------------------------------------------------------------
    draw_box(5.7, 7.3, 4.4, 1.0, r"Validation Split $\mathcal{D}_{\mathrm{val}}$ (10%)", 
             r"$\hat{p}_{\mathrm{val}} = \sigma(f_{\mathrm{core}}(\mathbf{x}))$ via Held-Out Split", facecolor='#FFFFFF', edgecolor='#99F6E4')

    draw_box(5.7, 5.2, 4.4, 1.2, r"$k=5$ Stratified OOF Cross-Fitting", 
             r"Extract Alerts ($\hat{p} \geq \tau^*$)" + "\n" + r"Local TreeSHAP $\phi(\mathbf{x})$ + Train RF Meta", facecolor='#FFFFFF', edgecolor='#0D9488', lw=1.5)

    draw_arrow(4.7, 5.8, 5.7, 5.8, label=r"Deploy $f_{\mathrm{core}}$")
    draw_arrow(7.9, 7.3, 7.9, 6.4, label="")

    # Cost calibration box on the right of Phase 2
    draw_box(10.8, 5.2, 4.4, 3.1, r"Cost-Sensitive Calibration", 
             r"1. Threshold Search: $\tau^* = \arg\min \mathrm{Cost}(\tau)$" + "\n" + 
             r"   Validation Grid Search on $\mathcal{D}_{\mathrm{val}}$" + "\n" +
             r"2. Safety Guard: $\tau_{\mathrm{meta}}^*$ Calibration" + "\n" +
             r"   $\tau_{\mathrm{meta}}^* = \min \{q : \Delta\mathrm{FN}_{\mathrm{OOF}}(q) = 0\}$" + "\n" +
             r"3. Enforces Zero Additional FN Leaks", 
             facecolor='#F8FAFC', edgecolor='#0F766E', lw=1.5)

    draw_arrow(10.1, 5.8, 10.8, 5.8, label="OOF Context")

    # -------------------------------------------------------------------------
    # Phase 3 Components (Streaming Inference)
    # -------------------------------------------------------------------------
    # Input stream
    draw_box(0.6, 2.0, 1.8, 1.35, r"Live Transaction $\mathbf{x}_k$", 
             "Unlabeled Production Stream\n(B=1 or Batched B=1000)", facecolor='#FFFFFF', edgecolor='#C4B5FD')

    # Preprocessing
    draw_box(2.7, 2.0, 1.8, 1.35, "Preprocessing", 
             r"Feature Encoding & Pipeline" + "\n" + r"Low-Latency Streaming Engine", facecolor='#FFFFFF', edgecolor='#A78BFA')

    # Stage 1 Scoring
    draw_box(4.8, 2.0, 2.0, 1.35, "Stage 1 XGBoost", 
             r"Compute $\hat{p}_k = \sigma(f_{\mathrm{core}}(\mathbf{x}_k))$" + "\n" + r"High-Throughput Fast-Path Engine", facecolor='#FFFFFF', edgecolor='#7C3AED', lw=1.5)

    draw_arrow(2.4, 2.675, 2.7, 2.675)
    draw_arrow(4.5, 2.675, 4.8, 2.675)

    # Decision Gate 1 (Threshold check) - centered at x=7.8
    diamond1 = patches.Polygon([[7.2, 2.675], [7.8, 3.35], [8.4, 2.675], [7.8, 2.0]], 
                                closed=True, edgecolor='#D97706', facecolor='#FEF3C7', lw=1.5)
    ax.add_patch(diamond1)
    ax.text(7.8, 2.675, r"$\hat{p}_k \geq \tau^*$ ?", fontsize=9.0, fontweight='bold', ha='center', va='center', color='#92400E')

    draw_arrow(6.8, 2.675, 7.2, 2.675)

    # Fast-Path Bypass (Downward branch from Diamond to Cleared box)
    draw_arrow(7.8, 2.0, 7.8, 1.20, label="", color='#059669')
    ax.text(7.8, 1.6, r"NO (Fast-Path: $\hat{p}_k < \tau^*$)", fontsize=7.5, fontweight='bold', ha='center', va='center',
            color='#059669', bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFFFFF', edgecolor='#059669', lw=0.8))

    draw_box(4.2, 0.50, 4.6, 0.70, r"Fast-Path Clearance: Emit $\hat{y}_k = 0$ (Cleared)", 
             "Bypasses Heavy XAI & Human Review", facecolor='#ECFDF5', edgecolor='#10B981', textcolor='#065F46', lw=1.3)

    # Candidate Alert Branch (Rightward arrow to SHAP)
    draw_arrow(8.4, 2.675, 9.5, 2.675, label="", color='#DC2626')
    ax.text(8.95, 2.95, r"YES (Alert: $\hat{p}_k \geq \tau^*$)", fontsize=7.8, fontweight='bold', ha='center', va='center',
            color='#DC2626', bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFFFFF', edgecolor='#DC2626', lw=0.8, alpha=0.95))

    # Stage 3 Explainer & Meta-Classifier
    draw_box(9.5, 2.0, 2.3, 1.35, "TreeExplainer SHAP", 
             r"Compute Attribution $\phi(\mathbf{x}_k)$" + "\n" + r"Selective Trigger (Alerts Only)", facecolor='#FFFFFF', edgecolor='#EC4899')

    draw_arrow(11.8, 2.675, 12.2, 2.675)

    draw_box(12.2, 2.0, 2.3, 1.35, "Stage 3 RF Meta", 
             r"Input Space: $[\mathbf{x}_k, \phi(\mathbf{x}_k)]$" + "\n" + r"Predict Meta-Prob $q_k \in [0, 1]$", facecolor='#FFFFFF', edgecolor='#6366F1')

    # Arrow from Stage 3 RF Meta down to Safety Gate
    draw_arrow(13.35, 2.0, 13.35, 1.20, label="", color='#4F46E5')

    # Safety Gate Box
    draw_box(9.6, 0.50, 4.2, 0.70, "Safety-Guarded Decision Gate", 
             r"Suppressed ($s_k=1$) if $q_k \geq \tau_{\mathrm{meta}}^*$; else Actionable Alert ($s_k=0$)" + "\n" + r"Preserves Stage 1 Recall (0.00 FN Leaks)", 
             facecolor='#EFF6FF', edgecolor='#2563EB', lw=1.4, textcolor='#1E40AF')

    # Suppression arrow (left to Cleared box)
    draw_arrow(9.6, 0.85, 8.8, 0.85, label="", color='#059669')
    ax.text(9.2, 1.05, r"Suppressed ($s_k=1$)", fontsize=7.0, fontweight='bold', ha='center', va='center', color='#059669')

    # Confirmed Alert box (Right)
    draw_box(14.2, 0.50, 1.4, 0.70, r"Emit $\hat{y}_k = 1$", 
             "Actionable Alert", facecolor='#FEF2F2', edgecolor='#EF4444', textcolor='#991B1B', lw=1.3)

    # Confirmed Fraud arrow (right to Alert box)
    draw_arrow(13.8, 0.85, 14.2, 0.85, label="", color='#DC2626')
    ax.text(14.0, 1.05, r"Alert ($s_k=0$)", fontsize=7.0, fontweight='bold', ha='center', va='center', color='#DC2626')

    plt.tight_layout()
    
    # Save formats
    base_dir = "/Users/shyamsundar/Dropbox/4 SHYAM DOCs/PHD/PET 2021/PROJECT/Scopus ready Paper/Paper I/Array Elsivier/Array revision Second"
    results_dir = os.path.join(base_dir, "02_R2_Results")
    repl_dir = os.path.join(base_dir, "ACXIS_Replication_Package")
    
    pdf_path = os.path.join(results_dir, "Figure_1_ACXIS_Architecture.pdf")
    svg_path = os.path.join(results_dir, "Figure_1_ACXIS_Architecture.svg")
    png_path = os.path.join(results_dir, "Figure_1_ACXIS_Architecture.png")
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    
    if os.path.exists(repl_dir):
        plt.savefig(os.path.join(repl_dir, "Figure_1_ACXIS_Architecture.png"), format='png', bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(repl_dir, "Figure_1_ACXIS_Architecture.pdf"), format='pdf', bbox_inches='tight', dpi=300)
    
    plt.close()
    
    print(f"Figure 1 successfully generated in vector PDF, SVG, and 300-DPI PNG:\n  {pdf_path}\n  {svg_path}\n  {png_path}")

if __name__ == '__main__':
    create_architecture_diagram()
