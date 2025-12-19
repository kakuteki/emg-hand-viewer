"""
Publication-Quality Feature Ranking Figure for EMG Analysis
Creates figures suitable for academic journal submission
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import json

# Load analysis results
with open('novel_feature_analysis_results.json', 'r') as f:
    novel_results = json.load(f)

with open('feature_analysis_results.json', 'r') as f:
    basic_results = json.load(f)

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0,
})

# Color palette (colorblind-friendly)
COLOR_NOVEL = '#E64B35'       # Red - Novel features
COLOR_TRADITIONAL = '#4DBBD5' # Blue - Traditional features
COLOR_TIME = '#00A087'        # Green - Time-domain
COLOR_FREQ = '#3C5488'        # Dark blue - Frequency-domain
COLOR_GRADIENT = '#F39B7F'    # Light coral - Gradient

# =============================================================================
# Figure 1: Comprehensive Feature Ranking (Correlation-based)
# =============================================================================
def create_correlation_ranking_figure():
    """Create publication-quality correlation ranking figure"""

    fig, ax = plt.subplots(figsize=(7, 6))

    # Get data from novel analysis (includes both novel and traditional)
    corr_data = novel_results['correlation_ranking']

    # Sort by mean correlation (top 15)
    top_features = corr_data[:15]

    features = [f['feature'] for f in top_features]
    mean_corrs = [f['mean_correlation'] for f in top_features]
    max_corrs = [f['max_correlation'] for f in top_features]
    is_novel = [f['is_novel'] for f in top_features]

    y_pos = np.arange(len(features))

    # Color based on novel vs traditional
    colors = [COLOR_NOVEL if n else COLOR_TRADITIONAL for n in is_novel]

    # Create horizontal bar chart
    bars = ax.barh(y_pos, mean_corrs, height=0.7, color=colors,
                   edgecolor='black', linewidth=0.5, alpha=0.85)

    # Add error bars (showing range from mean to max)
    for i, (mean_c, max_c) in enumerate(zip(mean_corrs, max_corrs)):
        ax.plot([mean_c, max_c], [i, i], 'k-', linewidth=1.5, alpha=0.7)
        ax.plot(max_c, i, 'k|', markersize=6, markeredgewidth=1.5)

    # Add correlation values
    for i, (mean_c, max_c) in enumerate(zip(mean_corrs, max_corrs)):
        ax.text(mean_c + 0.01, i, f'{mean_c:.3f}', va='center', ha='left',
                fontsize=8, fontweight='medium')

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Absolute Correlation with Joint Angles', fontweight='bold')
    ax.set_xlim(0, 0.85)
    ax.xaxis.set_major_locator(MaxNLocator(5))

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_NOVEL, edgecolor='black',
                       linewidth=0.5, label='Novel Features'),
        mpatches.Patch(facecolor=COLOR_TRADITIONAL, edgecolor='black',
                       linewidth=0.5, label='Traditional Features'),
        plt.Line2D([0], [0], color='black', linewidth=1.5,
                   label='Max Correlation')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
              fancybox=False, edgecolor='black')

    # Add grid
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)

    # Title
    ax.set_title('(A) Feature Ranking by Correlation with Joint Angles',
                 fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig('Fig_FeatureRanking_Correlation.png', dpi=300,
                facecolor='white', edgecolor='none')
    plt.savefig('Fig_FeatureRanking_Correlation.pdf', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: Fig_FeatureRanking_Correlation.png/pdf")

# =============================================================================
# Figure 2: Feature Importance (Random Forest)
# =============================================================================
def create_importance_ranking_figure():
    """Create publication-quality importance ranking figure"""

    fig, ax = plt.subplots(figsize=(7, 6))

    # Get importance data
    imp_data = novel_results['importance_ranking']

    # Top 15 features
    top_features = imp_data[:15]

    features = [f['feature'] for f in top_features]
    importance = [f['importance'] * 100 for f in top_features]  # Convert to percentage
    is_novel = [f['is_novel'] for f in top_features]

    y_pos = np.arange(len(features))

    # Color based on novel vs traditional
    colors = [COLOR_NOVEL if n else COLOR_TRADITIONAL for n in is_novel]

    # Create horizontal bar chart
    bars = ax.barh(y_pos, importance, height=0.7, color=colors,
                   edgecolor='black', linewidth=0.5, alpha=0.85)

    # Add importance values
    for i, imp in enumerate(importance):
        ax.text(imp + 0.3, i, f'{imp:.1f}%', va='center', ha='left',
                fontsize=8, fontweight='medium')

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (%)', fontweight='bold')
    ax.set_xlim(0, 25)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_NOVEL, edgecolor='black',
                       linewidth=0.5, label='Novel Features'),
        mpatches.Patch(facecolor=COLOR_TRADITIONAL, edgecolor='black',
                       linewidth=0.5, label='Traditional Features')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
              fancybox=False, edgecolor='black')

    # Add grid
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)

    # Title
    ax.set_title('(B) Feature Importance Ranking (Random Forest)',
                 fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig('Fig_FeatureRanking_Importance.png', dpi=300,
                facecolor='white', edgecolor='none')
    plt.savefig('Fig_FeatureRanking_Importance.pdf', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: Fig_FeatureRanking_Importance.png/pdf")

# =============================================================================
# Figure 3: Combined Figure (Two-panel)
# =============================================================================
def create_combined_figure():
    """Create combined two-panel publication figure"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Panel A: Correlation Ranking ---
    ax1 = axes[0]

    corr_data = novel_results['correlation_ranking']
    top_corr = corr_data[:12]

    features_corr = [f['feature'] for f in top_corr]
    mean_corrs = [f['mean_correlation'] for f in top_corr]
    max_corrs = [f['max_correlation'] for f in top_corr]
    is_novel_corr = [f['is_novel'] for f in top_corr]

    y_pos_corr = np.arange(len(features_corr))
    colors_corr = [COLOR_NOVEL if n else COLOR_TRADITIONAL for n in is_novel_corr]

    bars1 = ax1.barh(y_pos_corr, mean_corrs, height=0.65, color=colors_corr,
                     edgecolor='black', linewidth=0.5, alpha=0.85)

    # Error bars for max correlation
    for i, (mean_c, max_c) in enumerate(zip(mean_corrs, max_corrs)):
        ax1.plot([mean_c, max_c], [i, i], 'k-', linewidth=1.2, alpha=0.6)
        ax1.plot(max_c, i, 'k|', markersize=5, markeredgewidth=1.2)

    # Values
    for i, mean_c in enumerate(mean_corrs):
        ax1.text(mean_c + 0.01, i, f'{mean_c:.3f}', va='center', ha='left', fontsize=8)

    ax1.set_yticks(y_pos_corr)
    ax1.set_yticklabels(features_corr)
    ax1.invert_yaxis()
    ax1.set_xlabel('Mean Absolute Correlation', fontweight='bold')
    ax1.set_xlim(0, 0.85)
    ax1.set_axisbelow(True)
    ax1.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.set_title('(A) Correlation with Joint Angles', fontweight='bold', pad=10)

    # --- Panel B: Importance Ranking ---
    ax2 = axes[1]

    imp_data = novel_results['importance_ranking']
    top_imp = imp_data[:12]

    features_imp = [f['feature'] for f in top_imp]
    importance = [f['importance'] * 100 for f in top_imp]
    is_novel_imp = [f['is_novel'] for f in top_imp]

    y_pos_imp = np.arange(len(features_imp))
    colors_imp = [COLOR_NOVEL if n else COLOR_TRADITIONAL for n in is_novel_imp]

    bars2 = ax2.barh(y_pos_imp, importance, height=0.65, color=colors_imp,
                     edgecolor='black', linewidth=0.5, alpha=0.85)

    # Values
    for i, imp in enumerate(importance):
        ax2.text(imp + 0.2, i, f'{imp:.1f}%', va='center', ha='left', fontsize=8)

    ax2.set_yticks(y_pos_imp)
    ax2.set_yticklabels(features_imp)
    ax2.invert_yaxis()
    ax2.set_xlabel('Feature Importance (%)', fontweight='bold')
    ax2.set_xlim(0, 25)
    ax2.set_axisbelow(True)
    ax2.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax2.set_title('(B) Random Forest Feature Importance', fontweight='bold', pad=10)

    # Shared legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_NOVEL, edgecolor='black',
                       linewidth=0.5, label='Novel Features (2024)'),
        mpatches.Patch(facecolor=COLOR_TRADITIONAL, edgecolor='black',
                       linewidth=0.5, label='Traditional Features'),
        plt.Line2D([0], [0], color='black', linewidth=1.2,
                   label='Maximum Correlation')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 0.02), frameon=True, fancybox=False,
               edgecolor='black')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    plt.savefig('Fig_FeatureRanking_Combined.png', dpi=300,
                facecolor='white', edgecolor='none')
    plt.savefig('Fig_FeatureRanking_Combined.pdf', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: Fig_FeatureRanking_Combined.png/pdf")

# =============================================================================
# Figure 4: Comprehensive Feature Analysis (Single Publication Figure)
# =============================================================================
def create_comprehensive_figure():
    """
    Create comprehensive single figure for publication
    Includes: Correlation ranking, Importance ranking, Novel vs Traditional comparison
    """

    fig = plt.figure(figsize=(10, 10))

    # Create grid spec for flexible layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 0.8],
                          hspace=0.35, wspace=0.3)

    # --- Panel A: Correlation Ranking (Top Left) ---
    ax1 = fig.add_subplot(gs[0, 0])

    corr_data = novel_results['correlation_ranking']
    top_corr = corr_data[:10]

    features_corr = [f['feature'] for f in top_corr]
    mean_corrs = [f['mean_correlation'] for f in top_corr]
    is_novel_corr = [f['is_novel'] for f in top_corr]

    y_pos = np.arange(len(features_corr))
    colors = [COLOR_NOVEL if n else COLOR_TRADITIONAL for n in is_novel_corr]

    ax1.barh(y_pos, mean_corrs, height=0.6, color=colors,
             edgecolor='black', linewidth=0.5, alpha=0.85)

    for i, c in enumerate(mean_corrs):
        ax1.text(c + 0.008, i, f'{c:.3f}', va='center', ha='left', fontsize=8)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(features_corr, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('Mean Correlation', fontsize=9)
    ax1.set_xlim(0, 0.55)
    ax1.set_axisbelow(True)
    ax1.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.set_title('(A) Correlation-based Ranking', fontweight='bold', fontsize=10)

    # --- Panel B: Importance Ranking (Top Right) ---
    ax2 = fig.add_subplot(gs[0, 1])

    imp_data = novel_results['importance_ranking']
    top_imp = imp_data[:10]

    features_imp = [f['feature'] for f in top_imp]
    importance = [f['importance'] * 100 for f in top_imp]
    is_novel_imp = [f['is_novel'] for f in top_imp]

    y_pos = np.arange(len(features_imp))
    colors = [COLOR_NOVEL if n else COLOR_TRADITIONAL for n in is_novel_imp]

    ax2.barh(y_pos, importance, height=0.6, color=colors,
             edgecolor='black', linewidth=0.5, alpha=0.85)

    for i, imp in enumerate(importance):
        ax2.text(imp + 0.2, i, f'{imp:.1f}%', va='center', ha='left', fontsize=8)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(features_imp, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel('Importance (%)', fontsize=9)
    ax2.set_xlim(0, 25)
    ax2.set_axisbelow(True)
    ax2.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax2.set_title('(B) Random Forest Importance', fontweight='bold', fontsize=10)

    # --- Panel C: Correlation vs Importance Scatter (Middle Left) ---
    ax3 = fig.add_subplot(gs[1, 0])

    # Merge correlation and importance data
    feature_dict = {}
    for item in corr_data:
        feature_dict[item['feature']] = {
            'corr': item['mean_correlation'],
            'is_novel': item['is_novel']
        }
    for item in imp_data:
        if item['feature'] in feature_dict:
            feature_dict[item['feature']]['imp'] = item['importance'] * 100

    # Filter features that have both values
    scatter_data = [(k, v['corr'], v['imp'], v['is_novel'])
                    for k, v in feature_dict.items() if 'imp' in v]

    for name, corr, imp, is_novel in scatter_data:
        color = COLOR_NOVEL if is_novel else COLOR_TRADITIONAL
        ax3.scatter(corr, imp, c=color, s=60, edgecolors='black',
                    linewidth=0.5, alpha=0.8)
        # Label top features
        if corr > 0.35 or imp > 8:
            ax3.annotate(name, (corr, imp), fontsize=7,
                        xytext=(3, 3), textcoords='offset points')

    ax3.set_xlabel('Mean Correlation', fontsize=9)
    ax3.set_ylabel('Feature Importance (%)', fontsize=9)
    ax3.set_axisbelow(True)
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.set_title('(C) Correlation vs Importance', fontweight='bold', fontsize=10)

    # --- Panel D: Novel vs Traditional Comparison (Middle Right) ---
    ax4 = fig.add_subplot(gs[1, 1])

    novel_vs_trad = novel_results['novel_vs_traditional']

    categories = ['Novel\nFeatures', 'Traditional\nFeatures']
    corr_values = [novel_vs_trad['novel_max_correlation'],
                   novel_vs_trad['traditional_max_correlation']]
    imp_values = [novel_vs_trad['novel_total_importance'] * 100,
                  novel_vs_trad['traditional_total_importance'] * 100]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax4.bar(x - width/2, corr_values, width, label='Max Correlation',
                    color='#3C5488', edgecolor='black', linewidth=0.5, alpha=0.85)
    bars2 = ax4.bar(x + width/2, [v/100 for v in imp_values], width,
                    label='Total Importance',
                    color='#E64B35', edgecolor='black', linewidth=0.5, alpha=0.85)

    ax4.set_ylabel('Value', fontsize=9)
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, fontsize=9)
    ax4.legend(fontsize=8, loc='upper right')
    ax4.set_ylim(0, 0.85)
    ax4.set_axisbelow(True)
    ax4.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax4.set_title('(D) Novel vs Traditional Comparison', fontweight='bold', fontsize=10)

    # Add values on bars
    for bar, val in zip(bars1, corr_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, [v/100 for v in imp_values]):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    # --- Panel E: Recommended Feature Set (Bottom, spanning both columns) ---
    ax5 = fig.add_subplot(gs[2, :])

    recommended = novel_results['recommendations']['best_combined']

    # Get correlation and importance for recommended features
    rec_corr = []
    rec_imp = []
    rec_novel = []
    for feat in recommended:
        for item in corr_data:
            if item['feature'] == feat:
                rec_corr.append(item['mean_correlation'])
                rec_novel.append(item['is_novel'])
                break
        for item in imp_data:
            if item['feature'] == feat:
                rec_imp.append(item['importance'] * 100)
                break

    x = np.arange(len(recommended))
    width = 0.35

    colors_rec = [COLOR_NOVEL if n else COLOR_TRADITIONAL for n in rec_novel]

    ax5.bar(x - width/2, rec_corr, width, label='Correlation',
            color=colors_rec, edgecolor='black', linewidth=0.5, alpha=0.6)
    ax5.bar(x + width/2, [v/50 for v in rec_imp], width, label='Importance (scaled)',
            color=colors_rec, edgecolor='black', linewidth=0.5, alpha=0.9,
            hatch='///')

    ax5.set_ylabel('Value', fontsize=9)
    ax5.set_xticks(x)
    ax5.set_xticklabels(recommended, fontsize=9, rotation=30, ha='right')
    ax5.legend(fontsize=8, loc='upper right')
    ax5.set_axisbelow(True)
    ax5.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax5.set_title('(E) Recommended Feature Set', fontweight='bold', fontsize=10)

    # Main legend at bottom
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_NOVEL, edgecolor='black',
                       linewidth=0.5, label='Novel Features (2024)'),
        mpatches.Patch(facecolor=COLOR_TRADITIONAL, edgecolor='black',
                       linewidth=0.5, label='Traditional Features')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.02), frameon=True, fancybox=False,
               edgecolor='black', fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    plt.savefig('Fig_FeatureAnalysis_Comprehensive.png', dpi=300,
                facecolor='white', edgecolor='none')
    plt.savefig('Fig_FeatureAnalysis_Comprehensive.pdf', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: Fig_FeatureAnalysis_Comprehensive.png/pdf")

# =============================================================================
# Figure 5: Simple Clean Feature Ranking (Most Journal-Friendly)
# =============================================================================
def create_simple_ranking_figure():
    """
    Create simple, clean feature ranking figure
    Most suitable for single-column journal format
    """

    fig, ax = plt.subplots(figsize=(5, 7))

    # Combine all features from both analyses
    corr_data = novel_results['correlation_ranking']
    imp_data = novel_results['importance_ranking']

    # Create combined score (normalized correlation + normalized importance)
    feature_scores = {}

    max_corr = max(f['mean_correlation'] for f in corr_data)
    max_imp = max(f['importance'] for f in imp_data)

    for item in corr_data:
        feature_scores[item['feature']] = {
            'corr_norm': item['mean_correlation'] / max_corr,
            'is_novel': item['is_novel'],
            'corr': item['mean_correlation']
        }

    for item in imp_data:
        if item['feature'] in feature_scores:
            feature_scores[item['feature']]['imp_norm'] = item['importance'] / max_imp
            feature_scores[item['feature']]['imp'] = item['importance'] * 100

    # Calculate combined score
    for feat, data in feature_scores.items():
        if 'imp_norm' in data:
            data['combined'] = 0.5 * data['corr_norm'] + 0.5 * data['imp_norm']
        else:
            data['combined'] = data['corr_norm'] * 0.5

    # Sort by combined score
    sorted_features = sorted(feature_scores.items(),
                            key=lambda x: x[1]['combined'], reverse=True)

    # Top 15
    top_features = sorted_features[:15]

    names = [f[0] for f in top_features]
    combined_scores = [f[1]['combined'] for f in top_features]
    is_novel = [f[1]['is_novel'] for f in top_features]

    y_pos = np.arange(len(names))
    colors = [COLOR_NOVEL if n else COLOR_TRADITIONAL for n in is_novel]

    # Create bars
    bars = ax.barh(y_pos, combined_scores, height=0.65, color=colors,
                   edgecolor='black', linewidth=0.5, alpha=0.85)

    # Add rank numbers
    for i, (name, score) in enumerate(zip(names, combined_scores)):
        ax.text(-0.03, i, f'{i+1}', va='center', ha='right', fontsize=9,
                fontweight='bold', color='#333333')
        ax.text(score + 0.015, i, f'{score:.3f}', va='center', ha='left', fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Combined Score\n(Correlation + Importance)', fontsize=9, fontweight='bold')
    ax.set_xlim(-0.08, 1.15)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)

    # Remove left spine
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_NOVEL, edgecolor='black',
                       linewidth=0.5, label='Novel (2024)'),
        mpatches.Patch(facecolor=COLOR_TRADITIONAL, edgecolor='black',
                       linewidth=0.5, label='Traditional')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
              fancybox=False, edgecolor='black', fontsize=8)

    ax.set_title('EMG Feature Ranking\nfor Joint Angle Prediction',
                 fontweight='bold', fontsize=11, pad=10)

    plt.tight_layout()

    plt.savefig('Fig_FeatureRanking_Simple.png', dpi=300,
                facecolor='white', edgecolor='none')
    plt.savefig('Fig_FeatureRanking_Simple.pdf', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: Fig_FeatureRanking_Simple.png/pdf")

# =============================================================================
# Create Summary Table Data
# =============================================================================
def create_summary_table():
    """Create summary table data for paper"""

    corr_data = novel_results['correlation_ranking']
    imp_data = novel_results['importance_ranking']

    print("\n" + "="*80)
    print("TABLE: Feature Ranking Summary (for copy-paste to paper)")
    print("="*80)

    # Create merged data
    feature_dict = {}
    for item in corr_data:
        feature_dict[item['feature']] = {
            'rank_corr': item['rank'],
            'corr': item['mean_correlation'],
            'max_corr': item['max_correlation'],
            'is_novel': item['is_novel']
        }

    for item in imp_data:
        if item['feature'] in feature_dict:
            feature_dict[item['feature']]['rank_imp'] = item['rank']
            feature_dict[item['feature']]['imp'] = item['importance'] * 100

    # Sort by correlation rank
    sorted_data = sorted(feature_dict.items(), key=lambda x: x[1]['rank_corr'])

    print("\nFeature\t\tType\t\tCorr Rank\tMean Corr\tImp Rank\tImportance(%)")
    print("-" * 90)

    for name, data in sorted_data[:15]:
        feat_type = "Novel" if data['is_novel'] else "Traditional"
        imp_rank = data.get('rank_imp', 'N/A')
        imp_val = f"{data.get('imp', 0):.2f}" if 'imp' in data else 'N/A'

        # Pad feature name for alignment
        name_padded = name.ljust(12)
        print(f"{name_padded}\t{feat_type}\t\t{data['rank_corr']}\t\t{data['corr']:.4f}\t\t{imp_rank}\t\t{imp_val}")

    print("\n" + "="*80)

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == '__main__':
    print("Creating publication-quality figures...")
    print("="*60)

    create_correlation_ranking_figure()
    create_importance_ranking_figure()
    create_combined_figure()
    create_comprehensive_figure()
    create_simple_ranking_figure()
    create_summary_table()

    print("\n" + "="*60)
    print("All figures created successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - Fig_FeatureRanking_Correlation.png/pdf")
    print("  - Fig_FeatureRanking_Importance.png/pdf")
    print("  - Fig_FeatureRanking_Combined.png/pdf")
    print("  - Fig_FeatureAnalysis_Comprehensive.png/pdf")
    print("  - Fig_FeatureRanking_Simple.png/pdf")
    print("\nRecommended for journal submission:")
    print("  - Single column: Fig_FeatureRanking_Simple.pdf")
    print("  - Double column: Fig_FeatureRanking_Combined.pdf")
    print("  - Full page: Fig_FeatureAnalysis_Comprehensive.pdf")
