"""
論文品質の特徴量順位図（日本語版）
Publication-Quality Feature Ranking Figure for EMG Analysis (Japanese)
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

# Configure matplotlib for publication quality with Japanese font
plt.rcParams.update({
    'font.family': ['MS Gothic', 'Meiryo', 'Yu Gothic', 'Hiragino Sans', 'sans-serif'],
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
    'axes.unicode_minus': False,
})

# Color palette (colorblind-friendly)
COLOR_NOVEL = '#E64B35'       # 赤 - 新規特徴量
COLOR_TRADITIONAL = '#4DBBD5' # 青 - 従来特徴量

# =============================================================================
# 図1: 相関係数に基づく特徴量順位
# =============================================================================
def create_correlation_ranking_figure():
    """相関係数順位図（日本語）"""

    fig, ax = plt.subplots(figsize=(7, 6))

    corr_data = novel_results['correlation_ranking']
    top_features = corr_data[:15]

    features = [f['feature'] for f in top_features]
    mean_corrs = [f['mean_correlation'] for f in top_features]
    max_corrs = [f['max_correlation'] for f in top_features]
    is_novel = [f['is_novel'] for f in top_features]

    y_pos = np.arange(len(features))
    colors = [COLOR_NOVEL if n else COLOR_TRADITIONAL for n in is_novel]

    bars = ax.barh(y_pos, mean_corrs, height=0.7, color=colors,
                   edgecolor='black', linewidth=0.5, alpha=0.85)

    # 最大相関を示す線分
    for i, (mean_c, max_c) in enumerate(zip(mean_corrs, max_corrs)):
        ax.plot([mean_c, max_c], [i, i], 'k-', linewidth=1.5, alpha=0.7)
        ax.plot(max_c, i, 'k|', markersize=6, markeredgewidth=1.5)

    # 数値表示
    for i, (mean_c, max_c) in enumerate(zip(mean_corrs, max_corrs)):
        ax.text(mean_c + 0.01, i, f'{mean_c:.3f}', va='center', ha='left',
                fontsize=8, fontweight='medium')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('関節角度との平均絶対相関係数', fontweight='bold')
    ax.set_xlim(0, 0.85)
    ax.xaxis.set_major_locator(MaxNLocator(5))

    # 凡例
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_NOVEL, edgecolor='black',
                       linewidth=0.5, label='新規特徴量'),
        mpatches.Patch(facecolor=COLOR_TRADITIONAL, edgecolor='black',
                       linewidth=0.5, label='従来特徴量'),
        plt.Line2D([0], [0], color='black', linewidth=1.5,
                   label='最大相関係数')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
              fancybox=False, edgecolor='black')

    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)

    ax.set_title('(A) 相関係数に基づく特徴量の順位付け',
                 fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig('Fig_特徴量順位_相関係数.png', dpi=300,
                facecolor='white', edgecolor='none')
    plt.savefig('Fig_特徴量順位_相関係数.pdf', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
    print("保存完了: Fig_特徴量順位_相関係数.png/pdf")

# =============================================================================
# 図2: 決定木重要度に基づく順位
# =============================================================================
def create_importance_ranking_figure():
    """重要度順位図（日本語）"""

    fig, ax = plt.subplots(figsize=(7, 6))

    imp_data = novel_results['importance_ranking']
    top_features = imp_data[:15]

    features = [f['feature'] for f in top_features]
    importance = [f['importance'] * 100 for f in top_features]
    is_novel = [f['is_novel'] for f in top_features]

    y_pos = np.arange(len(features))
    colors = [COLOR_NOVEL if n else COLOR_TRADITIONAL for n in is_novel]

    bars = ax.barh(y_pos, importance, height=0.7, color=colors,
                   edgecolor='black', linewidth=0.5, alpha=0.85)

    for i, imp in enumerate(importance):
        ax.text(imp + 0.3, i, f'{imp:.1f}%', va='center', ha='left',
                fontsize=8, fontweight='medium')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('特徴量重要度 (%)', fontweight='bold')
    ax.set_xlim(0, 25)

    legend_elements = [
        mpatches.Patch(facecolor=COLOR_NOVEL, edgecolor='black',
                       linewidth=0.5, label='新規特徴量'),
        mpatches.Patch(facecolor=COLOR_TRADITIONAL, edgecolor='black',
                       linewidth=0.5, label='従来特徴量')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
              fancybox=False, edgecolor='black')

    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)

    ax.set_title('(B) 決定木法による特徴量重要度の順位付け',
                 fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig('Fig_特徴量順位_重要度.png', dpi=300,
                facecolor='white', edgecolor='none')
    plt.savefig('Fig_特徴量順位_重要度.pdf', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
    print("保存完了: Fig_特徴量順位_重要度.png/pdf")

# =============================================================================
# 図3: 2パネル統合図
# =============================================================================
def create_combined_figure():
    """2パネル統合図（日本語）"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- (A) 相関係数順位 ---
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

    for i, (mean_c, max_c) in enumerate(zip(mean_corrs, max_corrs)):
        ax1.plot([mean_c, max_c], [i, i], 'k-', linewidth=1.2, alpha=0.6)
        ax1.plot(max_c, i, 'k|', markersize=5, markeredgewidth=1.2)

    for i, mean_c in enumerate(mean_corrs):
        ax1.text(mean_c + 0.01, i, f'{mean_c:.3f}', va='center', ha='left', fontsize=8)

    ax1.set_yticks(y_pos_corr)
    ax1.set_yticklabels(features_corr)
    ax1.invert_yaxis()
    ax1.set_xlabel('平均絶対相関係数', fontweight='bold')
    ax1.set_xlim(0, 0.85)
    ax1.set_axisbelow(True)
    ax1.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.set_title('(A) 関節角度との相関係数', fontweight='bold', pad=10)

    # --- (B) 重要度順位 ---
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

    for i, imp in enumerate(importance):
        ax2.text(imp + 0.2, i, f'{imp:.1f}%', va='center', ha='left', fontsize=8)

    ax2.set_yticks(y_pos_imp)
    ax2.set_yticklabels(features_imp)
    ax2.invert_yaxis()
    ax2.set_xlabel('特徴量重要度 (%)', fontweight='bold')
    ax2.set_xlim(0, 25)
    ax2.set_axisbelow(True)
    ax2.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax2.set_title('(B) 決定木法による特徴量重要度', fontweight='bold', pad=10)

    # 共通凡例
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_NOVEL, edgecolor='black',
                       linewidth=0.5, label='新規特徴量'),
        mpatches.Patch(facecolor=COLOR_TRADITIONAL, edgecolor='black',
                       linewidth=0.5, label='従来特徴量'),
        plt.Line2D([0], [0], color='black', linewidth=1.2,
                   label='最大相関係数')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 0.02), frameon=True, fancybox=False,
               edgecolor='black')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    plt.savefig('Fig_特徴量順位_統合.png', dpi=300,
                facecolor='white', edgecolor='none')
    plt.savefig('Fig_特徴量順位_統合.pdf', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
    print("保存完了: Fig_特徴量順位_統合.png/pdf")

# =============================================================================
# 図4: 包括的分析図（5パネル）
# =============================================================================
def create_comprehensive_figure():
    """包括的5パネル分析図（日本語）"""

    fig = plt.figure(figsize=(10, 10))

    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 0.8],
                          hspace=0.35, wspace=0.3)

    # --- (A) 相関係数順位 ---
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
    ax1.set_xlabel('平均相関係数', fontsize=9)
    ax1.set_xlim(0, 0.55)
    ax1.set_axisbelow(True)
    ax1.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.set_title('(A) 相関係数による順位', fontweight='bold', fontsize=10)

    # --- (B) 重要度順位 ---
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
    ax2.set_xlabel('重要度 (%)', fontsize=9)
    ax2.set_xlim(0, 25)
    ax2.set_axisbelow(True)
    ax2.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax2.set_title('(B) 決定木法による重要度', fontweight='bold', fontsize=10)

    # --- (C) 相関係数と重要度の散布図 ---
    ax3 = fig.add_subplot(gs[1, 0])

    feature_dict = {}
    for item in corr_data:
        feature_dict[item['feature']] = {
            'corr': item['mean_correlation'],
            'is_novel': item['is_novel']
        }
    for item in imp_data:
        if item['feature'] in feature_dict:
            feature_dict[item['feature']]['imp'] = item['importance'] * 100

    scatter_data = [(k, v['corr'], v['imp'], v['is_novel'])
                    for k, v in feature_dict.items() if 'imp' in v]

    for name, corr, imp, is_novel in scatter_data:
        color = COLOR_NOVEL if is_novel else COLOR_TRADITIONAL
        ax3.scatter(corr, imp, c=color, s=60, edgecolors='black',
                    linewidth=0.5, alpha=0.8)
        if corr > 0.35 or imp > 8:
            ax3.annotate(name, (corr, imp), fontsize=7,
                        xytext=(3, 3), textcoords='offset points')

    ax3.set_xlabel('平均相関係数', fontsize=9)
    ax3.set_ylabel('特徴量重要度 (%)', fontsize=9)
    ax3.set_axisbelow(True)
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.set_title('(C) 相関係数と重要度の関係', fontweight='bold', fontsize=10)

    # --- (D) 新規特徴量と従来特徴量の比較 ---
    ax4 = fig.add_subplot(gs[1, 1])

    novel_vs_trad = novel_results['novel_vs_traditional']

    categories = ['新規\n特徴量', '従来\n特徴量']
    corr_values = [novel_vs_trad['novel_max_correlation'],
                   novel_vs_trad['traditional_max_correlation']]
    imp_values = [novel_vs_trad['novel_total_importance'] * 100,
                  novel_vs_trad['traditional_total_importance'] * 100]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax4.bar(x - width/2, corr_values, width, label='最大相関係数',
                    color='#3C5488', edgecolor='black', linewidth=0.5, alpha=0.85)
    bars2 = ax4.bar(x + width/2, [v/100 for v in imp_values], width,
                    label='総重要度',
                    color='#E64B35', edgecolor='black', linewidth=0.5, alpha=0.85)

    ax4.set_ylabel('値', fontsize=9)
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, fontsize=9)
    ax4.legend(fontsize=8, loc='upper right')
    ax4.set_ylim(0, 0.85)
    ax4.set_axisbelow(True)
    ax4.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax4.set_title('(D) 新規特徴量と従来特徴量の比較', fontweight='bold', fontsize=10)

    for bar, val in zip(bars1, corr_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, [v/100 for v in imp_values]):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    # --- (E) 推奨特徴量 ---
    ax5 = fig.add_subplot(gs[2, :])

    recommended = novel_results['recommendations']['best_combined']

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

    ax5.bar(x - width/2, rec_corr, width, label='相関係数',
            color=colors_rec, edgecolor='black', linewidth=0.5, alpha=0.6)
    ax5.bar(x + width/2, [v/50 for v in rec_imp], width, label='重要度（尺度調整済）',
            color=colors_rec, edgecolor='black', linewidth=0.5, alpha=0.9,
            hatch='///')

    ax5.set_ylabel('値', fontsize=9)
    ax5.set_xticks(x)
    ax5.set_xticklabels(recommended, fontsize=9, rotation=30, ha='right')
    ax5.legend(fontsize=8, loc='upper right')
    ax5.set_axisbelow(True)
    ax5.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax5.set_title('(E) 推奨特徴量', fontweight='bold', fontsize=10)

    # 凡例
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_NOVEL, edgecolor='black',
                       linewidth=0.5, label='新規特徴量'),
        mpatches.Patch(facecolor=COLOR_TRADITIONAL, edgecolor='black',
                       linewidth=0.5, label='従来特徴量')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.02), frameon=True, fancybox=False,
               edgecolor='black', fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    plt.savefig('Fig_特徴量分析_包括図.png', dpi=300,
                facecolor='white', edgecolor='none')
    plt.savefig('Fig_特徴量分析_包括図.pdf', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
    print("保存完了: Fig_特徴量分析_包括図.png/pdf")

# =============================================================================
# 図5: 簡易順位図（論文向け推奨）
# =============================================================================
def create_simple_ranking_figure():
    """簡易総合順位図（日本語）"""

    fig, ax = plt.subplots(figsize=(5, 7))

    corr_data = novel_results['correlation_ranking']
    imp_data = novel_results['importance_ranking']

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

    for feat, data in feature_scores.items():
        if 'imp_norm' in data:
            data['combined'] = 0.5 * data['corr_norm'] + 0.5 * data['imp_norm']
        else:
            data['combined'] = data['corr_norm'] * 0.5

    sorted_features = sorted(feature_scores.items(),
                            key=lambda x: x[1]['combined'], reverse=True)

    top_features = sorted_features[:15]

    names = [f[0] for f in top_features]
    combined_scores = [f[1]['combined'] for f in top_features]
    is_novel = [f[1]['is_novel'] for f in top_features]

    y_pos = np.arange(len(names))
    colors = [COLOR_NOVEL if n else COLOR_TRADITIONAL for n in is_novel]

    bars = ax.barh(y_pos, combined_scores, height=0.65, color=colors,
                   edgecolor='black', linewidth=0.5, alpha=0.85)

    for i, (name, score) in enumerate(zip(names, combined_scores)):
        ax.text(-0.03, i, f'{i+1}', va='center', ha='right', fontsize=9,
                fontweight='bold', color='#333333')
        ax.text(score + 0.015, i, f'{score:.3f}', va='center', ha='left', fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('総合評価値\n（相関係数＋重要度）', fontsize=9, fontweight='bold')
    ax.set_xlim(-0.08, 1.15)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)

    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)

    legend_elements = [
        mpatches.Patch(facecolor=COLOR_NOVEL, edgecolor='black',
                       linewidth=0.5, label='新規特徴量'),
        mpatches.Patch(facecolor=COLOR_TRADITIONAL, edgecolor='black',
                       linewidth=0.5, label='従来特徴量')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
              fancybox=False, edgecolor='black', fontsize=8)

    ax.set_title('筋電特徴量の総合順位',
                 fontweight='bold', fontsize=11, pad=10)

    plt.tight_layout()

    plt.savefig('Fig_特徴量順位_簡易図.png', dpi=300,
                facecolor='white', edgecolor='none')
    plt.savefig('Fig_特徴量順位_簡易図.pdf', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
    print("保存完了: Fig_特徴量順位_簡易図.png/pdf")

# =============================================================================
# 表データ出力
# =============================================================================
def create_summary_table():
    """論文用表データ"""

    corr_data = novel_results['correlation_ranking']
    imp_data = novel_results['importance_ranking']

    print("\n" + "="*80)
    print("表: 特徴量の順位付け結果（論文用）")
    print("="*80)

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

    sorted_data = sorted(feature_dict.items(), key=lambda x: x[1]['rank_corr'])

    print("\n特徴量\t\t種類\t\t相関順位\t平均相関\t重要度順位\t重要度(%)")
    print("-" * 90)

    for name, data in sorted_data[:15]:
        feat_type = "新規" if data['is_novel'] else "従来"
        imp_rank = data.get('rank_imp', '-')
        imp_val = f"{data.get('imp', 0):.2f}" if 'imp' in data else '-'

        name_padded = name.ljust(12)
        print(f"{name_padded}\t{feat_type}\t\t{data['rank_corr']}\t\t{data['corr']:.4f}\t\t{imp_rank}\t\t{imp_val}")

    print("\n" + "="*80)

# =============================================================================
# 実行
# =============================================================================
if __name__ == '__main__':
    print("論文品質の図を作成中（日本語版）...")
    print("="*60)

    create_correlation_ranking_figure()
    create_importance_ranking_figure()
    create_combined_figure()
    create_comprehensive_figure()
    create_simple_ranking_figure()
    create_summary_table()

    print("\n" + "="*60)
    print("全ての図を正常に作成しました")
    print("="*60)
    print("\n生成ファイル:")
    print("  - Fig_特徴量順位_相関係数.png/pdf")
    print("  - Fig_特徴量順位_重要度.png/pdf")
    print("  - Fig_特徴量順位_統合.png/pdf")
    print("  - Fig_特徴量分析_包括図.png/pdf")
    print("  - Fig_特徴量順位_簡易図.png/pdf")
    print("\n論文投稿への推奨:")
    print("  - 単段組: Fig_特徴量順位_簡易図.pdf")
    print("  - 二段組: Fig_特徴量順位_統合.pdf")
    print("  - 全頁図: Fig_特徴量分析_包括図.pdf")
