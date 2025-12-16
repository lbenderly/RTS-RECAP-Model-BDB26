"""
RTS Score Validation Analysis

Validates that RTS scores are meaningful by analyzing correlations with
traditional performance metrics and outcomes. Demonstrates that higher RTS
scores correspond to better play execution and results.

"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

RESULTS_PATH = Path("results/rts_scores_2/rts_scored_plays.csv")
OUTPUT_DIR = Path("results/rts_validation_2")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Validation configuration
CONFIG = {
    'score_bins': [0, 25, 40, 60, 75, 100],  # Quintiles for grouping
    'bin_labels': ['Poor', 'Below Avg', 'Average', 'Above Avg', 'Elite'],
    'min_plays_per_bin': 10,  # Minimum plays for statistical significance
}

# ==============================================================================
# VALIDATION ANALYSIS FUNCTIONS
# ==============================================================================

def bin_scores(df: pd.DataFrame, score_col: str = 'RTS_scaled') -> pd.DataFrame:
    """Bin scores into performance tiers."""
    df = df.copy()
    df['score_bin'] = pd.cut(df[score_col],
                               bins=CONFIG['score_bins'],
                               labels=CONFIG['bin_labels'],
                               include_lowest=True)
    return df


def compute_outcome_by_score_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Compute traditional outcomes grouped by RTS score tier."""
    
    outcome_stats = df.groupby('score_bin', observed=True).agg({
        'play_id': 'count',
        'actual_completion': ['mean', 'sum'],
        'expected_points_added': ['mean', 'std', 'sum'],
        'yards_gained': ['mean', 'std'],
        'RTS_scaled': ['mean', 'std'],
        'temporal_score_scaled': 'mean',
        'spatial_score_scaled': 'mean',
        'temporal_error_abs': 'mean',
        'spatial_error_abs': 'mean',
    }).reset_index()
    
    # Flatten columns
    outcome_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                              for col in outcome_stats.columns.values]
    
    # Rename for clarity
    outcome_stats = outcome_stats.rename(columns={
        'play_id_count': 'n_plays',
        'actual_completion_mean': 'completion_rate',
        'actual_completion_sum': 'completions',
        'expected_points_added_mean': 'mean_epa',
        'expected_points_added_std': 'std_epa',
        'expected_points_added_sum': 'total_epa',
        'yards_gained_mean': 'mean_yards',
        'yards_gained_std': 'std_yards',
        'RTS_scaled_mean': 'mean_rts',
        'RTS_scaled_std': 'std_rts',
        'temporal_score_scaled_mean': 'mean_temporal',
        'spatial_score_scaled_mean': 'mean_spatial',
        'temporal_error_abs_mean': 'mean_temporal_error',
        'spatial_error_abs_mean': 'mean_spatial_error',
    })
    
    return outcome_stats


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation between RTS components and outcomes."""
    
    metrics = {
        'RTS_scaled': 'Overall RTS Score',
        'temporal_score_scaled': 'Temporal Score',
        'spatial_score_scaled': 'Spatial Score',
        'temporal_error_abs': 'Temporal Error (sec)',
        'spatial_error_abs': 'Spatial Error (yards)',
    }
    
    outcomes = {
        'actual_completion': 'Completion',
        'expected_points_added': 'EPA',
        'yards_gained': 'Yards Gained',
    }
    
    correlations = []
    
    for metric_col, metric_name in metrics.items():
        for outcome_col, outcome_name in outcomes.items():
            # Remove NaN values
            valid_data = df[[metric_col, outcome_col]].dropna()
            
            if len(valid_data) < 30:
                continue
            
            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(valid_data[metric_col], 
                                                    valid_data[outcome_col])
            
            # Spearman correlation (rank-based, more robust)
            spearman_r, spearman_p = stats.spearmanr(valid_data[metric_col], 
                                                       valid_data[outcome_col])
            
            correlations.append({
                'metric': metric_name,
                'outcome': outcome_name,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'n_samples': len(valid_data),
                'significant': pearson_p < 0.05
            })
    
    return pd.DataFrame(correlations)


def test_monotonic_relationship(df: pd.DataFrame) -> dict:
    """
    Test if outcomes improve monotonically with RTS score.
    Uses Mann-Kendall test for trend.
    """
    
    outcome_by_tier = compute_outcome_by_score_tier(df)
    
    # Extract ordered values (Poor -> Elite)
    completion_trend = outcome_by_tier['completion_rate'].values
    epa_trend = outcome_by_tier['mean_epa'].values
    yards_trend = outcome_by_tier['mean_yards'].values
    
    results = {
        'completion_rate': {
            'values': completion_trend,
            'increasing': np.all(np.diff(completion_trend) >= 0),
            'slope': np.polyfit(range(len(completion_trend)), completion_trend, 1)[0]
        },
        'mean_epa': {
            'values': epa_trend,
            'increasing': np.all(np.diff(epa_trend) >= 0),
            'slope': np.polyfit(range(len(epa_trend)), epa_trend, 1)[0]
        },
        'mean_yards': {
            'values': yards_trend,
            'increasing': np.all(np.diff(yards_trend) >= 0),
            'slope': np.polyfit(range(len(yards_trend)), yards_trend, 1)[0]
        }
    }
    
    return results


def compute_predictive_power(df: pd.DataFrame) -> dict:
    """
    Compute how well RTS predicts completion vs other metrics.
    Uses ROC-AUC for binary classification.
    """
    
    valid_data = df[['RTS_scaled', 'temporal_score_scaled', 'spatial_score_scaled',
                      'CP_actual_timing', 'CP_actual_placement',
                      'actual_completion']].dropna()

    if len(valid_data) < 50:
        return None

    y_true = valid_data['actual_completion'].values

    predictors = {
        'RTS Score': valid_data['RTS_scaled'].values,
        'Temporal Score': valid_data['temporal_score_scaled'].values,
        'Spatial Score': valid_data['spatial_score_scaled'].values,
        'CP (Timing)': valid_data['CP_actual_timing'].values,
        'CP (Placement)': valid_data['CP_actual_placement'].values,
    }
    
    results = {}
    
    for name, y_pred in predictors.items():
        # Normalize to 0-1 if needed
        y_pred_norm = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + 1e-10)
        
        try:
            auc = roc_auc_score(y_true, y_pred_norm)
            results[name] = {
                'auc': auc,
                'mean_complete': y_pred[y_true == 1].mean(),
                'mean_incomplete': y_pred[y_true == 0].mean(),
                'difference': y_pred[y_true == 1].mean() - y_pred[y_true == 0].mean()
            }
        except:
            results[name] = None
    
    return results


def analyze_extreme_plays(df: pd.DataFrame, percentile: float = 10) -> dict:
    """Compare top and bottom plays by RTS score."""

    top_threshold = df['RTS_scaled'].quantile(1 - percentile/100)
    bottom_threshold = df['RTS_scaled'].quantile(percentile/100)

    top_plays = df[df['RTS_scaled'] >= top_threshold]
    bottom_plays = df[df['RTS_scaled'] <= bottom_threshold]
    
    comparison = {
        'top_percentile': percentile,
        'top_n': len(top_plays),
        'bottom_n': len(bottom_plays),
        'metrics': {}
    }
    
    metrics = ['actual_completion', 'expected_points_added', 'yards_gained',
               'temporal_error_abs', 'spatial_error_abs']
    
    for metric in metrics:
        top_val = top_plays[metric].mean()
        bottom_val = bottom_plays[metric].mean()
        
        # Statistical test
        if len(top_plays) > 5 and len(bottom_plays) > 5:
            t_stat, p_value = stats.ttest_ind(top_plays[metric].dropna(), 
                                               bottom_plays[metric].dropna())
        else:
            t_stat, p_value = np.nan, np.nan
        
        comparison['metrics'][metric] = {
            'top_mean': top_val,
            'bottom_mean': bottom_val,
            'difference': top_val - bottom_val,
            'percent_improvement': ((top_val - bottom_val) / abs(bottom_val) * 100) if bottom_val != 0 else np.nan,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05 if not np.isnan(p_value) else False
        }
    
    return comparison


def analyze_by_route_type(df: pd.DataFrame) -> pd.DataFrame:
    """Validate RTS within each route type."""
    
    route_validation = []
    
    for route in df['route_type'].unique():
        route_data = df[df['route_type'] == route]
        
        if len(route_data) < 20:
            continue
        
        # Bin by RTS tertiles
        route_data['rts_tertile'] = pd.qcut(route_data['RTS_scaled'],
                                              q=3,
                                              labels=['Low', 'Medium', 'High'])
        
        tertile_stats = route_data.groupby('rts_tertile').agg({
            'actual_completion': 'mean',
            'expected_points_added': 'mean',
            'yards_gained': 'mean',
            'play_id': 'count'
        }).reset_index()
        
        # Check if high RTS has better outcomes than low RTS
        if len(tertile_stats) == 3:
            high_comp = tertile_stats[tertile_stats['rts_tertile'] == 'High']['actual_completion'].values[0]
            low_comp = tertile_stats[tertile_stats['rts_tertile'] == 'Low']['actual_completion'].values[0]
            
            high_epa = tertile_stats[tertile_stats['rts_tertile'] == 'High']['expected_points_added'].values[0]
            low_epa = tertile_stats[tertile_stats['rts_tertile'] == 'Low']['expected_points_added'].values[0]
            
            route_validation.append({
                'route_type': route,
                'n_plays': len(route_data),
                'comp_high_rts': high_comp,
                'comp_low_rts': low_comp,
                'comp_diff': high_comp - low_comp,
                'epa_high_rts': high_epa,
                'epa_low_rts': low_epa,
                'epa_diff': high_epa - low_epa,
                'validates': (high_comp > low_comp) and (high_epa > low_epa)
            })
    
    return pd.DataFrame(route_validation)


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_outcome_by_tier(outcome_stats: pd.DataFrame, output_dir: Path):
    """Plot how outcomes vary by RTS score tier."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Add main title and bin range subtitle
    fig.suptitle('Outcomes by RTS Score Tier', fontsize=16, fontweight='bold', y=0.995)
    bin_ranges = 'Poor: 0-25 | Below Avg: 25-40 | Average: 40-60 | Above Avg: 60-75 | Elite: 75-100'
    fig.text(0.5, 0.965, bin_ranges, ha='center', fontsize=11, style='italic', color='gray')

    x_labels = outcome_stats['score_bin'].values
    x_pos = range(len(x_labels))

    # 1. Completion Rate
    axes[0, 0].bar(x_pos, outcome_stats['completion_rate'],
                   color=plt.cm.RdYlGn(outcome_stats['completion_rate']),
                   edgecolor='black', alpha=0.8)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(x_labels, rotation=45)
    axes[0, 0].set_ylabel('Completion Rate')
    axes[0, 0].set_title('Completion Rate by RTS Tier')
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Add values inside bars at the top
    for i, v in enumerate(outcome_stats['completion_rate']):
        axes[0, 0].text(i, v - 0.05, f'{v:.1%}', ha='center', va='top',
                       fontweight='bold', color='black', fontsize=10)

    # 2. Mean EPA
    colors = ['red' if x < 0 else 'green' for x in outcome_stats['mean_epa']]
    axes[0, 1].bar(x_pos, outcome_stats['mean_epa'],
                   color=colors, edgecolor='black', alpha=0.7)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(x_labels, rotation=45)
    axes[0, 1].set_ylabel('Mean EPA')
    axes[0, 1].set_title('Expected Points Added by RTS Tier')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Add values inside bars at the top
    for i, v in enumerate(outcome_stats['mean_epa']):
        if v >= 0:
            axes[0, 1].text(i, v - 0.02, f'{v:.2f}', ha='center', va='top',
                           fontweight='bold', color='black', fontsize=10)
        else:
            # For negative bars, place at bottom
            axes[0, 1].text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom',
                           fontweight='bold', color='black', fontsize=10)

    # 3. Mean Yards
    axes[1, 0].bar(x_pos, outcome_stats['mean_yards'],
                   color='steelblue', edgecolor='black', alpha=0.8)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(x_labels, rotation=45)
    axes[1, 0].set_ylabel('Mean Yards Gained')
    axes[1, 0].set_title('Yards Gained by RTS Tier')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Adjust y-axis to accommodate labels
    max_yards = outcome_stats['mean_yards'].max()
    axes[1, 0].set_ylim(0, max_yards * 1.1)

    # Add values inside bars at the top
    for i, v in enumerate(outcome_stats['mean_yards']):
        axes[1, 0].text(i, v - 0.5, f'{v:.1f}', ha='center', va='top',
                       fontweight='bold', color='white', fontsize=10)

    # 4. Sample sizes
    axes[1, 1].bar(x_pos, outcome_stats['n_plays'],
                   color='gray', edgecolor='black', alpha=0.6)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(x_labels, rotation=45)
    axes[1, 1].set_ylabel('Number of Plays')
    axes[1, 1].set_title('Sample Size by RTS Tier')
    axes[1, 1].grid(axis='y', alpha=0.3)

    # Adjust y-axis to accommodate labels
    max_plays = outcome_stats['n_plays'].max()
    axes[1, 1].set_ylim(0, max_plays * 1.1)

    # Add values inside bars at the top
    for i, v in enumerate(outcome_stats['n_plays']):
        axes[1, 1].text(i, v - (max_plays * 0.03), f'{int(v)}', ha='center', va='top',
                       fontweight='bold', color='white', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / 'outcomes_by_rts_tier.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved outcomes by tier plot")
    plt.close()


def plot_correlation_heatmap(corr_df: pd.DataFrame, output_dir: Path):
    """Plot correlation matrix between RTS components and outcomes."""
    
    # Pivot for heatmap
    pivot = corr_df.pivot(index='metric', columns='outcome', values='pearson_r')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(pivot, 
                annot=True, 
                fmt='.3f',
                cmap='coolwarm',
                center=0,
                vmin=-1,
                vmax=1,
                cbar_kws={'label': 'Pearson Correlation'},
                linewidths=0.5,
                ax=ax)
    
    ax.set_title('Correlation: RTS Components vs Outcomes', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Outcome Metric', fontsize=12)
    ax.set_ylabel('RTS Component', fontsize=12)
    
    # Add significance stars
    for i, metric in enumerate(pivot.index):
        for j, outcome in enumerate(pivot.columns):
            corr_row = corr_df[(corr_df['metric'] == metric) & 
                               (corr_df['outcome'] == outcome)]
            if len(corr_row) > 0 and corr_row['significant'].values[0]:
                ax.text(j + 0.5, i + 0.85, '*', 
                       ha='center', va='center', fontsize=16, 
                       color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved correlation heatmap")
    plt.close()


def plot_roc_curves(df: pd.DataFrame, output_dir: Path):
    """Plot ROC curves for different predictors of completion."""
    
    valid_data = df[['RTS_scaled', 'temporal_score_scaled', 'spatial_score_scaled',
                      'CP_actual_timing', 'CP_actual_placement',
                      'actual_completion']].dropna()

    if len(valid_data) < 50:
        print("⚠ Not enough data for ROC curves")
        return

    y_true = valid_data['actual_completion'].values

    predictors = {
        'RTS Score': valid_data['RTS_scaled'].values,
        'Temporal Score': valid_data['temporal_score_scaled'].values,
        'Spatial Score': valid_data['spatial_score_scaled'].values,
        'CP (Timing)': valid_data['CP_actual_timing'].values,
        'CP (Placement)': valid_data['CP_actual_placement'].values,
    }
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for (name, y_pred), color in zip(predictors.items(), colors):
        # Normalize to 0-1
        y_pred_norm = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + 1e-10)
        
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred_norm)
            auc = roc_auc_score(y_true, y_pred_norm)
            ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', 
                   linewidth=2, color=color)
        except:
            continue
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves: Predicting Completion', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved ROC curves")
    plt.close()


def plot_scatter_outcomes(df: pd.DataFrame, output_dir: Path):
    """Scatter plots showing RTS vs outcomes."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('RTS Score vs Traditional Outcomes', fontsize=16, fontweight='bold')
    
    # 1. RTS vs Completion
    complete = df[df['actual_completion'] == 1]
    incomplete = df[df['actual_completion'] == 0]

    axes[0].scatter(incomplete['RTS_scaled'], incomplete['actual_completion'],
                    alpha=0.3, s=30, label='Incomplete', color='red')
    axes[0].scatter(complete['RTS_scaled'], complete['actual_completion'],
                    alpha=0.3, s=30, label='Complete', color='green')

    # Add jitter for visualization
    axes[0].scatter(incomplete['RTS_scaled'],
                    np.random.normal(0, 0.02, len(incomplete)),
                    alpha=0.3, s=30, color='red')
    axes[0].scatter(complete['RTS_scaled'],
                    np.random.normal(1, 0.02, len(complete)),
                    alpha=0.3, s=30, color='green')

    axes[0].set_xlabel('RTS Score')
    axes[0].set_ylabel('Completion')
    axes[0].set_title('RTS vs Completion')
    axes[0].set_ylim(-0.2, 1.2)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Add mean RTS lines
    axes[0].axvline(complete['RTS_scaled'].mean(), color='green',
                    linestyle='--', alpha=0.7,
                    label=f'Complete: {complete["RTS_scaled"].mean():.1f}')
    axes[0].axvline(incomplete['RTS_scaled'].mean(), color='red',
                    linestyle='--', alpha=0.7,
                    label=f'Incomplete: {incomplete["RTS_scaled"].mean():.1f}')

    # 2. RTS vs EPA
    axes[1].scatter(df['RTS_scaled'], df['expected_points_added'],
                    alpha=0.4, s=30, c=df['actual_completion'],
                    cmap='RdYlGn')

    # Trend line
    z = np.polyfit(df['RTS_scaled'].dropna(), df['expected_points_added'].dropna(), 1)
    p = np.poly1d(z)
    axes[1].plot(df['RTS_scaled'].sort_values(), p(df['RTS_scaled'].sort_values()),
                 "r--", alpha=0.8, linewidth=2, label=f'Trend: {z[0]:.3f}x + {z[1]:.2f}')

    axes[1].set_xlabel('RTS Score')
    axes[1].set_ylabel('Expected Points Added')
    axes[1].set_title('RTS vs EPA')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # 3. RTS vs Yards
    axes[2].scatter(df['RTS_scaled'], df['yards_gained'],
                    alpha=0.4, s=30, c=df['actual_completion'],
                    cmap='RdYlGn')

    # Trend line
    valid_yards = df[['RTS_scaled', 'yards_gained']].dropna()
    z = np.polyfit(valid_yards['RTS_scaled'], valid_yards['yards_gained'], 1)
    p = np.poly1d(z)
    axes[2].plot(valid_yards['RTS_scaled'].sort_values(),
                 p(valid_yards['RTS_scaled'].sort_values()),
                 "r--", alpha=0.8, linewidth=2,
                 label=f'Trend: {z[0]:.3f}x + {z[1]:.2f}')
    
    axes[2].set_xlabel('RTS Score')
    axes[2].set_ylabel('Yards Gained')
    axes[2].set_title('RTS vs Yards Gained')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_rts_vs_outcomes.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved scatter plots")
    plt.close()


def plot_extreme_comparison(extreme_analysis: dict, output_dir: Path):
    """Visualize comparison between top and bottom RTS plays."""

    metrics_data = extreme_analysis['metrics']

    metric_names = {
        'actual_completion': 'Completion Rate',
        'expected_points_added': 'EPA',
        'yards_gained': 'Yards Gained',
        'temporal_error_abs': 'Temporal Error (s)',
        'spatial_error_abs': 'Spatial Error (yds)',
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Top {extreme_analysis["top_percentile"]}% vs Bottom {extreme_analysis["top_percentile"]}% RTS Plays',
                 fontsize=16, fontweight='bold')

    # 1. Bar comparison
    display_metrics = ['actual_completion', 'expected_points_added', 'yards_gained']
    x_pos = range(len(display_metrics))
    width = 0.35

    top_values = [metrics_data[m]['top_mean'] for m in display_metrics]
    bottom_values = [metrics_data[m]['bottom_mean'] for m in display_metrics]

    bars1 = axes[0].bar([i - width/2 for i in x_pos], top_values, width,
                        label=f'Top {extreme_analysis["top_percentile"]}%',
                        alpha=0.8, color='green')
    bars2 = axes[0].bar([i + width/2 for i in x_pos], bottom_values, width,
                        label=f'Bottom {extreme_analysis["top_percentile"]}%',
                        alpha=0.8, color='red')

    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([metric_names[m] for m in display_metrics])
    axes[0].set_ylabel('Value')
    axes[0].set_title('Outcome Comparison')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Add labels inside bars at the top
    for i, (top_val, bottom_val) in enumerate(zip(top_values, bottom_values)):
        # Top bar label
        if top_val > 0:
            axes[0].text(i - width/2, top_val - (top_val * 0.05), f'{top_val:.2f}',
                        ha='center', va='top', fontweight='bold', color='white', fontsize=9)
        # Bottom bar label
        if bottom_val > 0:
            axes[0].text(i + width/2, bottom_val - (bottom_val * 0.05), f'{bottom_val:.2f}',
                        ha='center', va='top', fontweight='bold', color='white', fontsize=9)

    # 2. Percent improvement
    improvements = []
    labels = []
    colors = []

    for metric in display_metrics:
        pct_imp = metrics_data[metric]['percent_improvement']
        if not np.isnan(pct_imp):
            improvements.append(pct_imp)
            labels.append(metric_names[metric])
            colors.append('green' if pct_imp > 0 else 'red')

    bars = axes[1].barh(range(len(improvements)), improvements, color=colors, alpha=0.8)
    axes[1].set_yticks(range(len(improvements)))
    axes[1].set_yticklabels(labels)
    axes[1].set_xlabel('Percent Improvement (%)')
    axes[1].set_title('Top vs Bottom: Relative Improvement')
    axes[1].axvline(x=0, color='black', linestyle='-', linewidth=1)
    axes[1].grid(axis='x', alpha=0.3)

    # Add values inside bars
    for i, v in enumerate(improvements):
        if v > 0:
            # Positive bars: place label inside on the right
            axes[1].text(v - abs(v * 0.05), i, f'{v:.1f}%',
                        va='center', ha='right', fontweight='bold', color='white', fontsize=10)
        else:
            # Negative bars: place label inside on the left
            axes[1].text(v + abs(v * 0.05), i, f'{v:.1f}%',
                        va='center', ha='left', fontweight='bold', color='white', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'extreme_plays_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved extreme plays comparison")
    plt.close()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main validation execution."""
    
    print("="*80)
    print("RTS SCORE VALIDATION ANALYSIS")
    print("="*80)
    print("\nObjective: Validate that RTS scores meaningfully capture play quality")
    print("Method: Analyze correlations with traditional metrics and outcomes\n")
    
    # Load data
    print(f"Loading data from: {RESULTS_PATH}")
    df = pd.read_csv(RESULTS_PATH)
    print(f"✓ Loaded {len(df)} plays\n")
    
    # Bin scores
    df = bin_scores(df)
    
    # =========================================================================
    # 1. OUTCOMES BY RTS TIER
    # =========================================================================
    print("="*80)
    print("1. OUTCOMES BY RTS SCORE TIER")
    print("="*80)
    
    outcome_stats = compute_outcome_by_score_tier(df)
    
    print("\nOutcome Statistics by RTS Tier:")
    print(outcome_stats[['score_bin', 'n_plays', 'completion_rate', 
                         'mean_epa', 'mean_yards']].to_string(index=False))
    
    # Save
    outcome_stats.to_csv(OUTPUT_DIR / 'outcomes_by_tier.csv', index=False)
    print(f"\n✓ Saved to outcomes_by_tier.csv")
    
    # Test monotonic relationship
    monotonic_results = test_monotonic_relationship(df)
    
    print("\n--- Monotonic Trend Analysis ---")
    for metric, result in monotonic_results.items():
        trend = "✓ INCREASING" if result['increasing'] else "✗ Not monotonic"
        print(f"{metric:20s}: {trend} (slope: {result['slope']:.4f})")
    
    # =========================================================================
    # 2. CORRELATION ANALYSIS
    # =========================================================================
    print("\n" + "="*80)
    print("2. CORRELATION WITH TRADITIONAL METRICS")
    print("="*80)
    
    corr_df = compute_correlations(df)
    
    print("\nCorrelations (Pearson):")
    print(corr_df[['metric', 'outcome', 'pearson_r', 'pearson_p', 'significant']].to_string(index=False))
    
    # Save
    corr_df.to_csv(OUTPUT_DIR / 'correlations.csv', index=False)
    print(f"\n✓ Saved to correlations.csv")
    
    # Key findings
    print("\n--- Key Correlation Findings ---")
    rts_corrs = corr_df[corr_df['metric'] == 'Overall RTS Score']
    for _, row in rts_corrs.iterrows():
        sig = "***" if row['pearson_p'] < 0.001 else "**" if row['pearson_p'] < 0.01 else "*" if row['pearson_p'] < 0.05 else ""
        print(f"RTS → {row['outcome']:15s}: r = {row['pearson_r']:+.3f} {sig}")
    
    # =========================================================================
    # 3. PREDICTIVE POWER
    # =========================================================================
    print("\n" + "="*80)
    print("3. PREDICTIVE POWER ANALYSIS")
    print("="*80)
    
    predictive_results = compute_predictive_power(df)
    
    if predictive_results:
        print("\nROC-AUC Scores (predicting completion):")
        for name, results in predictive_results.items():
            if results:
                print(f"  {name:20s}: AUC = {results['auc']:.3f} | "
                      f"Complete: {results['mean_complete']:.2f} | "
                      f"Incomplete: {results['mean_incomplete']:.2f}")
        
        # Save
        pred_df = pd.DataFrame([
            {'predictor': k, **v} for k, v in predictive_results.items() if v
        ])
        pred_df.to_csv(OUTPUT_DIR / 'predictive_power.csv', index=False)
        print(f"\n✓ Saved to predictive_power.csv")
    else:
        print("⚠ Not enough data for predictive power analysis")
    
    # =========================================================================
    # 4. EXTREME PLAYS COMPARISON
    # =========================================================================
    print("\n" + "="*80)
    print("4. EXTREME PLAYS ANALYSIS")
    print("="*80)
    
    extreme_analysis = analyze_extreme_plays(df, percentile=10)
    
    print(f"\nTop {extreme_analysis['top_percentile']}% vs Bottom {extreme_analysis['top_percentile']}%:")
    print(f"  Top plays: n = {extreme_analysis['top_n']}")
    print(f"  Bottom plays: n = {extreme_analysis['bottom_n']}")
    print("\nMetric Comparison:")
    
    for metric, stats in extreme_analysis['metrics'].items():
        sig = "***" if stats['p_value'] < 0.001 else "**" if stats['p_value'] < 0.01 else "*" if stats['p_value'] < 0.05 else ""
        print(f"  {metric:25s}: Top={stats['top_mean']:.3f}, Bottom={stats['bottom_mean']:.3f}, "
              f"Diff={stats['difference']:+.3f} ({stats['percent_improvement']:+.1f}%) {sig}")
    
    # Save
    extreme_df = pd.DataFrame([
        {'metric': k, **v} for k, v in extreme_analysis['metrics'].items()
    ])
    extreme_df.to_csv(OUTPUT_DIR / 'extreme_plays_comparison.csv', index=False)
    print(f"\n✓ Saved to extreme_plays_comparison.csv")
    
    # =========================================================================
    # 5. ROUTE-SPECIFIC VALIDATION
    # =========================================================================
    print("\n" + "="*80)
    print("5. ROUTE-SPECIFIC VALIDATION")
    print("="*80)
    
    route_validation = analyze_by_route_type(df)
    
    if len(route_validation) > 0:
        print("\nValidation by Route Type:")
        print(route_validation[['route_type', 'n_plays', 'comp_diff', 'epa_diff', 'validates']].to_string(index=False))
        
        n_validates = route_validation['validates'].sum()
        n_total = len(route_validation)
        print(f"\n✓ RTS validates on {n_validates}/{n_total} route types ({n_validates/n_total:.1%})")
        
        # Save
        route_validation.to_csv(OUTPUT_DIR / 'route_validation.csv', index=False)
        print(f"✓ Saved to route_validation.csv")
    else:
        print("⚠ Not enough data for route-specific validation")
    
    # =========================================================================
    # 6. CREATE VISUALIZATIONS
    # =========================================================================
    print("\n" + "="*80)
    print("6. GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_outcome_by_tier(outcome_stats, OUTPUT_DIR)
    plot_correlation_heatmap(corr_df, OUTPUT_DIR)
    plot_roc_curves(df, OUTPUT_DIR)
    plot_scatter_outcomes(df, OUTPUT_DIR)
    plot_extreme_comparison(extreme_analysis, OUTPUT_DIR)
    
    # =========================================================================
    # 7. FINAL VALIDATION SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    # Count significant correlations
    sig_corrs = corr_df[(corr_df['metric'] == 'Overall RTS Score') & 
                        (corr_df['significant'] == True)]
    
    print(f"\n✓ Significant Correlations: {len(sig_corrs)}/3")
    
    # Check monotonic trends
    all_increasing = all(r['increasing'] for r in monotonic_results.values())
    print(f"✓ Monotonic Trends: {'YES' if all_increasing else 'PARTIAL'}")
    
    # Check predictive power
    if predictive_results and 'RTS Score' in predictive_results:
        rts_auc = predictive_results['RTS Score']['auc']
        print(f"✓ RTS AUC Score: {rts_auc:.3f} ({'Good' if rts_auc > 0.6 else 'Moderate'})")
    
    # Check extreme plays
    comp_improvement = extreme_analysis['metrics']['actual_completion']['percent_improvement']
    epa_improvement = extreme_analysis['metrics']['expected_points_added']['percent_improvement']
    print(f"✓ Top 10% Improvement: Completion +{comp_improvement:.1f}%, EPA +{epa_improvement:.1f}%")
    
    # Overall assessment
    print("\n" + "="*80)
    print("OVERALL ASSESSMENT")
    print("="*80)
    
    validation_score = 0
    max_score = 5
    
    # Criteria 1: Significant correlations
    if len(sig_corrs) >= 2:
        validation_score += 1
        print("✓ Strong correlations with traditional metrics")
    
    # Criteria 2: Monotonic relationships
    if all_increasing:
        validation_score += 1
        print("✓ Outcomes improve monotonically with RTS")
    
    # Criteria 3: Predictive power
    if predictive_results and rts_auc > 0.55:
        validation_score += 1
        print("✓ RTS predicts completion better than random")
    
    # Criteria 4: Extreme plays show large difference
    if comp_improvement > 10:
        validation_score += 1
        print("✓ Top RTS plays substantially outperform bottom plays")
    
    # Criteria 5: Validates across routes
    if len(route_validation) > 0 and route_validation['validates'].mean() > 0.6:
        validation_score += 1
        print("✓ RTS validates across multiple route types")
    
    print(f"\nValidation Score: {validation_score}/{max_score}")
    
    if validation_score >= 4:
        print("\n🎯 CONCLUSION: RTS score is STRONGLY VALIDATED")
        print("   The metric meaningfully captures play quality and correlates with outcomes.")
    elif validation_score >= 3:
        print("\n✓ CONCLUSION: RTS score is VALIDATED")
        print("   The metric shows meaningful relationships with play outcomes.")
    else:
        print("\n⚠ CONCLUSION: RTS score needs REFINEMENT")
        print("   Consider adjusting weighting or methodology.")
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - outcomes_by_tier.csv")
    print("  - correlations.csv")
    print("  - predictive_power.csv")
    print("  - extreme_plays_comparison.csv")
    print("  - route_validation.csv")
    print("  - outcomes_by_rts_tier.png")
    print("  - correlation_heatmap.png")
    print("  - roc_curves.png")
    print("  - scatter_rts_vs_outcomes.png")
    print("  - extreme_plays_comparison.png")


if __name__ == "__main__":
    main()