"""
RTS (Route Timing Synchronization) Score Post-Processing

This script loads the raw RSS evaluation results and computes percentile-based
RTS scores from temporal and spatial errors.

"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Path to your results file
RESULTS_PATH = Path("results/rts_evaluation/evaluation_results.csv")  # UPDATE THIS
OUTPUT_DIR = Path("results/rts_scores")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Scoring configuration
SCORING_CONFIG = {
    'temporal_weight': 0.5,      # Weight for temporal component (0-1)
    'spatial_weight': 0.5,       # Weight for spatial component (0-1)
    'route_specific': True,     # Whether to score within route types
    'absolute_errors': True,     # Use absolute values of errors
}

# ==============================================================================
# CORE RTS SCORING FUNCTIONS
# ==============================================================================

def compute_rts_scores(
    df: pd.DataFrame,
    temporal_col: str = 'temporal_error_seconds',
    spatial_col: str = 'spatial_error',
    temporal_weight: float = 0.5,
    spatial_weight: float = 0.5,
    group_by: str = None
) -> pd.DataFrame:
    """
    Compute percentile-based RTS scores.
    
    Args:
        df: DataFrame with temporal and spatial errors
        temporal_col: Column name for temporal error (seconds)
        spatial_col: Column name for spatial error (yards)
        temporal_weight: Weight for temporal component (0-1)
        spatial_weight: Weight for spatial component (0-1)
        group_by: Optional column to group by (e.g., 'route_type' for route-specific scoring)
    
    Returns:
        DataFrame with added RTS score columns
    """
    df = df.copy()
    
    # Ensure weights sum to 1
    total_weight = temporal_weight + spatial_weight
    temporal_weight = temporal_weight / total_weight
    spatial_weight = spatial_weight / total_weight
    
    # Use absolute values of errors
    df['temporal_error_abs'] = df[temporal_col].abs()
    df['spatial_error_abs'] = df[spatial_col].abs()
    
    if group_by is not None:
        # Route-specific scoring
        print(f"Computing route-specific RTS scores (grouped by {group_by})...")
        
        def score_group(group):
            # Compute percentiles within this group (lower error = lower percentile)
            group['temporal_percentile'] = group['temporal_error_abs'].rank(pct=True) * 100
            group['spatial_percentile'] = group['spatial_error_abs'].rank(pct=True) * 100
            
            # Invert to scores (lower error = higher score)
            group['temporal_score'] = 100 - group['temporal_percentile']
            group['spatial_score'] = 100 - group['spatial_percentile']
            
            # Combined RTS
            group['RTS'] = (temporal_weight * group['temporal_score'] + 
                           spatial_weight * group['spatial_score'])
            
            return group
        
        df = df.groupby(group_by, group_keys=False).apply(score_group)
        
    else:
        # Global scoring across all plays
        print("Computing global RTS scores...")
        
        # Compute percentiles (lower error = lower percentile)
        df['temporal_percentile'] = df['temporal_error_abs'].rank(pct=True) * 100
        df['spatial_percentile'] = df['spatial_error_abs'].rank(pct=True) * 100
        
        # Invert to scores (lower error = higher score)
        df['temporal_score'] = 100 - df['temporal_percentile']
        df['spatial_score'] = 100 - df['spatial_percentile']
        
        # Combined RTS
        df['RTS'] = (temporal_weight * df['temporal_score'] + 
                    spatial_weight * df['spatial_score'])
    
    return df


def compute_scaled_scores(
    df: pd.DataFrame,
    temporal_weight: float = 0.5,
    spatial_weight: float = 0.5,
    group_by: str = None
) -> pd.DataFrame:
    """
    Compute scaled scores by normalizing temporal and spatial scores to 0-100 range.

    This addresses the issue where temporal scores may not use the full 0-100 range,
    limiting the maximum achievable RTS score.

    Args:
        df: DataFrame with temporal_score and spatial_score columns
        temporal_weight: Weight for temporal component (0-1)
        spatial_weight: Weight for spatial component (0-1)
        group_by: Optional column to group by (e.g., 'route_type' for route-specific scaling)

    Returns:
        DataFrame with added scaled score columns
    """
    df = df.copy()

    # Ensure weights sum to 1
    total_weight = temporal_weight + spatial_weight
    temporal_weight = temporal_weight / total_weight
    spatial_weight = spatial_weight / total_weight

    if group_by is not None:
        # Route-specific scaling
        print(f"Computing route-specific scaled scores (grouped by {group_by})...")

        def scale_group(group):
            # Scale temporal score to 0-100 within this group
            temporal_min = group['temporal_score'].min()
            temporal_max = group['temporal_score'].max()
            temporal_range = temporal_max - temporal_min

            if temporal_range > 0:
                group['temporal_score_scaled'] = ((group['temporal_score'] - temporal_min) / temporal_range) * 100
            else:
                group['temporal_score_scaled'] = 50.0  # Default to middle if no variation

            # Scale spatial score to 0-100 within this group
            spatial_min = group['spatial_score'].min()
            spatial_max = group['spatial_score'].max()
            spatial_range = spatial_max - spatial_min

            if spatial_range > 0:
                group['spatial_score_scaled'] = ((group['spatial_score'] - spatial_min) / spatial_range) * 100
            else:
                group['spatial_score_scaled'] = 50.0  # Default to middle if no variation

            # Combined scaled RTS
            group['RTS_scaled'] = (temporal_weight * group['temporal_score_scaled'] +
                                   spatial_weight * group['spatial_score_scaled'])

            return group

        df = df.groupby(group_by, group_keys=False).apply(scale_group)

    else:
        # Global scaling across all plays
        print("Computing global scaled scores...")

        # Scale temporal score to 0-100
        temporal_min = df['temporal_score'].min()
        temporal_max = df['temporal_score'].max()
        temporal_range = temporal_max - temporal_min

        if temporal_range > 0:
            df['temporal_score_scaled'] = ((df['temporal_score'] - temporal_min) / temporal_range) * 100
        else:
            df['temporal_score_scaled'] = 50.0

        # Scale spatial score to 0-100
        spatial_min = df['spatial_score'].min()
        spatial_max = df['spatial_score'].max()
        spatial_range = spatial_max - spatial_min

        if spatial_range > 0:
            df['spatial_score_scaled'] = ((df['spatial_score'] - spatial_min) / spatial_range) * 100
        else:
            df['spatial_score_scaled'] = 50.0

        # Combined scaled RTS
        df['RTS_scaled'] = (temporal_weight * df['temporal_score_scaled'] +
                           spatial_weight * df['spatial_score_scaled'])

    return df


def add_performance_tiers(df: pd.DataFrame, score_col: str = 'RTS') -> pd.DataFrame:
    """Add performance tier labels based on RTS scores."""
    df = df.copy()

    def assign_tier(score):
        if pd.isna(score):
            return 'N/A'
        elif score >= 90:
            return 'Elite'
        elif score >= 75:
            return 'Excellent'
        elif score >= 60:
            return 'Above Average'
        elif score >= 40:
            return 'Average'
        elif score >= 25:
            return 'Below Average'
        else:
            return 'Poor'

    df['performance_tier'] = df[score_col].apply(assign_tier)

    # Also add tiers for scaled RTS if it exists
    if 'RTS_scaled' in df.columns:
        df['performance_tier_scaled'] = df['RTS_scaled'].apply(assign_tier)

    return df


# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================

def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics by passer and route type."""

    summary_stats = []

    # Overall stats
    overall = {
        'category': 'Overall',
        'subcategory': 'All Plays',
        'n_plays': len(df),
        'mean_RTS': df['RTS'].mean(),
        'median_RTS': df['RTS'].median(),
        'std_RTS': df['RTS'].std(),
        'mean_RTS_scaled': df['RTS_scaled'].mean(),
        'median_RTS_scaled': df['RTS_scaled'].median(),
        'std_RTS_scaled': df['RTS_scaled'].std(),
        'mean_temporal_score': df['temporal_score'].mean(),
        'mean_spatial_score': df['spatial_score'].mean(),
        'mean_temporal_score_scaled': df['temporal_score_scaled'].mean(),
        'mean_spatial_score_scaled': df['spatial_score_scaled'].mean(),
        'mean_temporal_error_sec': df['temporal_error_abs'].mean(),
        'mean_spatial_error_yards': df['spatial_error_abs'].mean(),
        'completion_rate': df['actual_completion'].mean(),
        'mean_epa': df['expected_points_added'].mean(),
    }
    summary_stats.append(overall)
    
    # By passer
    passer_stats = df.groupby('passer').agg({
        'RTS': ['count', 'mean', 'median', 'std'],
        'RTS_scaled': ['mean', 'median', 'std'],
        'temporal_score': 'mean',
        'spatial_score': 'mean',
        'temporal_score_scaled': 'mean',
        'spatial_score_scaled': 'mean',
        'temporal_error_abs': 'mean',
        'spatial_error_abs': 'mean',
        'actual_completion': 'mean',
        'expected_points_added': 'mean',
    }).round(2)

    for passer in passer_stats.index:
        summary_stats.append({
            'category': 'Passer',
            'subcategory': passer,
            'n_plays': passer_stats.loc[passer, ('RTS', 'count')],
            'mean_RTS': passer_stats.loc[passer, ('RTS', 'mean')],
            'median_RTS': passer_stats.loc[passer, ('RTS', 'median')],
            'std_RTS': passer_stats.loc[passer, ('RTS', 'std')],
            'mean_RTS_scaled': passer_stats.loc[passer, ('RTS_scaled', 'mean')],
            'median_RTS_scaled': passer_stats.loc[passer, ('RTS_scaled', 'median')],
            'std_RTS_scaled': passer_stats.loc[passer, ('RTS_scaled', 'std')],
            'mean_temporal_score': passer_stats.loc[passer, ('temporal_score', 'mean')],
            'mean_spatial_score': passer_stats.loc[passer, ('spatial_score', 'mean')],
            'mean_temporal_score_scaled': passer_stats.loc[passer, ('temporal_score_scaled', 'mean')],
            'mean_spatial_score_scaled': passer_stats.loc[passer, ('spatial_score_scaled', 'mean')],
            'mean_temporal_error_sec': passer_stats.loc[passer, ('temporal_error_abs', 'mean')],
            'mean_spatial_error_yards': passer_stats.loc[passer, ('spatial_error_abs', 'mean')],
            'completion_rate': passer_stats.loc[passer, ('actual_completion', 'mean')],
            'mean_epa': passer_stats.loc[passer, ('expected_points_added', 'mean')],
        })

    # By route type
    route_stats = df.groupby('route_type').agg({
        'RTS': ['count', 'mean', 'median', 'std'],
        'RTS_scaled': ['mean', 'median', 'std'],
        'temporal_score': 'mean',
        'spatial_score': 'mean',
        'temporal_score_scaled': 'mean',
        'spatial_score_scaled': 'mean',
        'temporal_error_abs': 'mean',
        'spatial_error_abs': 'mean',
        'actual_completion': 'mean',
        'expected_points_added': 'mean',
    }).round(2)

    for route in route_stats.index:
        summary_stats.append({
            'category': 'Route Type',
            'subcategory': route,
            'n_plays': route_stats.loc[route, ('RTS', 'count')],
            'mean_RTS': route_stats.loc[route, ('RTS', 'mean')],
            'median_RTS': route_stats.loc[route, ('RTS', 'median')],
            'std_RTS': route_stats.loc[route, ('RTS', 'std')],
            'mean_RTS_scaled': route_stats.loc[route, ('RTS_scaled', 'mean')],
            'median_RTS_scaled': route_stats.loc[route, ('RTS_scaled', 'median')],
            'std_RTS_scaled': route_stats.loc[route, ('RTS_scaled', 'std')],
            'mean_temporal_score': route_stats.loc[route, ('temporal_score', 'mean')],
            'mean_spatial_score': route_stats.loc[route, ('spatial_score', 'mean')],
            'mean_temporal_score_scaled': route_stats.loc[route, ('temporal_score_scaled', 'mean')],
            'mean_spatial_score_scaled': route_stats.loc[route, ('spatial_score_scaled', 'mean')],
            'mean_temporal_error_sec': route_stats.loc[route, ('temporal_error_abs', 'mean')],
            'mean_spatial_error_yards': route_stats.loc[route, ('spatial_error_abs', 'mean')],
            'completion_rate': route_stats.loc[route, ('actual_completion', 'mean')],
            'mean_epa': route_stats.loc[route, ('expected_points_added', 'mean')],
        })
    
    return pd.DataFrame(summary_stats)


def create_diagnostic_plots(df: pd.DataFrame, output_dir: Path):
    """Create diagnostic visualizations."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('RTS Score Analysis', fontsize=16, fontweight='bold')
    
    # 1. RTS Distribution
    axes[0, 0].hist(df['RTS'].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df['RTS'].mean(), color='red', linestyle='--', label=f'Mean: {df["RTS"].mean():.1f}')
    axes[0, 0].axvline(df['RTS'].median(), color='blue', linestyle='--', label=f'Median: {df["RTS"].median():.1f}')
    axes[0, 0].set_xlabel('RTS Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('RTS Score Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Temporal vs Spatial Scores
    axes[0, 1].scatter(df['temporal_score'], df['spatial_score'], alpha=0.5, s=20)
    axes[0, 1].plot([0, 100], [0, 100], 'r--', alpha=0.5, label='Equal Performance')
    axes[0, 1].set_xlabel('Temporal Score')
    axes[0, 1].set_ylabel('Spatial Score')
    axes[0, 1].set_title('Temporal vs Spatial Performance')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. RTS by Route Type
    route_data = df.groupby('route_type')['RTS'].mean().sort_values(ascending=False)
    axes[0, 2].barh(range(len(route_data)), route_data.values)
    axes[0, 2].set_yticks(range(len(route_data)))
    axes[0, 2].set_yticklabels(route_data.index)
    axes[0, 2].set_xlabel('Mean RTS Score')
    axes[0, 2].set_title('RTS by Route Type')
    axes[0, 2].grid(axis='x', alpha=0.3)
    
    # 4. Temporal Error Distribution
    axes[1, 0].hist(df['temporal_error_abs'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].axvline(df['temporal_error_abs'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["temporal_error_abs"].mean():.2f}s')
    axes[1, 0].set_xlabel('Temporal Error (seconds)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Temporal Error Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 5. Spatial Error Distribution
    axes[1, 1].hist(df['spatial_error_abs'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1, 1].axvline(df['spatial_error_abs'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["spatial_error_abs"].mean():.2f} yds')
    axes[1, 1].set_xlabel('Spatial Error (yards)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Spatial Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # 6. RTS vs Completion Rate
    axes[1, 2].scatter(df['RTS'], df['actual_completion'], alpha=0.5, s=20)
    axes[1, 2].set_xlabel('RTS Score')
    axes[1, 2].set_ylabel('Completion (1=Complete, 0=Incomplete)')
    axes[1, 2].set_title('RTS vs Completion Outcome')
    axes[1, 2].grid(alpha=0.3)
    
    # Add correlation
    corr = df[['RTS', 'actual_completion']].corr().iloc[0, 1]
    axes[1, 2].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=axes[1, 2].transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rts_diagnostic_plots.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved diagnostic plots to {output_dir / 'rts_diagnostic_plots.png'}")
    plt.close()


def create_passer_comparison(df: pd.DataFrame, output_dir: Path):
    """Create passer comparison visualizations."""
    
    # Get top passers by play count
    top_passers = df['passer'].value_counts().head(10).index
    df_top = df[df['passer'].isin(top_passers)]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Quarterback Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Overall RTS by Passer
    passer_rts = df_top.groupby('passer')['RTS'].mean().sort_values(ascending=True)
    axes[0].barh(range(len(passer_rts)), passer_rts.values)
    axes[0].set_yticks(range(len(passer_rts)))
    axes[0].set_yticklabels(passer_rts.index)
    axes[0].set_xlabel('Mean RTS Score')
    axes[0].set_title('Overall RTS Score by QB')
    axes[0].grid(axis='x', alpha=0.3)
    
    # 2. Temporal vs Spatial Breakdown
    passer_scores = df_top.groupby('passer')[['temporal_score', 'spatial_score', 'RTS']].mean()
    passer_scores = passer_scores.sort_values('RTS', ascending=True)
    
    x = range(len(passer_scores))
    width = 0.35
    
    axes[1].barh([i - width/2 for i in x], passer_scores['temporal_score'], width, 
                 label='Temporal Score', alpha=0.8)
    axes[1].barh([i + width/2 for i in x], passer_scores['spatial_score'], width, 
                 label='Spatial Score', alpha=0.8)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(passer_scores.index)
    axes[1].set_xlabel('Score')
    axes[1].set_title('Temporal vs Spatial Scores by QB')
    axes[1].legend()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'passer_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved passer comparison to {output_dir / 'passer_comparison.png'}")
    plt.close()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function."""
    
    print("="*80)
    print("RTS SCORE POST-PROCESSING")
    print("="*80)
    
    # Load results
    print(f"\nLoading results from: {RESULTS_PATH}")
    df = pd.read_csv(RESULTS_PATH)
    print(f"✓ Loaded {len(df)} plays")
    
    # Display basic stats
    print("\nRaw Error Statistics:")
    print(f"  Temporal error: {df['temporal_error_seconds'].abs().mean():.3f}s (mean)")
    print(f"  Spatial error: {df['spatial_error'].abs().mean():.2f} yards (mean)")
    
    # Compute RTS scores
    print("\n" + "="*80)
    print("COMPUTING RTS SCORES")
    print("="*80)

    if SCORING_CONFIG['route_specific']:
        df = compute_rts_scores(
            df,
            temporal_weight=SCORING_CONFIG['temporal_weight'],
            spatial_weight=SCORING_CONFIG['spatial_weight'],
            group_by='route_type'
        )
        print("✓ Route-specific RTS scores computed")
    else:
        df = compute_rts_scores(
            df,
            temporal_weight=SCORING_CONFIG['temporal_weight'],
            spatial_weight=SCORING_CONFIG['spatial_weight']
        )
        print("✓ Global RTS scores computed")

    # Compute scaled RTS scores
    print("\n" + "="*80)
    print("COMPUTING SCALED RTS SCORES")
    print("="*80)

    if SCORING_CONFIG['route_specific']:
        df = compute_scaled_scores(
            df,
            temporal_weight=SCORING_CONFIG['temporal_weight'],
            spatial_weight=SCORING_CONFIG['spatial_weight'],
            group_by='route_type'
        )
        print("✓ Route-specific scaled RTS scores computed")
    else:
        df = compute_scaled_scores(
            df,
            temporal_weight=SCORING_CONFIG['temporal_weight'],
            spatial_weight=SCORING_CONFIG['spatial_weight']
        )
        print("✓ Global scaled RTS scores computed")

    # Add performance tiers
    df = add_performance_tiers(df)
    print("✓ Performance tiers assigned")

    # Display RTS statistics
    print("\nOriginal RTS Score Statistics:")
    print(f"  Mean: {df['RTS'].mean():.2f}")
    print(f"  Median: {df['RTS'].median():.2f}")
    print(f"  Std Dev: {df['RTS'].std():.2f}")
    print(f"  Min: {df['RTS'].min():.2f}")
    print(f"  Max: {df['RTS'].max():.2f}")

    print("\nScaled RTS Score Statistics:")
    print(f"  Mean: {df['RTS_scaled'].mean():.2f}")
    print(f"  Median: {df['RTS_scaled'].median():.2f}")
    print(f"  Std Dev: {df['RTS_scaled'].std():.2f}")
    print(f"  Min: {df['RTS_scaled'].min():.2f}")
    print(f"  Max: {df['RTS_scaled'].max():.2f}")

    print("\nOriginal Performance Tier Distribution:")
    print(df['performance_tier'].value_counts().sort_index())

    print("\nScaled Performance Tier Distribution:")
    print(df['performance_tier_scaled'].value_counts().sort_index())
    
    # Generate summary statistics
    print("\n" + "="*80)
    print("GENERATING SUMMARY STATISTICS")
    print("="*80)
    
    summary_df = generate_summary_statistics(df)
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save scored dataset
    scored_path = OUTPUT_DIR / 'rts_scored_plays.csv'
    df.to_csv(scored_path, index=False)
    print(f"✓ Saved scored plays: {scored_path}")
    
    # Save summary statistics
    summary_path = OUTPUT_DIR / 'rts_summary_statistics.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved summary statistics: {summary_path}")
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    create_diagnostic_plots(df, OUTPUT_DIR)
    create_passer_comparison(df, OUTPUT_DIR)
    
    # Top/Bottom plays (Original RTS)
    print("\n" + "="*80)
    print("TOP 10 PLAYS BY ORIGINAL RTS")
    print("="*80)
    top_plays = df.nlargest(10, 'RTS')[['game_id', 'play_id', 'passer', 'targeted_receiver',
                                          'route_type', 'RTS', 'temporal_score', 'spatial_score',
                                          'actual_completion']]
    print(top_plays.to_string(index=False))

    print("\n" + "="*80)
    print("BOTTOM 10 PLAYS BY ORIGINAL RTS")
    print("="*80)
    bottom_plays = df.nsmallest(10, 'RTS')[['game_id', 'play_id', 'passer', 'targeted_receiver',
                                              'route_type', 'RTS', 'temporal_score', 'spatial_score',
                                              'actual_completion']]
    print(bottom_plays.to_string(index=False))

    # Top/Bottom plays (Scaled RTS)
    print("\n" + "="*80)
    print("TOP 10 PLAYS BY SCALED RTS")
    print("="*80)
    top_plays_scaled = df.nlargest(10, 'RTS_scaled')[['game_id', 'play_id', 'passer', 'targeted_receiver',
                                                        'route_type', 'RTS_scaled', 'temporal_score_scaled',
                                                        'spatial_score_scaled', 'actual_completion']]
    print(top_plays_scaled.to_string(index=False))

    print("\n" + "="*80)
    print("BOTTOM 10 PLAYS BY SCALED RTS")
    print("="*80)
    bottom_plays_scaled = df.nsmallest(10, 'RTS_scaled')[['game_id', 'play_id', 'passer', 'targeted_receiver',
                                                            'route_type', 'RTS_scaled', 'temporal_score_scaled',
                                                            'spatial_score_scaled', 'actual_completion']]
    print(bottom_plays_scaled.to_string(index=False))
    
    print("\n" + "="*80)
    print("POST-PROCESSING COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    print(f"  - rts_scored_plays.csv (full dataset with RTS scores)")
    print(f"  - rts_summary_statistics.csv (summary by passer/route)")
    print(f"  - rts_diagnostic_plots.png (6-panel visualization)")
    print(f"  - passer_comparison.png (QB performance comparison)")


if __name__ == "__main__":
    main()