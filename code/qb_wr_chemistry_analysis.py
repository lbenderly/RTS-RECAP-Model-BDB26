"""
QB-WR Chemistry Analysis

Analyzes quarterback-receiver pairings to identify chemistry patterns
across different route types and overall performance.

"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Path to your scored results
RESULTS_PATH = Path("results/rts_scores/rts_scored_plays.csv")
OUTPUT_DIR = Path("results/qb_wr_chemistry")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Analysis configuration
CONFIG = {
    'min_plays_overall': 5,      # Minimum plays for overall pairing analysis
    'min_plays_by_route': 3,     # Minimum plays for route-specific analysis
    'top_n_pairings': 15,        # Number of top pairings to visualize
    'highlight_qb': 'Jared Goff', # QB to highlight in analyses
}

# ==============================================================================
# CHEMISTRY ANALYSIS FUNCTIONS
# ==============================================================================

def compute_pairing_stats(
    df: pd.DataFrame,
    group_cols: list,
    min_plays: int = 5
) -> pd.DataFrame:
    """
    Compute statistics for QB-WR pairings.
    
    Args:
        df: Scored plays dataframe
        group_cols: Columns to group by (e.g., ['passer', 'targeted_receiver'])
        min_plays: Minimum number of plays to include pairing
    
    Returns:
        DataFrame with pairing statistics
    """
    stats = df.groupby(group_cols).agg({
        'RTS_scaled': ['count', 'mean', 'std', 'min', 'max'],
        'temporal_score_scaled': 'mean',
        'spatial_score_scaled': 'mean',
        'temporal_error_abs': 'mean',
        'spatial_error_abs': 'mean',
        'actual_completion': ['mean', 'sum'],
        'expected_points_added': ['mean', 'sum'],
        'yards_gained': 'mean',
    }).reset_index()

    # Flatten column names
    stats.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                     for col in stats.columns.values]

    # Rename for clarity
    stats = stats.rename(columns={
        'RTS_scaled_count': 'n_plays',
        'RTS_scaled_mean': 'mean_RTS',
        'RTS_scaled_std': 'std_RTS',
        'RTS_scaled_min': 'min_RTS',
        'RTS_scaled_max': 'max_RTS',
        'temporal_score_scaled_mean': 'mean_temporal_score',
        'spatial_score_scaled_mean': 'mean_spatial_score',
        'temporal_error_abs_mean': 'mean_temporal_error_sec',
        'spatial_error_abs_mean': 'mean_spatial_error_yards',
        'actual_completion_mean': 'completion_rate',
        'actual_completion_sum': 'completions',
        'expected_points_added_mean': 'mean_epa',
        'expected_points_added_sum': 'total_epa',
        'yards_gained_mean': 'mean_yards',
    })
    
    # Filter by minimum plays
    stats = stats[stats['n_plays'] >= min_plays].copy()
    
    # Sort by mean RTS
    stats = stats.sort_values('mean_RTS', ascending=False).reset_index(drop=True)
    
    return stats


def analyze_overall_chemistry(df: pd.DataFrame, min_plays: int = 5) -> pd.DataFrame:
    """Analyze overall QB-WR chemistry across all routes."""
    print(f"\nAnalyzing overall QB-WR chemistry (min {min_plays} plays)...")
    
    stats = compute_pairing_stats(
        df,
        group_cols=['passer', 'targeted_receiver'],
        min_plays=min_plays
    )
    
    print(f"✓ Found {len(stats)} QB-WR pairings with {min_plays}+ plays")
    
    return stats


def analyze_chemistry_by_route(df: pd.DataFrame, min_plays: int = 3) -> pd.DataFrame:
    """Analyze QB-WR chemistry for specific route types."""
    print(f"\nAnalyzing route-specific QB-WR chemistry (min {min_plays} plays)...")
    
    stats = compute_pairing_stats(
        df,
        group_cols=['passer', 'targeted_receiver', 'route_type'],
        min_plays=min_plays
    )
    
    print(f"✓ Found {len(stats)} QB-WR-Route combinations with {min_plays}+ plays")
    
    return stats


def create_qb_specific_report(
    df: pd.DataFrame,
    overall_stats: pd.DataFrame,
    route_stats: pd.DataFrame,
    qb_name: str
) -> dict:
    """Create detailed report for a specific QB."""
    
    qb_overall = overall_stats[overall_stats['passer'] == qb_name].copy()
    qb_routes = route_stats[route_stats['passer'] == qb_name].copy()
    
    report = {
        'qb_name': qb_name,
        'total_plays': len(df[df['passer'] == qb_name]),
        'n_receivers': qb_overall['targeted_receiver'].nunique(),
        'overall_rts': df[df['passer'] == qb_name]['RTS_scaled'].mean(),
        'best_receiver_overall': None,
        'worst_receiver_overall': None,
        'top_route_combos': [],
    }
    
    if len(qb_overall) > 0:
        report['best_receiver_overall'] = qb_overall.iloc[0]['targeted_receiver']
        report['worst_receiver_overall'] = qb_overall.iloc[-1]['targeted_receiver']
    
    if len(qb_routes) > 0:
        report['top_route_combos'] = qb_routes.nlargest(10, 'mean_RTS')[[
            'targeted_receiver', 'route_type', 'n_plays', 'mean_RTS', 
            'completion_rate', 'mean_epa'
        ]].to_dict('records')
    
    return report


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_top_pairings(stats: pd.DataFrame, output_dir: Path, top_n: int = 15):
    """Plot top QB-WR pairings by RTS score."""
    
    top_pairings = stats.nlargest(top_n, 'mean_RTS')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create pairing labels
    labels = [f"{row['passer']} → {row['targeted_receiver']}\n({row['n_plays']} plays)" 
              for _, row in top_pairings.iterrows()]
    
    y_pos = range(len(labels))
    
    # Plot bars
    bars = ax.barh(y_pos, top_pairings['mean_RTS'], alpha=0.8)
    
    # Color bars by completion rate
    colors = plt.cm.RdYlGn(top_pairings['completion_rate'])
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Mean RTS Score', fontsize=12)
    ax.set_title(f'Top {top_n} QB-WR Pairings by RTS Score', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add colorbar for completion rate
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, 
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Completion Rate', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_qb_wr_pairings.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved top pairings plot")
    plt.close()


def plot_qb_receiver_heatmap(df: pd.DataFrame, qb_name: str, output_dir: Path):
    """Create heatmap of QB's performance with different receivers by route."""
    
    qb_data = df[df['passer'] == qb_name].copy()
    
    if len(qb_data) == 0:
        print(f"⚠ No data found for {qb_name}")
        return
    
    # Create pivot table: receivers x routes
    pivot = qb_data.groupby(['targeted_receiver', 'route_type']).agg({
        'RTS_scaled': 'mean',
        'play_id': 'count'
    }).reset_index()

    # Only include combinations with 2+ plays
    pivot = pivot[pivot['play_id'] >= 2]

    heatmap_data = pivot.pivot(index='targeted_receiver',
                                 columns='route_type',
                                 values='RTS_scaled')
    
    count_data = pivot.pivot(index='targeted_receiver', 
                              columns='route_type', 
                              values='play_id')
    
    if heatmap_data.empty:
        print(f"⚠ Not enough data for {qb_name} heatmap")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot heatmap
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt='.1f',
                cmap='RdYlGn',
                center=50,
                vmin=0,
                vmax=100,
                cbar_kws={'label': 'RTS Score'},
                linewidths=0.5,
                linecolor='gray',
                ax=ax)
    
    # Add play counts as text
    for i, receiver in enumerate(heatmap_data.index):
        for j, route in enumerate(heatmap_data.columns):
            count = count_data.loc[receiver, route] if pd.notna(count_data.loc[receiver, route]) else 0
            if count > 0:
                ax.text(j + 0.5, i + 0.75, f'n={int(count)}', 
                       ha='center', va='center', fontsize=7, color='black', alpha=0.6)
    
    ax.set_title(f'{qb_name}: RTS Score by Receiver & Route Type', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Route Type', fontsize=12)
    ax.set_ylabel('Receiver', fontsize=12)
    
    plt.tight_layout()
    
    # Save with QB name in filename
    filename = f"{qb_name.replace(' ', '_').lower()}_receiver_route_heatmap.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap for {qb_name}")
    plt.close()


def plot_chemistry_comparison(route_stats: pd.DataFrame, qb_name: str, 
                               route_type: str, output_dir: Path):
    """Compare specific route performance across receivers for a QB."""
    
    qb_route_data = route_stats[
        (route_stats['passer'] == qb_name) & 
        (route_stats['route_type'] == route_type)
    ].copy()
    
    if len(qb_route_data) < 2:
        print(f"⚠ Not enough data for {qb_name} on {route_type} routes")
        return
    
    qb_route_data = qb_route_data.sort_values('mean_RTS', ascending=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{qb_name}: {route_type} Route Performance by Receiver', 
                 fontsize=14, fontweight='bold')
    
    receivers = qb_route_data['targeted_receiver']
    y_pos = range(len(receivers))
    
    # 1. Overall RTS
    bars1 = axes[0].barh(y_pos, qb_route_data['mean_RTS'], alpha=0.8, color='steelblue')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(receivers)
    axes[0].set_xlabel('Mean RTS Score')
    axes[0].set_title('Overall RTS Score')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Add play counts
    for i, (idx, row) in enumerate(qb_route_data.iterrows()):
        axes[0].text(row['mean_RTS'] + 1, i, f"n={int(row['n_plays'])}", 
                    va='center', fontsize=9)
    
    # 2. Temporal vs Spatial breakdown
    width = 0.35
    axes[1].barh([i - width/2 for i in y_pos], qb_route_data['mean_temporal_score'], 
                 width, label='Temporal', alpha=0.8)
    axes[1].barh([i + width/2 for i in y_pos], qb_route_data['mean_spatial_score'], 
                 width, label='Spatial', alpha=0.8)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(receivers)
    axes[1].set_xlabel('Score')
    axes[1].set_title('Temporal vs Spatial Scores')
    axes[1].legend()
    axes[1].grid(axis='x', alpha=0.3)
    
    # 3. Completion rate and EPA
    ax3_1 = axes[2]
    ax3_2 = ax3_1.twiny()
    
    bars_comp = ax3_1.barh(y_pos, qb_route_data['completion_rate'], 
                           alpha=0.6, color='green', label='Completion Rate')
    bars_epa = ax3_2.barh([i + 0.3 for i in y_pos], qb_route_data['mean_epa'], 
                          alpha=0.6, color='orange', label='Mean EPA')
    
    ax3_1.set_yticks(y_pos)
    ax3_1.set_yticklabels(receivers)
    ax3_1.set_xlabel('Completion Rate', color='green')
    ax3_1.tick_params(axis='x', labelcolor='green')
    ax3_1.set_xlim(0, 1)
    
    ax3_2.set_xlabel('Mean EPA', color='orange')
    ax3_2.tick_params(axis='x', labelcolor='orange')
    
    axes[2].set_title('Completion & EPA')
    
    plt.tight_layout()
    
    filename = f"{qb_name.replace(' ', '_').lower()}_{route_type.lower()}_comparison.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {route_type} comparison for {qb_name}")
    plt.close()


def create_receiver_matrix(df: pd.DataFrame, qb_name: str, output_dir: Path):
    """Create comparison matrix showing best/worst receiver by route."""
    
    qb_data = df[df['passer'] == qb_name].copy()
    
    if len(qb_data) == 0:
        return
    
    # Get unique routes
    routes = sorted(qb_data['route_type'].unique())
    
    # For each route, find best and worst receiver
    results = []
    for route in routes:
        route_data = qb_data[qb_data['route_type'] == route]
        
        if len(route_data) < 2:
            continue
        
        receiver_stats = route_data.groupby('targeted_receiver').agg({
            'RTS_scaled': ['mean', 'count'],
            'actual_completion': 'mean'
        })

        # Filter to receivers with 2+ plays on this route
        receiver_stats = receiver_stats[receiver_stats[('RTS_scaled', 'count')] >= 2]

        if len(receiver_stats) < 2:
            continue

        best_receiver = receiver_stats[('RTS_scaled', 'mean')].idxmax()
        worst_receiver = receiver_stats[('RTS_scaled', 'mean')].idxmin()

        results.append({
            'route_type': route,
            'best_receiver': best_receiver,
            'best_rts': receiver_stats.loc[best_receiver, ('RTS_scaled', 'mean')],
            'best_plays': int(receiver_stats.loc[best_receiver, ('RTS_scaled', 'count')]),
            'worst_receiver': worst_receiver,
            'worst_rts': receiver_stats.loc[worst_receiver, ('RTS_scaled', 'mean')],
            'worst_plays': int(receiver_stats.loc[worst_receiver, ('RTS_scaled', 'count')]),
        })
    
    if not results:
        print(f"⚠ Not enough data for {qb_name} receiver matrix")
        return
    
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    filename = f"{qb_name.replace(' ', '_').lower()}_receiver_matrix.csv"
    results_df.to_csv(output_dir / filename, index=False)
    print(f"✓ Saved receiver matrix for {qb_name}")
    
    return results_df


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function."""
    
    print("="*80)
    print("QB-WR CHEMISTRY ANALYSIS")
    print("="*80)
    
    # Load data
    print(f"\nLoading scored plays from: {RESULTS_PATH}")
    df = pd.read_csv(RESULTS_PATH)
    print(f"✓ Loaded {len(df)} plays")
    
    # Basic info
    print(f"\nDataset overview:")
    print(f"  Unique QBs: {df['passer'].nunique()}")
    print(f"  Unique receivers: {df['targeted_receiver'].nunique()}")
    print(f"  Unique routes: {df['route_type'].nunique()}")
    print(f"  Route types: {', '.join(sorted(df['route_type'].unique()))}")
    
    # =========================================================================
    # 1. OVERALL QB-WR CHEMISTRY
    # =========================================================================
    print("\n" + "="*80)
    print("OVERALL QB-WR CHEMISTRY ANALYSIS")
    print("="*80)
    
    overall_stats = analyze_overall_chemistry(df, min_plays=CONFIG['min_plays_overall'])
    
    # Save overall stats
    overall_path = OUTPUT_DIR / 'overall_qb_wr_chemistry.csv'
    overall_stats.to_csv(overall_path, index=False)
    print(f"✓ Saved overall chemistry stats: {overall_path}")
    
    # Display top 10
    print("\nTop 10 QB-WR Pairings (Overall):")
    print(overall_stats.head(10)[['passer', 'targeted_receiver', 'n_plays', 
                                   'mean_RTS', 'completion_rate', 'mean_epa']].to_string(index=False))
    
    # =========================================================================
    # 2. ROUTE-SPECIFIC CHEMISTRY
    # =========================================================================
    print("\n" + "="*80)
    print("ROUTE-SPECIFIC QB-WR CHEMISTRY ANALYSIS")
    print("="*80)
    
    route_stats = analyze_chemistry_by_route(df, min_plays=CONFIG['min_plays_by_route'])
    
    # Save route-specific stats
    route_path = OUTPUT_DIR / 'route_specific_qb_wr_chemistry.csv'
    route_stats.to_csv(route_path, index=False)
    print(f"✓ Saved route-specific chemistry stats: {route_path}")
    
    # Display top 10
    print("\nTop 10 QB-WR-Route Combinations:")
    print(route_stats.head(10)[['passer', 'targeted_receiver', 'route_type', 
                                 'n_plays', 'mean_RTS', 'completion_rate']].to_string(index=False))
    
    # =========================================================================
    # 3. QB-SPECIFIC REPORTS
    # =========================================================================
    print("\n" + "="*80)
    print("GENERATING QB-SPECIFIC REPORTS")
    print("="*80)
    
    # Get all QBs with enough plays
    qb_play_counts = df['passer'].value_counts()
    qbs_to_analyze = qb_play_counts[qb_play_counts >= 10].index.tolist()
    
    print(f"Analyzing {len(qbs_to_analyze)} QBs with 10+ plays")
    
    qb_reports = {}
    for qb in qbs_to_analyze:
        report = create_qb_specific_report(df, overall_stats, route_stats, qb)
        qb_reports[qb] = report
        
        # Print highlighted QB report
        if qb == CONFIG['highlight_qb']:
            print(f"\n{'='*60}")
            print(f"DETAILED REPORT: {qb}")
            print(f"{'='*60}")
            print(f"Total plays: {report['total_plays']}")
            print(f"Unique receivers: {report['n_receivers']}")
            print(f"Overall RTS: {report['overall_rts']:.2f}")
            print(f"Best receiver (overall): {report['best_receiver_overall']}")
            print(f"Worst receiver (overall): {report['worst_receiver_overall']}")
            print(f"\nTop 10 Route-Receiver Combinations:")
            for i, combo in enumerate(report['top_route_combos'][:10], 1):
                print(f"  {i}. {combo['targeted_receiver']} - {combo['route_type']}: "
                      f"RTS={combo['mean_RTS']:.1f}, Comp%={combo['completion_rate']:.1%}, "
                      f"EPA={combo['mean_epa']:.2f} ({combo['n_plays']} plays)")
    
    # =========================================================================
    # 4. VISUALIZATIONS
    # =========================================================================
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Top pairings
    plot_top_pairings(overall_stats, OUTPUT_DIR, top_n=CONFIG['top_n_pairings'])
    
    # Heatmaps for main QBs
    for qb in qbs_to_analyze[:5]:  # Top 5 QBs by play count
        plot_qb_receiver_heatmap(df, qb, OUTPUT_DIR)
    
    # Receiver matrix for highlighted QB
    if CONFIG['highlight_qb'] in qbs_to_analyze:
        create_receiver_matrix(df, CONFIG['highlight_qb'], OUTPUT_DIR)
    
    # Route-specific comparisons for highlighted QB
    if CONFIG['highlight_qb'] in qbs_to_analyze:
        qb_routes = route_stats[route_stats['passer'] == CONFIG['highlight_qb']]['route_type'].unique()
        for route in qb_routes[:5]:  # Top 5 most common routes
            plot_chemistry_comparison(route_stats, CONFIG['highlight_qb'], route, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    print(f"  - overall_qb_wr_chemistry.csv")
    print(f"  - route_specific_qb_wr_chemistry.csv")
    print(f"  - top_qb_wr_pairings.png")
    print(f"  - [QB]_receiver_route_heatmap.png (for each QB)")
    print(f"  - [QB]_receiver_matrix.csv (best/worst by route)")
    print(f"  - [QB]_[ROUTE]_comparison.png (route-specific)")


if __name__ == "__main__":
    main()
