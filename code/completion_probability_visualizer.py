"""
Completion Probability Visualizer

Standalone script for generating temporal and spatial visualizations of completion probability.
Takes a game_id and play_id as input and produces:
1. Static temporal line chart showing completion probability over time
2. Animated temporal line chart (MP4) showing probability evolving frame-by-frame
3. Individual spatial heatmap images for each frame
4. MP4 animation of all spatial heatmaps

Usage:
    python completion_probability_visualizer.py --game_id 2023090700 --play_id 56

"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import argparse
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class RouteAwareCompletionModel(nn.Module):
    """Route-aware completion probability model with transformer architecture"""

    def __init__(
        self,
        num_routes: int,
        route_embedding_dim: int = 32,
        player_feature_dim: int = 256,
        player_input_dim: int = 8,
        momentum_feature_dim: int = 7,
        ngs_feature_dim: int = 5,
        hidden_dim: int = 512,
        num_transformer_layers: int = 4,
        num_attention_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_routes = num_routes
        self.route_embedding_dim = route_embedding_dim
        self.player_feature_dim = player_feature_dim

        # Route type embedding
        self.route_embedding = nn.Embedding(
            num_embeddings=num_routes,
            embedding_dim=route_embedding_dim
        )

        # Player feature normalization and projection
        self.player_norm = nn.BatchNorm1d(player_input_dim)
        self.player_input_projection = nn.Sequential(
            nn.Linear(player_input_dim, player_feature_dim),
            nn.ReLU(),
            nn.LayerNorm(player_feature_dim),
            nn.Dropout(dropout)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=player_feature_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.player_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_transformer_layers
        )

        # Receiver momentum encoder
        self.momentum_encoder = nn.Sequential(
            nn.Linear(momentum_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # NGS feature encoder
        self.ngs_encoder = nn.Sequential(
            nn.Linear(ngs_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Completion prediction head
        combined_dim = player_feature_dim + route_embedding_dim + 32 + 32

        self.completion_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(
        self,
        player_features: torch.Tensor,
        route_ids: torch.Tensor,
        momentum_features: torch.Tensor,
        ngs_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass returning logits"""
        batch_size = player_features.shape[0]

        # Encode route type
        route_embedded = self.route_embedding(route_ids)

        # Encode player features
        B, P, F = player_features.shape
        player_normed = self.player_norm(
            player_features.permute(0, 2, 1)
        ).permute(0, 2, 1)

        player_projected = self.player_input_projection(player_normed)
        player_encoded = self.player_encoder(player_projected)
        player_pooled = player_encoded.mean(dim=1)

        # Encode momentum and NGS
        momentum_encoded = self.momentum_encoder(momentum_features)
        ngs_encoded = self.ngs_encoder(ngs_features)

        # Combine and predict
        combined = torch.cat([
            player_pooled,
            route_embedded,
            momentum_encoded,
            ngs_encoded
        ], dim=1)

        logits = self.completion_head(combined).squeeze(-1)
        return logits

    def predict_proba(self, *args, **kwargs) -> torch.Tensor:
        """Get probabilities instead of logits"""
        logits = self.forward(*args, **kwargs)
        return torch.sigmoid(logits)


# ============================================================================
# COUNTERFACTUAL PREDICTOR
# ============================================================================

class CounterfactualCompletionPredictor:
    """Wrapper for making counterfactual predictions"""

    def __init__(self, model: nn.Module, route_to_id: dict, device: str = 'cpu'):
        self.model = model
        self.model.eval()
        self.route_to_id = route_to_id
        self.device = device

    def predict_at_frame(
        self,
        play_data: pd.DataFrame,
        hypothetical_frame: int,
        target_x: float,
        target_y: float
    ) -> Dict:
        """Predict completion probability for a hypothetical throw"""
        frame_data = play_data[play_data['frame_id'] == hypothetical_frame].copy()

        if len(frame_data) == 0:
            raise ValueError(f"Frame {hypothetical_frame} not found")

        # Get QB data
        qb_data = frame_data[frame_data['player_role'] == 'Passer']
        if len(qb_data) == 0:
            raise ValueError("No passer found")
        qb_data = qb_data.iloc[0]

        qb_x, qb_y = qb_data['x'], qb_data['y']
        air_distance = np.sqrt((target_x - qb_x)**2 + (target_y - qb_y)**2)

        # Get receiver data
        receiver_data = frame_data[frame_data['player_role'] == 'Targeted Receiver']
        if len(receiver_data) == 0:
            receiver_vx, receiver_vy = 0.0, 0.0
            receiver_speed = 0.0
            receiver_x, receiver_y = target_x, target_y
            receiver_ox, receiver_oy = 1.0, 0.0
        else:
            receiver_data = receiver_data.iloc[0]
            receiver_x, receiver_y = receiver_data['x'], receiver_data['y']
            receiver_vx, receiver_vy = receiver_data['vx'], receiver_data['vy']
            receiver_speed = receiver_data['s']
            receiver_ox = receiver_data.get('ox', 1.0)
            receiver_oy = receiver_data.get('oy', 0.0)

        # Calculate features
        target_vec_x = target_x - receiver_x
        target_vec_y = target_y - receiver_y
        target_vec_mag = np.sqrt(target_vec_x**2 + target_vec_y**2) + 1e-6

        momentum_alignment = (receiver_vx * target_vec_x + receiver_vy * target_vec_y) / target_vec_mag
        orientation_alignment = (receiver_ox * target_vec_x + receiver_oy * target_vec_y) / target_vec_mag
        receiver_to_ball_distance = target_vec_mag

        # Target separation
        defenders = frame_data[frame_data['player_role'] == 'Defensive Coverage']
        if len(defenders) > 0:
            def_dists = np.sqrt(
                (defenders['x'] - target_x)**2 +
                (defenders['y'] - target_y)**2
            )
            target_separation = def_dists.min()
        else:
            target_separation = 10.0

        sideline_separation = min(target_y, 53.3 - target_y)

        # Time calculations
        snap_frame = play_data['frame_id'].min()
        time_to_throw = (hypothetical_frame - snap_frame) / 10.0

        # Route type
        route_type = play_data['route_type'].iloc[0] if 'route_type' in play_data.columns else 'UNKNOWN'
        route_id = self.route_to_id.get(route_type, 0)

        # Prepare features
        player_cols = ['x', 'y', 'vx', 'vy', 'ox', 'oy', 's', 'a']
        player_features = frame_data[player_cols].values.astype(np.float32)

        # Pad to 22 players
        if len(player_features) < 22:
            padding = np.zeros((22 - len(player_features), 8), dtype=np.float32)
            player_features = np.vstack([player_features, padding])
        elif len(player_features) > 22:
            player_features = player_features[:22]

        player_features = np.nan_to_num(player_features, 0.0)

        # Momentum features (CRITICAL ORDER)
        momentum_features = np.array([
            receiver_vx, receiver_vy, receiver_speed,
            momentum_alignment, orientation_alignment,
            receiver_to_ball_distance,
            time_to_throw
        ], dtype=np.float32)
        momentum_features = np.nan_to_num(momentum_features, 0.0)

        # NGS features
        ngs_features = np.array([
            air_distance,
            target_separation,
            sideline_separation,
            qb_data['s'],
            time_to_throw
        ], dtype=np.float32)
        ngs_features = np.nan_to_num(ngs_features, 0.0)

        # Convert to tensors
        player_features = torch.tensor(player_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        route_id_tensor = torch.tensor([route_id], dtype=torch.long).to(self.device)
        momentum_features = torch.tensor(momentum_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        ngs_features = torch.tensor(ngs_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(player_features, route_id_tensor, momentum_features, ngs_features)
            completion_prob = torch.sigmoid(logits).item()

        return {
            'completion_prob': completion_prob,
            'air_distance': air_distance,
            'target_separation': target_separation,
            'time_to_throw': time_to_throw,
            'momentum_alignment': momentum_alignment
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def sanitize_name(name: str) -> str:
    """
    Sanitize player name for use in file paths.
    Removes spaces, special characters, and converts to safe format.
    """
    import re
    # Remove any non-alphanumeric characters except hyphens
    sanitized = re.sub(r'[^a-zA-Z0-9\-]', '', name.replace(' ', ''))
    return sanitized if sanitized else 'Unknown'


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_play_tracking_data(game_id: int, play_id: int, input_dir: str) -> pd.DataFrame:
    """Load tracking data for a specific play"""
    input_path = Path(input_dir)

    # Load all input files and filter
    csv_files = list(input_path.glob("input_*.csv"))

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        play_data = df[(df['game_id'] == game_id) & (df['play_id'] == play_id)]
        if len(play_data) > 0:
            return play_data

    raise ValueError(f"Play {game_id}-{play_id} not found in input data")


def convert_tracking_to_cartesian(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Convert polar coordinates to Cartesian"""
    df = tracking_df.copy()

    dir_adjusted = ((df['dir'] - 90) * -1) % 360
    o_adjusted = ((df['o'] - 90) * -1) % 360

    df['vx'] = df['s'] * np.cos(np.radians(dir_adjusted))
    df['vy'] = df['s'] * np.sin(np.radians(dir_adjusted))
    df['ox'] = np.cos(np.radians(o_adjusted))
    df['oy'] = np.sin(np.radians(o_adjusted))

    return df


def standardize_tracking_directions(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize play directions to always moving left to right"""
    df = tracking_df.copy()

    # Flip if playing to the left
    mask = df['play_direction'] == 'left'

    df.loc[mask, 'x'] = 120 - df.loc[mask, 'x']
    df.loc[mask, 'y'] = 53.3 - df.loc[mask, 'y']
    df.loc[mask, 'vx'] = -df.loc[mask, 'vx']
    df.loc[mask, 'vy'] = -df.loc[mask, 'vy']
    df.loc[mask, 'ox'] = -df.loc[mask, 'ox']
    df.loc[mask, 'oy'] = -df.loc[mask, 'oy']
    df.loc[mask, 'ball_land_x'] = 120 - df.loc[mask, 'ball_land_x']
    df.loc[mask, 'ball_land_y'] = 53.3 - df.loc[mask, 'ball_land_y']

    return df


def load_model(model_path: str, device: str = 'cpu'):
    """Load trained model from checkpoint"""
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except:
        # Fallback for older PyTorch versions
        checkpoint = torch.load(model_path, map_location=device)

    route_to_id = checkpoint['route_to_id']
    num_routes = checkpoint['num_routes']

    model = RouteAwareCompletionModel(
        num_routes=num_routes,
        route_embedding_dim=32,
        player_feature_dim=256,
        num_transformer_layers=4,
        num_attention_heads=8,
        dropout=0.1
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    predictor = CounterfactualCompletionPredictor(model, route_to_id, device)

    return predictor, route_to_id


# ============================================================================
# TEMPORAL ANALYSIS FUNCTIONS
# ============================================================================

def compute_temporal_analysis(
    play_data: pd.DataFrame,
    predictor: CounterfactualCompletionPredictor,
    actual_target_x: float,
    actual_target_y: float
) -> pd.DataFrame:
    """
    Compute completion probability across all frames for temporal analysis.
    Returns DataFrame with columns: frame_id, time_since_snap, completion_prob, is_actual_release
    """
    min_frame = play_data['frame_id'].min()
    max_frame = play_data['frame_id'].max()

    results = []
    for frame_id in range(min_frame, max_frame + 1):
        try:
            pred = predictor.predict_at_frame(play_data, frame_id, actual_target_x, actual_target_y)
            results.append({
                'frame_id': frame_id,
                'time_since_snap': (frame_id - min_frame) / 10.0,
                'completion_prob': pred['completion_prob'],
                'is_actual_release': (frame_id == max_frame)
            })
        except Exception as e:
            print(f"  Warning: Could not compute frame {frame_id}: {e}")
            pass

    return pd.DataFrame(results)


def plot_temporal_analysis(
    results: pd.DataFrame,
    save_path: Path,
    passer_name: str = 'Unknown',
    receiver_name: str = 'Unknown',
    route_type: str = 'Unknown',
    rts_scores: Optional[Dict] = None
):
    """
    Plot completion probability over time (static version).

    Args:
        rts_scores: Dict with keys: temporal_error_abs, spatial_error_abs,
                    temporal_score, spatial_score, RTS
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(results['time_since_snap'], results['completion_prob'],
            linewidth=3, marker='o', markersize=6, label='Completion Probability',
            color='#1f77b4', alpha=0.9)

    # Mark actual release (hollow red circle)
    actual_release = results[results['is_actual_release']]
    if len(actual_release) > 0:
        ax.scatter(actual_release['time_since_snap'], actual_release['completion_prob'],
                  s=500, facecolors='none', edgecolors='red', marker='o',
                  label='Actual Release', zorder=5, linewidths=4)

    # Mark optimal release (green star)
    optimal_idx = results['completion_prob'].idxmax()
    optimal = results.loc[optimal_idx]
    ax.scatter(optimal['time_since_snap'], optimal['completion_prob'],
              s=300, c='green', marker='*', label='Optimal Release', zorder=5)

    ax.set_xlabel('Time Since Snap (seconds)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Completion Probability', fontsize=14, fontweight='bold')

    # Enhanced title with player and route info
    title = f'Temporal Analysis: {passer_name} → {receiver_name}\n{route_type} Route'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Create custom legend with smaller red circle marker
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', linewidth=3, marker='o', markersize=8, label='Completion Probability'),
        Line2D([0], [0], color='red', linewidth=0, marker='o', markersize=10,
               markerfacecolor='none', markeredgecolor='red', markeredgewidth=3, label='Actual Release'),
        Line2D([0], [0], color='green', linewidth=0, marker='*', markersize=12, label='Optimal Release')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_ylim(0, 1)

    # Add RTS scores text box if available
    if rts_scores:
        score_text = (
            f"RTS Scores\n"
            f"━━━━━━━━━━\n"
            f"RTS: {rts_scores.get('RTS', 0):.3f}\n"
            f"\n"
            f"Temporal Error: {rts_scores.get('temporal_error_abs', 0):.3f}\n"
            f"Temporal Score: {rts_scores.get('temporal_score', 0):.3f}\n"
            f"\n"
            f"Spatial Error: {rts_scores.get('spatial_error_abs', 0):.3f}\n"
            f"Spatial Score: {rts_scores.get('spatial_score', 0):.3f}"
        )
        # Add text box in upper right
        ax.text(0.98, 0.03, score_text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='black', linewidth=2),
                family='monospace')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    if len(actual_release) > 0:
        print(f"  Actual Release: {actual_release['time_since_snap'].iloc[0]:.2f}s, "
              f"Prob: {actual_release['completion_prob'].iloc[0]:.1%}")
    print(f"  Optimal Release: {optimal['time_since_snap']:.2f}s, "
          f"Prob: {optimal['completion_prob']:.1%}")


def plot_temporal_frame(
    results: pd.DataFrame,
    current_frame_idx: int,
    save_path: Path,
    passer_name: str = 'Unknown',
    receiver_name: str = 'Unknown',
    route_type: str = 'Unknown',
    rts_scores: Optional[Dict] = None,
    figsize: Tuple[float, float] = (14, 7)
):
    """
    Plot temporal line chart up to a specific frame (for animation).

    Args:
        results: DataFrame with temporal analysis results
        current_frame_idx: Index of current frame (0-based) to show up to
        save_path: Path to save PNG
        passer_name: Name of the passer
        receiver_name: Name of the receiver
        route_type: Route type
        rts_scores: Optional RTS scores dictionary
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get data up to current frame
    data_to_show = results.iloc[:current_frame_idx + 1]

    # Plot line up to current frame
    if len(data_to_show) > 1:
        ax.plot(data_to_show['time_since_snap'], data_to_show['completion_prob'],
                linewidth=3, marker='o', markersize=6,
                color='#1f77b4', alpha=0.9)

    # Mark actual release (hollow red circle) - only show if we've reached it
    actual_release = results[results['is_actual_release']]
    if len(actual_release) > 0 and current_frame_idx >= len(results) - 1:
        ax.scatter(actual_release['time_since_snap'], actual_release['completion_prob'],
                  s=500, facecolors='none', edgecolors='red', marker='o',
                  label='Actual Release', zorder=5, linewidths=4)

    # Mark optimal release (green star) - only show once we've passed it
    optimal_idx = results['completion_prob'].idxmax()
    optimal = results.loc[optimal_idx]
    if current_frame_idx >= optimal_idx:
        ax.scatter(optimal['time_since_snap'], optimal['completion_prob'],
                  s=300, c='green', marker='*', label='Optimal Release', zorder=5)

    ax.set_xlabel('Time Since Snap (seconds)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Completion Probability', fontsize=14, fontweight='bold')

    # Get time for current frame
    if len(data_to_show) > 0:
        current_point = data_to_show.iloc[-1]
        current_time = current_point['time_since_snap']
        title = (f'Temporal Analysis: {passer_name} → {receiver_name} ({route_type} Route)\n'
                f'Time: {current_time:.2f}s | Completion Prob: {current_point["completion_prob"]:.1%}')
    else:
        title = f'Temporal Analysis: {passer_name} → {receiver_name} ({route_type} Route)'

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Legend
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)

    # Set fixed axis limits based on full data
    ax.set_xlim(results['time_since_snap'].min() - 0.1,
                results['time_since_snap'].max() + 0.1)
    ax.set_ylim(0, 1)

    # Add RTS scores text box if available
    if rts_scores:
        score_text = (
            f"RTS: {rts_scores.get('RTS', 0):.3f}\n"
            f"Temporal: {rts_scores.get('temporal_score', 0):.3f}\n"
            f"Spatial: {rts_scores.get('spatial_score', 0):.3f}"
        )
        ax.text(0.98, 0.03, score_text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='black', linewidth=2),
                family='monospace')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def create_temporal_animation(
    results: pd.DataFrame,
    frames_folder: Path,
    output_path: Path,
    passer_name: str = 'Unknown',
    receiver_name: str = 'Unknown',
    route_type: str = 'Unknown',
    rts_scores: Optional[Dict] = None,
    fps: int = 10
):
    """
    Create animated video of temporal line chart forming progressively.

    Args:
        results: DataFrame with temporal analysis results
        frames_folder: Folder to save individual frame images
        output_path: Path to save MP4
        passer_name: Name of the passer
        receiver_name: Name of the receiver
        route_type: Route type
        rts_scores: Optional RTS scores dictionary
        fps: Frames per second
    """
    print(f"Generating temporal animation frames...")

    # Create temporal frames subfolder
    temporal_frames_folder = frames_folder / "temporal_frames"
    temporal_frames_folder.mkdir(exist_ok=True)

    frame_files = []
    num_frames = len(results)

    for i in range(num_frames):
        frame_path = temporal_frames_folder / f"temporal_frame_{i+1:03d}.png"

        plot_temporal_frame(
            results=results,
            current_frame_idx=i,
            save_path=frame_path,
            passer_name=passer_name,
            receiver_name=receiver_name,
            route_type=route_type,
            rts_scores=rts_scores
        )

        frame_files.append(frame_path)

        if (i + 1) % 5 == 0 or i == num_frames - 1:
            print(f"  Generated frame {i+1}/{num_frames}")

    # Create video from frames
    print(f"Creating temporal video from {len(frame_files)} frames...")

    if len(frame_files) == 0:
        raise ValueError("No frames to create animation")

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    height, width, _ = first_frame.shape

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        out.write(frame)

    out.release()

    print(f"✓ Temporal animation saved to {output_path}")
    print(f"  Duration: {len(frame_files) / fps:.1f} seconds at {fps} FPS")


# ============================================================================
# SPATIAL HEATMAP FUNCTIONS
# ============================================================================

def generate_heatmap_for_frame(
    play_data: pd.DataFrame,
    frame_id: int,
    predictor: CounterfactualCompletionPredictor,
    actual_target_x: float,
    actual_target_y: float,
    grid_size: float = 2.0,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None
) -> Dict:
    """
    Generate completion probability heatmap for a single frame

    Args:
        grid_size: Grid spacing in yards
        x_range: (min_x, max_x) for search area. If None, uses full field (10, 110)
        y_range: (min_y, max_y) for search area. If None, uses full field (5, 48)
    """

    # Get receiver location at this frame for metadata
    frame_data = play_data[play_data['frame_id'] == frame_id]
    receiver_data = frame_data[frame_data['player_role'] == 'Targeted Receiver']

    if len(receiver_data) > 0:
        receiver_x = receiver_data.iloc[0]['x']
        receiver_y = receiver_data.iloc[0]['y']
    else:
        # Fallback to actual target if receiver not found
        receiver_x = actual_target_x
        receiver_y = actual_target_y

    # Use full field by default
    if x_range is None:
        x_range = (0, 120)
    if y_range is None:
        y_range = (0, 53.5)

    x_min, x_max = x_range
    y_min, y_max = y_range

    x_grid = np.arange(x_min, x_max, grid_size)
    y_grid = np.arange(y_min, y_max, grid_size)

    probabilities = np.zeros((len(y_grid), len(x_grid)))

    for i, y in enumerate(y_grid):
        for j, x in enumerate(x_grid):
            try:
                pred = predictor.predict_at_frame(play_data, frame_id, x, y)
                probabilities[i, j] = pred['completion_prob']
            except:
                probabilities[i, j] = np.nan

    # Find optimal
    max_prob = np.nanmax(probabilities)
    max_idx = np.unravel_index(np.nanargmax(probabilities), probabilities.shape)
    optimal_x = x_grid[max_idx[1]]
    optimal_y = y_grid[max_idx[0]]

    return {
        'probabilities': probabilities,
        'x_grid': x_grid,
        'y_grid': y_grid,
        'optimal_x': optimal_x,
        'optimal_y': optimal_y,
        'max_prob': max_prob,
        'frame_data': frame_data,
        'actual_target_x': actual_target_x,
        'actual_target_y': actual_target_y,
        'frame_id': frame_id,
        'receiver_x': receiver_x,
        'receiver_y': receiver_y,
        'x_range': x_range,
        'y_range': y_range
    }


def plot_heatmap_frame_matplotlib(
    heatmap_data: Dict,
    save_path: Path,
    passer_name: str = 'Unknown',
    receiver_name: str = 'Unknown',
    route_type: str = 'Unknown',
    rts_scores: Optional[Dict] = None,
    show_players: bool = True,
    show_optimal: bool = True,
    show_actual: bool = True
):
    """
    Create matplotlib figure with football field heatmap.

    Args:
        heatmap_data: Dict from generate_heatmap_for_frame
        save_path: Path to save PNG
        passer_name: Name of the passer (QB)
        receiver_name: Name of the targeted receiver
        route_type: Type of route being run
        rts_scores: Dict with RTS scores (optional)
        show_players: Whether to show player positions
        show_optimal: Whether to show optimal throw location
        show_actual: Whether to show actual throw location
    """
    fig, ax = plt.subplots(figsize=(20, 10), facecolor='#2F5233')

    # Plot heatmap
    X, Y = np.meshgrid(heatmap_data['x_grid'], heatmap_data['y_grid'])
    im = ax.contourf(X, Y, heatmap_data['probabilities'],
                     levels=20, cmap='RdYlGn', alpha=0.7)

    # Plot players if requested
    if show_players and len(heatmap_data['frame_data']) > 0:
        frame_data = heatmap_data['frame_data']

        # Targeted receiver (dark blue star)
        target = frame_data[frame_data['player_role'] == 'Targeted Receiver']
        if len(target) > 0:
            ax.scatter(target['x'], target['y'],
                      s=500, c='darkblue', marker='*',
                      label='Targeted Receiver', zorder=10,
                      edgecolors='white', linewidths=2)

        # Passer (blue triangle)
        passer = frame_data[frame_data['player_role'] == 'Passer']
        if len(passer) > 0:
            ax.scatter(passer['x'], passer['y'],
                      s=350, c='blue', marker='^',
                      label='Passer', zorder=10,
                      edgecolors='white', linewidths=2)

        # Other offense (light blue)
        other_offense = frame_data[
            (frame_data['player_role'] != 'Targeted Receiver') &
            (frame_data['player_role'] != 'Passer') &
            (frame_data['player_role'] != 'Defensive Coverage') &
            (frame_data['player_role'] != 'Pass Rush')
        ]
        if len(other_offense) > 0:
            ax.scatter(other_offense['x'], other_offense['y'],
                      s=200, c='lightblue', marker='o',
                      label='Other Offense', zorder=9,
                      edgecolors='white', linewidths=1.5)

        # Defense (red diamonds)
        defense = frame_data[
            (frame_data['player_role'] == 'Defensive Coverage') |
            (frame_data['player_role'] == 'Pass Rush')
        ]
        if len(defense) > 0:
            ax.scatter(defense['x'], defense['y'],
                      s=200, c='red', marker='D',
                      label='Defense', zorder=9,
                      edgecolors='white', linewidths=1.5)

    # Mark optimal location (lime green star)
    if show_optimal:
        ax.scatter(heatmap_data['optimal_x'], heatmap_data['optimal_y'],
                  s=600, c='lime', marker='*',
                  label=f"Optimal ({heatmap_data['max_prob']:.1%})",
                  zorder=15, edgecolors='black', linewidths=3)

    # Mark actual throw (yellow X)
    if show_actual:
        ax.scatter(heatmap_data['actual_target_x'], heatmap_data['actual_target_y'],
                  s=600, c='yellow', marker='X',
                  label='Actual Target', zorder=15,
                  edgecolors='black', linewidths=3)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Completion Probability', fontsize=14, weight='bold', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Field markings
    ax.axhline(y=0, color='white', linewidth=4, linestyle='--', alpha=0.5)
    ax.axhline(y=53.3, color='white', linewidth=4, linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='white', linewidth=4, linestyle='-', alpha=0.8)
    ax.axvline(x=120, color='white', linewidth=4, linestyle='-', alpha=0.8)

    # Yard lines
    for x in range(10, 120, 10):
        ax.axvline(x=x, color='white', linewidth=2, alpha=0.6)

    for x in range(5, 120, 5):
        if x % 10 != 0:
            ax.axvline(x=x, color='white', linewidth=1, alpha=0.3)

    # Yard markers
    for x in range(10, 120, 10):
        # Calculate yard line number (0-50 from each goal line)
        if x <= 60:
            yard_num = x - 10  # 10->0, 20->10, 30->20, 40->30, 50->40, 60->50
        else:
            yard_num = 110 - x  # 70->40, 80->30, 90->20, 100->10, 110->0

        label = "50" if yard_num == 50 else str(yard_num) if yard_num > 0 else "G"
        ax.text(x, -2, label, ha='center', va='top', fontsize=12,
                color='white', weight='bold')
        ax.text(x, 55.3, label, ha='center', va='bottom', fontsize=12,
                color='white', weight='bold')

    # Styling
    ax.set_xlim(-2, 122)
    ax.set_ylim(-3, 56.3)
    ax.set_aspect('equal')
    ax.set_facecolor('#2F5233')

    ax.set_xlabel('Field Position (yards)', fontsize=16, weight='bold', color='white')
    ax.set_ylabel('Field Width (yards)', fontsize=16, weight='bold', color='white')

    # Title with stats
    snap_frame = heatmap_data['frame_data']['frame_id'].min() if len(heatmap_data['frame_data']) > 0 else 0
    time_since_snap = (heatmap_data['frame_id'] - snap_frame) / 10.0

    title = (f'{passer_name} → {receiver_name} ({route_type} Route)\n'
            #  f'Time Post-Snap: {time_since_snap:.2f}s | '
             f'Optimal CP: {heatmap_data["max_prob"]:.1%} at '
             f'({heatmap_data["optimal_x"]:.1f}, {heatmap_data["optimal_y"]:.1f})')
    ax.set_title(title, fontsize=18, weight='bold', color='white', pad=20)

    # Legend
    if show_players:
        legend = ax.legend(fontsize=11, loc='upper center', ncol=4,
                          bbox_to_anchor=(0.5, -0.05),
                          framealpha=0.9, edgecolor='white')
        legend.get_frame().set_facecolor('#2F5233')
        for text in legend.get_texts():
            text.set_color('white')

    # Add RTS scores text box if available
    if rts_scores:
        score_text = (
            f"RTS: {rts_scores.get('RTS', 0):.3f}\n"
            f"Temporal: {rts_scores.get('temporal_score', 0):.3f} (Err: {rts_scores.get('temporal_error_abs', 0):.3f})\n"
            f"Spatial: {rts_scores.get('spatial_score', 0):.3f} (Err: {rts_scores.get('spatial_error_abs', 0):.3f})"
        )
        # Add text box in upper left
        ax.text(0.02, 0.98, score_text,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='white', linewidth=2),
                family='monospace',
                color='black',
                weight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, facecolor='#2F5233')
    plt.close()


# ============================================================================
# ANIMATION CREATION FUNCTION
# ============================================================================

def create_animation_from_frames(
    frame_files: List[Path],
    output_path: Path,
    fps: int = 10
):
    """
    Create MP4 video from frame images using OpenCV.

    Args:
        frame_files: List of PNG file paths (in order)
        output_path: Where to save MP4
        fps: Frames per second
    """
    if len(frame_files) == 0:
        raise ValueError("No frames to create animation")

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    height, width, _ = first_frame.shape

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        out.write(frame)

    out.release()


# ============================================================================
# MAIN ORCHESTRATION FUNCTION
# ============================================================================

def visualize_play(
    game_id: int,
    play_id: int,
    input_dir: str = 'data/raw/analytics/train',
    supp_data_path: str = 'data/raw/analytics/supplementary_data.csv',
    model_path: str = 'recap_model.pt',
    rts_scores_path: str = 'results/rts_scores/rts_scored_plays.csv',
    output_dir: str = 'visualizations',
    grid_size: float = 2.0,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    fps: int = 10,
    device: str = None
):
    """
    Main function to generate all visualizations for a play.

    Args:
        game_id: Game ID to analyze
        play_id: Play ID to analyze
        input_dir: Directory containing input tracking data
        supp_data_path: Path to supplementary data CSV
        model_path: Path to model checkpoint
        rts_scores_path: Path to RTS scores CSV
        output_dir: Base directory for outputs
        grid_size: Grid spacing for heatmap (yards)
        x_range: (min_x, max_x) for heatmap. If None, uses full field (10, 110)
        y_range: (min_y, max_y) for heatmap. If None, uses full field (5, 48)
        fps: Frames per second for animation
        device: Device for model (cpu/cuda), auto-detect if None
    """

    print("=" * 80)
    print(f"COMPLETION PROBABILITY VISUALIZER")
    print(f"Game ID: {game_id} | Play ID: {play_id}")
    print("=" * 80)
    print()

    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()

    # 1. Load model
    print(f"Loading model from {model_path}...")
    try:
        predictor, route_to_id = load_model(model_path, device)
        print("✓ Model loaded successfully!")
        print()
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # 2. Load tracking data
    print(f"Loading tracking data for game {game_id}, play {play_id}...")
    try:
        play_data = load_play_tracking_data(game_id, play_id, input_dir)
        print(f"✓ Tracking data loaded: {len(play_data)} rows")
    except Exception as e:
        print(f"✗ Error loading tracking data: {e}")
        return

    # 3. Load supplementary data for route type
    print(f"Loading supplementary data from {supp_data_path}...")
    try:
        supp_df = pd.read_csv(supp_data_path)
        supp_row = supp_df[(supp_df['game_id'] == game_id) & (supp_df['play_id'] == play_id)]

        if len(supp_row) > 0:
            route_type = supp_row.iloc[0].get('route_of_targeted_receiver', 'UNKNOWN')
            print(f"✓ Route type: {route_type}")
        else:
            route_type = 'UNKNOWN'
            print(f"⚠ Play not found in supplementary data, using route_type='UNKNOWN'")
    except Exception as e:
        print(f"⚠ Could not load supplementary data: {e}")
        route_type = 'UNKNOWN'
    print()

    # 4. Load RTS scores
    print(f"Loading RTS scores from {rts_scores_path}...")
    rts_scores = None
    try:
        rts_df = pd.read_csv(rts_scores_path)
        rts_row = rts_df[(rts_df['game_id'] == game_id) & (rts_df['play_id'] == play_id)]

        if len(rts_row) > 0:
            rts_scores = {
                'RTS': rts_row.iloc[0].get('RTS_scaled', 0),
                'temporal_error_abs': rts_row.iloc[0].get('temporal_error_abs', 0),
                'spatial_error_abs': rts_row.iloc[0].get('spatial_error_abs', 0),
                'temporal_score': rts_row.iloc[0].get('temporal_score_scaled', 0),
                'spatial_score': rts_row.iloc[0].get('spatial_score_scaled', 0),

                # 🔽 ADD THESE
                'optimal_x_rts': rts_row.iloc[0].get('optimal_x', None),
                'optimal_y_rts': rts_row.iloc[0].get('optimal_y', None),
            }
            print(f"✓ RTS scores loaded: RTS (scaled)={rts_scores['RTS']:.3f}")
        else:
            print(f"⚠ Play not found in RTS scores file")
    except Exception as e:
        print(f"⚠ Could not load RTS scores: {e}")
    print()

    # 4. Preprocess data
    print("Preprocessing tracking data...")
    play_data = convert_tracking_to_cartesian(play_data)
    play_data = standardize_tracking_directions(play_data)
    play_data['route_type'] = route_type

    # Extract metadata
    min_frame = play_data['frame_id'].min()
    max_frame = play_data['frame_id'].max()
    num_frames = max_frame - min_frame + 1
    actual_target_x = play_data['ball_land_x'].iloc[0]
    actual_target_y = play_data['ball_land_y'].iloc[0]

    # Extract player names
    passer_row = play_data[play_data['player_role'] == 'Passer']
    receiver_row = play_data[play_data['player_role'] == 'Targeted Receiver']

    passer_name = passer_row['player_name'].iloc[0] if len(passer_row) > 0 else 'Unknown'
    receiver_name = receiver_row['player_name'].iloc[0] if len(receiver_row) > 0 else 'Unknown'

    print(f"✓ Found {num_frames} frames (frame {min_frame} to {max_frame})")
    print(f"✓ Ball landing location: ({actual_target_x:.1f}, {actual_target_y:.1f})")
    print(f"✓ Passer: {passer_name} | Receiver: {receiver_name}")
    print()

    # 5. Create output folder with player names
    passer_sanitized = sanitize_name(passer_name)
    receiver_sanitized = sanitize_name(receiver_name)
    folder_name = f"game_{game_id}_play_{play_id}_{passer_sanitized}_to_{receiver_sanitized}"

    output_folder = Path(output_dir) / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)
    frames_folder = output_folder / "frames"
    frames_folder.mkdir(exist_ok=True)

    print(f"Output folder: {output_folder}")
    print()

    # 6. Generate temporal analysis
    print("=" * 80)
    print("TEMPORAL ANALYSIS")
    print("=" * 80)
    print("Generating temporal analysis (completion probability over time)...")

    try:
        temporal_results = compute_temporal_analysis(
            play_data, predictor, actual_target_x, actual_target_y
        )

        # Save static temporal chart
        print()
        print("Creating static temporal chart...")
        temporal_save_path = output_folder / "temporal_analysis.png"
        plot_temporal_analysis(
            temporal_results,
            temporal_save_path,
            passer_name=passer_name,
            receiver_name=receiver_name,
            route_type=route_type,
            rts_scores=rts_scores
        )
        print(f"✓ Static temporal chart saved to {temporal_save_path}")

        # Create temporal animation
        print()
        print("Creating temporal animation...")
        temporal_animation_path = output_folder / "temporal_animation.mp4"
        create_temporal_animation(
            temporal_results,
            frames_folder,
            temporal_animation_path,
            passer_name=passer_name,
            receiver_name=receiver_name,
            route_type=route_type,
            rts_scores=rts_scores,
            fps=fps
        )

    except Exception as e:
        print(f"✗ Error generating temporal visualizations: {e}")
        import traceback
        traceback.print_exc()
    print()

    # 7. Generate spatial heatmaps
    print("=" * 80)
    print("SPATIAL HEATMAPS")
    print("=" * 80)
    print(f"Generating spatial heatmaps ({num_frames} frames)...")

    # Display range info
    display_x_range = x_range if x_range else (0, 120)
    display_y_range = y_range if y_range else (0, 53.5)
    print(f"Grid size: {grid_size} yards")
    print(f"X range: {display_x_range[0]:.1f} - {display_x_range[1]:.1f} yards (field position)")
    print(f"Y range: {display_y_range[0]:.1f} - {display_y_range[1]:.1f} yards (field width)")
    print()

    heatmap_files = []
    frame_ids = list(range(min_frame, max_frame + 1))

    for idx, frame_id in enumerate(frame_ids):
        try:
            # Generate heatmap data
            heatmap_data = generate_heatmap_for_frame(
                play_data, frame_id, predictor,
                actual_target_x, actual_target_y,
                grid_size=grid_size,
                x_range=x_range,
                y_range=y_range
            )

            # Save heatmap image
            frame_save_path = frames_folder / f"frame_{idx+1:03d}.png"

            # Show actual target only on release frame
            show_actual = (frame_id == max_frame)

            plot_heatmap_frame_matplotlib(
                heatmap_data,
                frame_save_path,
                passer_name=passer_name,
                receiver_name=receiver_name,
                route_type=route_type,
                rts_scores=rts_scores,
                show_players=True,
                show_optimal=True,
                show_actual=show_actual
            )

            heatmap_files.append(frame_save_path)

            print(f"  [{idx+1}/{num_frames}] Frame {frame_id:3d} | "
                  f"Time: {(frame_id - min_frame) / 10.0:.1f}s | "
                  f"Optimal CP: {heatmap_data['max_prob']:.1%} | "
                  f"✓ Saved")

        except Exception as e:
            print(f"  [{idx+1}/{num_frames}] Frame {frame_id:3d} | ✗ Error: {e}")

    print()
    print(f"✓ All heatmaps generated! ({len(heatmap_files)} frames)")
    print()

    # 8. Create animation
    print("=" * 80)
    print("ANIMATION")
    print("=" * 80)
    print("Creating MP4 animation from heatmap frames...")

    try:
        animation_path = output_folder / "heatmap_animation.mp4"
        create_animation_from_frames(heatmap_files, animation_path, fps=fps)
        print(f"✓ Animation saved to {animation_path}")
        print(f"  Duration: {len(heatmap_files) / fps:.1f} seconds at {fps} FPS")
    except Exception as e:
        print(f"✗ Error creating animation: {e}")
        import traceback
        traceback.print_exc()
    print()

    # 9. Print summary
    print("=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"Output folder: {output_folder}")
    print(f"  - temporal_analysis.png (static temporal chart)")
    print(f"  - temporal_animation.mp4 (animated temporal chart)")
    print(f"  - heatmap_animation.mp4 (animated spatial heatmaps)")
    print(f"  - frames/ ({len(heatmap_files)} spatial heatmap images)")
    print(f"  - frames/temporal_frames/ (temporal animation frames)")
    print()
    print("Done!")


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate completion probability visualizations for a play',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full field heatmap (default)
  python completion_probability_visualizer.py --game_id 2023090700 --play_id 56

  # Faster with larger grid size
  python completion_probability_visualizer.py --game_id 2023090700 --play_id 56 --grid_size 3.0

  # Custom field area (e.g., just the right half of the field)
  python completion_probability_visualizer.py --game_id 2023090700 --play_id 56 --x_min 60 --x_max 110

  # Use GPU for faster processing
  python completion_probability_visualizer.py --game_id 2023090700 --play_id 56 --device cuda
        """
    )

    parser.add_argument('--game_id', type=int, required=True,
                       help='Game ID to analyze')
    parser.add_argument('--play_id', type=int, required=True,
                       help='Play ID to analyze')
    parser.add_argument('--grid_size', type=float, default=2.0,
                       help='Grid size for heatmap in yards (default: 2.0). Smaller = more detailed but slower.')
    parser.add_argument('--x_min', type=float, default=None,
                       help='Minimum X coordinate for heatmap (default: 10 yards, full field)')
    parser.add_argument('--x_max', type=float, default=None,
                       help='Maximum X coordinate for heatmap (default: 110 yards, full field)')
    parser.add_argument('--y_min', type=float, default=None,
                       help='Minimum Y coordinate for heatmap (default: 5 yards, full field)')
    parser.add_argument('--y_max', type=float, default=None,
                       help='Maximum Y coordinate for heatmap (default: 48 yards, full field)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for animation (default: 10)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use: cpu or cuda (default: auto-detect)')
    parser.add_argument('--input_dir', type=str, default='data/raw/analytics/train',
                       help='Directory containing input tracking data')
    parser.add_argument('--supp_data_path', type=str, default='data/raw/analytics/supplementary_data.csv',
                       help='Path to supplementary data CSV')
    parser.add_argument('--model_path', type=str, default='recap_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--rts_scores_path', type=str, default='results/rts_scores/rts_scored_plays.csv',
                       help='Path to RTS scores CSV')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Base directory for outputs')

    args = parser.parse_args()

    # Build x_range and y_range tuples if any values specified
    x_range = None
    y_range = None

    if args.x_min is not None or args.x_max is not None:
        x_min = args.x_min if args.x_min is not None else 10
        x_max = args.x_max if args.x_max is not None else 110
        x_range = (x_min, x_max)

    if args.y_min is not None or args.y_max is not None:
        y_min = args.y_min if args.y_min is not None else 5
        y_max = args.y_max if args.y_max is not None else 48
        y_range = (y_min, y_max)

    visualize_play(
        game_id=args.game_id,
        play_id=args.play_id,
        input_dir=args.input_dir,
        supp_data_path=args.supp_data_path,
        model_path=args.model_path,
        rts_scores_path=args.rts_scores_path,
        output_dir=args.output_dir,
        grid_size=args.grid_size,
        x_range=x_range,
        y_range=y_range,
        fps=args.fps,
        device=args.device
    )
