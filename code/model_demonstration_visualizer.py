"""
Model Demonstration Visualizer

Creates a progressive visualization showing how the completion probability model works:
1. Shows the play diagram (players on field)
2. Progressively adds evaluation points with their completion probabilities
3. Culminates in showing the full heatmap

This is useful for explaining the methodology and demonstrating how the model
evaluates completion probability at different target locations.

Usage:
    python model_demonstration_visualizer.py --game_id 2023090700 --play_id 56 --frame_id 100

Author: Big Data Bowl 2026 Project
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
    """Sanitize player name for use in file paths."""
    import re
    sanitized = re.sub(r'[^a-zA-Z0-9\-]', '', name.replace(' ', ''))
    return sanitized if sanitized else 'Unknown'


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_play_tracking_data(game_id: int, play_id: int, input_dir: str) -> pd.DataFrame:
    """Load tracking data for a specific play"""
    input_path = Path(input_dir)
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
# HEATMAP GENERATION
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
    """Generate completion probability heatmap for a single frame"""

    frame_data = play_data[play_data['frame_id'] == frame_id]

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
        'frame_id': frame_id
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_field_base(ax, frame_data: pd.DataFrame, passer_name: str, receiver_name: str,
                    route_type: str, frame_id: int, time_since_snap: float):
    """Plot the base football field with yard lines and markers"""

    # Field background
    ax.set_facecolor('#2F5233')

    # Sidelines
    ax.axhline(y=0, color='white', linewidth=4, linestyle='--', alpha=0.5)
    ax.axhline(y=53.3, color='white', linewidth=4, linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='white', linewidth=4, linestyle='-', alpha=0.8)
    ax.axvline(x=120, color='white', linewidth=4, linestyle='-', alpha=0.8)

    # Yard lines (every 10 yards)
    for x in range(10, 120, 10):
        ax.axvline(x=x, color='white', linewidth=2, alpha=0.6)

    # Hash marks (every 5 yards)
    for x in range(5, 120, 5):
        if x % 10 != 0:
            ax.axvline(x=x, color='white', linewidth=1, alpha=0.3)

    # Yard numbers
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

    # Plot players
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

    # Styling
    ax.set_xlim(-2, 122)
    ax.set_ylim(-3, 56.3)
    ax.set_aspect('equal')
    ax.set_xlabel('Field Position (yards)', fontsize=16, weight='bold', color='white')
    ax.set_ylabel('Field Width (yards)', fontsize=16, weight='bold', color='white')

    title = (f'{passer_name} → {receiver_name} ({route_type} Route)\n'
             f'Frame {frame_id} | Time: {time_since_snap:.1f}s')
    ax.set_title(title, fontsize=18, weight='bold', color='white', pad=20)


def create_demonstration_frames(
    play_data: pd.DataFrame,
    frame_id: int,
    predictor: CounterfactualCompletionPredictor,
    actual_target_x: float,
    actual_target_y: float,
    passer_name: str,
    receiver_name: str,
    route_type: str,
    frames_folder: Path,
    grid_size: float = 2.0,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None
) -> List[Path]:
    """
    Create progressive demonstration frames showing:
    1. Just the play diagram
    2. Add evaluation point 1 (near receiver)
    3. Add evaluation point 2 (suboptimal location)
    4. Add actual landing spot
    5. Show full heatmap
    """

    frame_data = play_data[play_data['frame_id'] == frame_id]
    snap_frame = play_data['frame_id'].min()
    time_since_snap = (frame_id - snap_frame) / 10.0

    # Get receiver position for point 1
    receiver_data = frame_data[frame_data['player_role'] == 'Targeted Receiver']
    if len(receiver_data) > 0:
        receiver_x = receiver_data.iloc[0]['x']
        receiver_y = receiver_data.iloc[0]['y']
    else:
        receiver_x = actual_target_x
        receiver_y = actual_target_y

    # Define evaluation points
    # Point 1: At receiver's current location
    point1_x, point1_y = receiver_x, receiver_y

    # Point 2: Suboptimal location (5 yards behind and to the side)
    point2_x = receiver_x + 3
    point2_y = min(max(receiver_y - 8, 5), 48)  # Keep in bounds

    # Point 3: Actual landing spot
    point3_x, point3_y = actual_target_x, actual_target_y

    # Get probabilities for each point
    try:
        pred1 = predictor.predict_at_frame(play_data, frame_id, point1_x, point1_y)
        prob1 = pred1['completion_prob']
    except:
        prob1 = 0.0

    try:
        pred2 = predictor.predict_at_frame(play_data, frame_id, point2_x, point2_y)
        prob2 = pred2['completion_prob']
    except:
        prob2 = 0.0

    try:
        pred3 = predictor.predict_at_frame(play_data, frame_id, point3_x, point3_y)
        prob3 = pred3['completion_prob']
    except:
        prob3 = 0.0

    frame_files = []

    # Frame 1: Just the play diagram
    print(f"  Generating frame 1/5: Play diagram")
    fig, ax = plt.subplots(figsize=(20, 10), facecolor='#2F5233')
    plot_field_base(ax, frame_data, passer_name, receiver_name, route_type, frame_id, time_since_snap)

    # Add explanation text
    ax.text(0.5, 0.95, "Step 1: Play Snapshot",
            transform=ax.transAxes, fontsize=20, weight='bold',
            ha='center', va='top', color='yellow',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='yellow', linewidth=3))

    legend = ax.legend(fontsize=11, loc='upper center', ncol=4,
                      bbox_to_anchor=(0.5, -0.05), framealpha=0.9, edgecolor='white')
    legend.get_frame().set_facecolor('#2F5233')
    for text in legend.get_texts():
        text.set_color('white')

    plt.tight_layout()
    frame1_path = frames_folder / "demo_frame_01.png"
    plt.savefig(frame1_path, dpi=100, facecolor='#2F5233')
    plt.close()
    frame_files.append(frame1_path)

    # Frame 2: Add point 1
    print(f"  Generating frame 2/5: Add evaluation point 1 (at receiver)")
    fig, ax = plt.subplots(figsize=(20, 10), facecolor='#2F5233')
    plot_field_base(ax, frame_data, passer_name, receiver_name, route_type, frame_id, time_since_snap)

    ax.scatter(point1_x, point1_y, s=800, c='cyan', marker='o',
              edgecolors='black', linewidths=3, zorder=15, label=f'Eval Point 1: {prob1:.1%}')
    ax.text(point1_x, point1_y - 3, f'{prob1:.1%}', ha='center', va='top',
           fontsize=14, weight='bold', color='cyan',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='cyan', linewidth=2))

    ax.text(0.5, 0.95, "Step 2: Model evaluates completion probability at receiver's location",
            transform=ax.transAxes, fontsize=20, weight='bold',
            ha='center', va='top', color='cyan',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='cyan', linewidth=3))

    legend = ax.legend(fontsize=11, loc='upper center', ncol=5,
                      bbox_to_anchor=(0.5, -0.05), framealpha=0.9, edgecolor='white')
    legend.get_frame().set_facecolor('#2F5233')
    for text in legend.get_texts():
        text.set_color('white')

    plt.tight_layout()
    frame2_path = frames_folder / "demo_frame_02.png"
    plt.savefig(frame2_path, dpi=100, facecolor='#2F5233')
    plt.close()
    frame_files.append(frame2_path)

    # Frame 3: Add point 2
    print(f"  Generating frame 3/5: Add evaluation point 2 (suboptimal)")
    fig, ax = plt.subplots(figsize=(20, 10), facecolor='#2F5233')
    plot_field_base(ax, frame_data, passer_name, receiver_name, route_type, frame_id, time_since_snap)

    ax.scatter(point1_x, point1_y, s=800, c='cyan', marker='o',
              edgecolors='black', linewidths=3, zorder=15, label=f'Eval Point 1: {prob1:.1%}')
    ax.text(point1_x, point1_y - 3, f'{prob1:.1%}', ha='center', va='top',
           fontsize=14, weight='bold', color='cyan',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='cyan', linewidth=2))

    ax.scatter(point2_x, point2_y, s=800, c='orange', marker='o',
              edgecolors='black', linewidths=3, zorder=15, label=f'Eval Point 2: {prob2:.1%}')
    ax.text(point2_x, point2_y - 3, f'{prob2:.1%}', ha='center', va='top',
           fontsize=14, weight='bold', color='orange',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='orange', linewidth=2))

    ax.text(0.5, 0.95, "Step 3: Model evaluates different target locations with varying probabilities",
            transform=ax.transAxes, fontsize=20, weight='bold',
            ha='center', va='top', color='orange',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='orange', linewidth=3))

    legend = ax.legend(fontsize=11, loc='upper center', ncol=6,
                      bbox_to_anchor=(0.5, -0.05), framealpha=0.9, edgecolor='white')
    legend.get_frame().set_facecolor('#2F5233')
    for text in legend.get_texts():
        text.set_color('white')

    plt.tight_layout()
    frame3_path = frames_folder / "demo_frame_03.png"
    plt.savefig(frame3_path, dpi=100, facecolor='#2F5233')
    plt.close()
    frame_files.append(frame3_path)

    # Frame 4: Add actual landing spot
    print(f"  Generating frame 4/5: Add actual landing spot")
    fig, ax = plt.subplots(figsize=(20, 10), facecolor='#2F5233')
    plot_field_base(ax, frame_data, passer_name, receiver_name, route_type, frame_id, time_since_snap)

    ax.scatter(point1_x, point1_y, s=800, c='cyan', marker='o',
              edgecolors='black', linewidths=3, zorder=15, label=f'Eval Point 1: {prob1:.1%}')
    ax.text(point1_x, point1_y - 3, f'{prob1:.1%}', ha='center', va='top',
           fontsize=14, weight='bold', color='cyan',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='cyan', linewidth=2))

    ax.scatter(point2_x, point2_y, s=800, c='orange', marker='o',
              edgecolors='black', linewidths=3, zorder=15, label=f'Eval Point 2: {prob2:.1%}')
    ax.text(point2_x, point2_y - 3, f'{prob2:.1%}', ha='center', va='top',
           fontsize=14, weight='bold', color='orange',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='orange', linewidth=2))

    ax.scatter(point3_x, point3_y, s=800, c='yellow', marker='X',
              edgecolors='black', linewidths=3, zorder=16, label=f'Actual Target: {prob3:.1%}')
    ax.text(point3_x, point3_y + 3, f'{prob3:.1%}', ha='center', va='bottom',
           fontsize=14, weight='bold', color='yellow',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='yellow', linewidth=2))

    ax.text(0.5, 0.95, "Step 4: Actual throw location and its predicted completion probability",
            transform=ax.transAxes, fontsize=20, weight='bold',
            ha='center', va='top', color='yellow',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='yellow', linewidth=3))

    legend = ax.legend(fontsize=11, loc='upper center', ncol=7,
                      bbox_to_anchor=(0.5, -0.05), framealpha=0.9, edgecolor='white')
    legend.get_frame().set_facecolor('#2F5233')
    for text in legend.get_texts():
        text.set_color('white')

    plt.tight_layout()
    frame4_path = frames_folder / "demo_frame_04.png"
    plt.savefig(frame4_path, dpi=100, facecolor='#2F5233')
    plt.close()
    frame_files.append(frame4_path)

    # Frame 5: Full heatmap
    print(f"  Generating frame 5/5: Full heatmap")
    print(f"    Computing heatmap (this may take a moment)...")
    heatmap_data = generate_heatmap_for_frame(
        play_data, frame_id, predictor,
        actual_target_x, actual_target_y,
        grid_size=grid_size,
        x_range=x_range,
        y_range=y_range
    )

    fig, ax = plt.subplots(figsize=(20, 10), facecolor='#2F5233')

    # Plot heatmap
    X, Y = np.meshgrid(heatmap_data['x_grid'], heatmap_data['y_grid'])
    im = ax.contourf(X, Y, heatmap_data['probabilities'],
                     levels=20, cmap='RdYlGn', alpha=0.7)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Completion Probability', fontsize=14, weight='bold', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    plot_field_base(ax, frame_data, passer_name, receiver_name, route_type, frame_id, time_since_snap)

    # Mark optimal location
    ax.scatter(heatmap_data['optimal_x'], heatmap_data['optimal_y'],
              s=600, c='lime', marker='*',
              label=f"Optimal ({heatmap_data['max_prob']:.1%})",
              zorder=15, edgecolors='black', linewidths=3)

    # Mark actual throw
    ax.scatter(actual_target_x, actual_target_y,
              s=600, c='yellow', marker='X',
              label=f'Actual Target: {prob3:.1%}', zorder=15,
              edgecolors='black', linewidths=3)

    ax.text(0.5, 0.95, "Step 5: Full heatmap showing completion probability across all field locations",
            transform=ax.transAxes, fontsize=20, weight='bold',
            ha='center', va='top', color='lime',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='lime', linewidth=3))

    legend = ax.legend(fontsize=11, loc='upper center', ncol=6,
                      bbox_to_anchor=(0.5, -0.05), framealpha=0.9, edgecolor='white')
    legend.get_frame().set_facecolor('#2F5233')
    for text in legend.get_texts():
        text.set_color('white')

    plt.tight_layout()
    frame5_path = frames_folder / "demo_frame_05.png"
    plt.savefig(frame5_path, dpi=100, facecolor='#2F5233')
    plt.close()
    frame_files.append(frame5_path)

    return frame_files


def create_animation_from_frames(frame_files: List[Path], output_path: Path, fps: int = 1):
    """Create MP4 video from frame images"""
    if len(frame_files) == 0:
        raise ValueError("No frames to create animation")

    first_frame = cv2.imread(str(frame_files[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        # Repeat each frame to make it stay longer
        for _ in range(2):  # Show each frame for 2 seconds at 1 FPS
            out.write(frame)

    out.release()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def create_model_demonstration(
    game_id: int,
    play_id: int,
    frame_id: Optional[int] = None,
    input_dir: str = 'data/raw/analytics/train',
    supp_data_path: str = 'data/raw/analytics/supplementary_data.csv',
    model_path: str = 'recap_model.pt',
    output_dir: str = 'visualizations/demonstration',
    grid_size: float = 2.0,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    fps: int = 1,
    device: str = None
):
    """
    Create model demonstration visualization.

    Args:
        game_id: Game ID
        play_id: Play ID
        frame_id: Specific frame to visualize (if None, uses release frame)
        input_dir: Input data directory
        supp_data_path: Supplementary data path
        model_path: Model checkpoint path
        output_dir: Output directory
        grid_size: Heatmap grid size
        x_range: X-axis range for heatmap
        y_range: Y-axis range for heatmap
        fps: Frames per second for animation
        device: Device (cpu/cuda)
    """

    print("=" * 80)
    print(f"MODEL DEMONSTRATION VISUALIZER")
    print(f"Game ID: {game_id} | Play ID: {play_id}")
    print("=" * 80)
    print()

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()

    # Load model
    print(f"Loading model from {model_path}...")
    try:
        predictor, route_to_id = load_model(model_path, device)
        print("✓ Model loaded successfully!")
        print()
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # Load tracking data
    print(f"Loading tracking data...")
    try:
        play_data = load_play_tracking_data(game_id, play_id, input_dir)
        print(f"✓ Tracking data loaded: {len(play_data)} rows")
    except Exception as e:
        print(f"✗ Error loading tracking data: {e}")
        return

    # Load supplementary data
    print(f"Loading supplementary data...")
    try:
        supp_df = pd.read_csv(supp_data_path)
        supp_row = supp_df[(supp_df['game_id'] == game_id) & (supp_df['play_id'] == play_id)]

        if len(supp_row) > 0:
            route_type = supp_row.iloc[0].get('route_of_targeted_receiver', 'UNKNOWN')
            print(f"✓ Route type: {route_type}")
        else:
            route_type = 'UNKNOWN'
            print(f"⚠ Play not found in supplementary data")
    except Exception as e:
        print(f"⚠ Could not load supplementary data: {e}")
        route_type = 'UNKNOWN'
    print()

    # Preprocess
    print("Preprocessing data...")
    play_data = convert_tracking_to_cartesian(play_data)
    play_data = standardize_tracking_directions(play_data)
    play_data['route_type'] = route_type

    min_frame = play_data['frame_id'].min()
    max_frame = play_data['frame_id'].max()
    actual_target_x = play_data['ball_land_x'].iloc[0]
    actual_target_y = play_data['ball_land_y'].iloc[0]

    # Use specified frame or release frame
    if frame_id is None:
        frame_id = max_frame
        print(f"✓ Using release frame: {frame_id}")
    else:
        print(f"✓ Using specified frame: {frame_id}")

    passer_row = play_data[play_data['player_role'] == 'Passer']
    receiver_row = play_data[play_data['player_role'] == 'Targeted Receiver']

    passer_name = passer_row['player_name'].iloc[0] if len(passer_row) > 0 else 'Unknown'
    receiver_name = receiver_row['player_name'].iloc[0] if len(receiver_row) > 0 else 'Unknown'

    print(f"✓ Passer: {passer_name} | Receiver: {receiver_name}")
    print()

    # Create output folder
    passer_sanitized = sanitize_name(passer_name)
    receiver_sanitized = sanitize_name(receiver_name)
    folder_name = f"demo_game_{game_id}_play_{play_id}_frame_{frame_id}"

    output_folder = Path(output_dir) / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)
    frames_folder = output_folder / "frames"
    frames_folder.mkdir(exist_ok=True)

    print(f"Output folder: {output_folder}")
    print()

    # Create demonstration frames
    print("=" * 80)
    print("CREATING DEMONSTRATION")
    print("=" * 80)

    frame_files = create_demonstration_frames(
        play_data=play_data,
        frame_id=frame_id,
        predictor=predictor,
        actual_target_x=actual_target_x,
        actual_target_y=actual_target_y,
        passer_name=passer_name,
        receiver_name=receiver_name,
        route_type=route_type,
        frames_folder=frames_folder,
        grid_size=grid_size,
        x_range=x_range,
        y_range=y_range
    )

    print()
    print(f"✓ All frames generated!")
    print()

    # Create animation
    print("Creating animation...")
    animation_path = output_folder / "model_demonstration.mp4"
    create_animation_from_frames(frame_files, animation_path, fps=fps)

    print(f"✓ Animation saved to {animation_path}")
    print()

    print("=" * 80)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print(f"Output folder: {output_folder}")
    print(f"  - frames/ (5 demonstration frames)")
    print(f"  - model_demonstration.mp4")
    print()
    print("Done!")


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create model demonstration visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use release frame (default)
  python model_demonstration_visualizer.py --game_id 2023090700 --play_id 56

  # Use specific frame
  python model_demonstration_visualizer.py --game_id 2023090700 --play_id 56 --frame_id 100

  # Faster heatmap generation with larger grid
  python model_demonstration_visualizer.py --game_id 2023090700 --play_id 56 --grid_size 3.0
        """
    )

    parser.add_argument('--game_id', type=int, required=True, help='Game ID')
    parser.add_argument('--play_id', type=int, required=True, help='Play ID')
    parser.add_argument('--frame_id', type=int, default=None,
                       help='Frame ID (default: use release frame)')
    parser.add_argument('--grid_size', type=float, default=2.0,
                       help='Heatmap grid size in yards (default: 2.0)')
    parser.add_argument('--x_min', type=float, default=None, help='Min X coordinate')
    parser.add_argument('--x_max', type=float, default=None, help='Max X coordinate')
    parser.add_argument('--y_min', type=float, default=None, help='Min Y coordinate')
    parser.add_argument('--y_max', type=float, default=None, help='Max Y coordinate')
    parser.add_argument('--fps', type=int, default=1,
                       help='Animation speed (default: 1 FPS, each frame shown ~2 seconds)')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda)')
    parser.add_argument('--input_dir', type=str, default='data/raw/analytics/train',
                       help='Input data directory')
    parser.add_argument('--supp_data_path', type=str,
                       default='data/raw/analytics/supplementary_data.csv',
                       help='Supplementary data path')
    parser.add_argument('--model_path', type=str,
                       default='data/recap_model.pt',
                       help='Model checkpoint path')
    parser.add_argument('--output_dir', type=str, default='visualizations/demonstration',
                       help='Output directory')

    args = parser.parse_args()

    x_range = None
    y_range = None

    if args.x_min is not None or args.x_max is not None:
        x_min = args.x_min if args.x_min is not None else 0
        x_max = args.x_max if args.x_max is not None else 120
        x_range = (x_min, x_max)

    if args.y_min is not None or args.y_max is not None:
        y_min = args.y_min if args.y_min is not None else 0
        y_max = args.y_max if args.y_max is not None else 53.5
        y_range = (y_min, y_max)

    create_model_demonstration(
        game_id=args.game_id,
        play_id=args.play_id,
        frame_id=args.frame_id,
        input_dir=args.input_dir,
        supp_data_path=args.supp_data_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        grid_size=args.grid_size,
        x_range=x_range,
        y_range=y_range,
        fps=args.fps,
        device=args.device
    )
