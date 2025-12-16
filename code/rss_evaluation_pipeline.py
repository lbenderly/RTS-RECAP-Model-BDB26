"""
Route Synchronization Score (RSS) Evaluation Pipeline Before RTS Score Calculation

This script evaluates plays using the RECAP completion probability model to compute:
- TSV (Timing Success Value): How well QB timed the throw
- PSV (Placement Success Value): How well QB placed the ball
- RSS (Route Synchronization Score): Combined QB-WR synchronization metric


"""



import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Data paths
INPUT_DATA_DIR = Path("data/raw/analytics/train")
SUPPLEMENTARY_DATA_PATH = INPUT_DATA_DIR.parent / "supplementary_data.csv"

# Model path
MODEL_PATH = Path("recap_model.pt")

# Output configuration
OUTPUT_DIR = Path("results/rts_evaluation")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Evaluation configuration
CONFIG = {
    'possession_team': None,     # Filter to specific team (e.g., 'DET' for Lions)
    'route_types': ['SLANT', 'IN', 'OUT', 'HITCH', 'CORNER', 'CROSS', 'POST'],  # None = all routes, or list like ['GO', 'SLANT', 'OUT']
    'spatial_search_radius': 20,  # Yards around actual target
    'spatial_grid_size': 2,       # Grid spacing in yards
    'checkpoint_frequency': 50,   # Save progress every N plays
    'max_plays': 10,            # None = all plays, or set to 100, 200, etc. for testing
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"Using device: {CONFIG['device']}")

# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================

class RouteAwareCompletionModel(nn.Module):
    """
    Route-aware completion probability model with transformer architecture.
    """
    
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
        
        # Transformer encoder for player interactions
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
        combined_dim = (
            player_feature_dim +
            route_embedding_dim +
            32 +  # From momentum encoder
            32    # From NGS encoder
        )
        
        self.completion_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)
    
    def forward(
        self,
        player_features: torch.Tensor,
        route_ids: torch.Tensor,
        momentum_features: torch.Tensor,
        ngs_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass returning logits."""
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
        """Get probabilities instead of logits."""
        logits = self.forward(*args, **kwargs)
        return torch.sigmoid(logits)


# ==============================================================================
# COUNTERFACTUAL PREDICTOR 
# ==============================================================================

class CounterfactualCompletionPredictor:
    """
    Wrapper for making counterfactual predictions.
    Enables "what if" scenario analysis.
    """
    
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
        """
        Predict completion probability for a hypothetical throw.
        
        Args:
            play_data: Full play tracking data (all frames)
            hypothetical_frame: Frame ID where we imagine the throw happens
            target_x: Ball landing x-coordinate
            target_y: Ball landing y-coordinate
        
        Returns:
            dict with completion_prob and auxiliary info
        """
        # Get player positions at hypothetical frame
        frame_data = play_data[play_data['frame_id'] == hypothetical_frame].copy()
        
        if len(frame_data) == 0:
            raise ValueError(f"Frame {hypothetical_frame} not found in play data")
        
        # Get QB data
        qb_data = frame_data[frame_data['player_role'] == 'Passer']
        if len(qb_data) == 0:
            raise ValueError("No passer found in frame data")
        qb_data = qb_data.iloc[0]
        
        qb_x, qb_y = qb_data['x'], qb_data['y']
        
        # Calculate air distance
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
        
        # Calculate receiver momentum alignment
        target_vec_x = target_x - receiver_x
        target_vec_y = target_y - receiver_y
        target_vec_mag = np.sqrt(target_vec_x**2 + target_vec_y**2) + 1e-6
        
        momentum_alignment = (receiver_vx * target_vec_x + receiver_vy * target_vec_y) / target_vec_mag
        orientation_alignment = (receiver_ox * target_vec_x + receiver_oy * target_vec_y) / target_vec_mag
        receiver_to_ball_distance = target_vec_mag
        
        # Get defender positions and calculate target separation
        defenders = frame_data[frame_data['player_role'] == 'Defensive Coverage']
        if len(defenders) > 0:
            def_dists = np.sqrt(
                (defenders['x'] - target_x)**2 +
                (defenders['y'] - target_y)**2
            )
            target_separation = def_dists.min()
        else:
            target_separation = 10.0
        
        # Calculate sideline separation
        sideline_separation = min(target_y, 53.3 - target_y)
        
        # Time calculations
        snap_frame = play_data['frame_id'].min()
        time_to_throw = (hypothetical_frame - snap_frame) / 10.0
        
        # Get route type
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
    
    def analyze_play_timing(
        self,
        play_data: pd.DataFrame,
        target_x: float = None,
        target_y: float = None
    ) -> pd.DataFrame:
        """
        Analyze completion probability across all input frames (snap to release).
        
        Args:
            play_data: Full play tracking data
            target_x, target_y: Target location (defaults to actual ball landing)
        
        Returns:
            DataFrame with completion probabilities for each frame
        """
        # Get actual target location if not provided
        if target_x is None:
            target_x = play_data['ball_land_x'].iloc[0]
        if target_y is None:
            target_y = play_data['ball_land_y'].iloc[0]
        
        # Get frame range (snap to release only)
        min_frame = play_data['frame_id'].min()
        max_frame = play_data['frame_id'].max()
        
        # Analyze all frames
        results = []
        for frame_id in range(min_frame, max_frame + 1):
            try:
                pred = self.predict_at_frame(play_data, frame_id, target_x, target_y)
                results.append({
                    'frame_id': frame_id,
                    'time_since_snap': (frame_id - min_frame) / 10.0,
                    'completion_prob': pred['completion_prob'],
                    'is_actual_release': frame_id == max_frame
                })
            except Exception as e:
                # Skip frames that fail
                pass
        
        return pd.DataFrame(results)
    
    def analyze_play_spatial(
        self,
        play_data: pd.DataFrame,
        release_frame: int,
        x_range: tuple,
        y_range: tuple = (5, 48),
        grid_size: float = 1.0
    ) -> Dict:
        """
        Create spatial grid of completion probabilities.
        
        Args:
            play_data: Full play tracking data
            release_frame: Frame to analyze (actual release)
            x_range: (min_x, max_x) for grid
            y_range: (min_y, max_y) for grid
            grid_size: Grid spacing in yards
        
        Returns:
            dict with optimal location and actual probability
        """
        actual_x = play_data['ball_land_x'].iloc[0]
        actual_y = play_data['ball_land_y'].iloc[0]
        
        # Create grid
        x_grid = np.arange(x_range[0], x_range[1], grid_size)
        y_grid = np.arange(y_range[0], y_range[1], grid_size)
        
        # Find optimal location
        max_prob = -1
        optimal_x, optimal_y = actual_x, actual_y
        
        for y in y_grid:
            for x in x_grid:
                try:
                    pred = self.predict_at_frame(play_data, release_frame, x, y)
                    prob = pred['completion_prob']
                    if prob > max_prob:
                        max_prob = prob
                        optimal_x = x
                        optimal_y = y
                except:
                    pass
        
        # Get actual probability
        try:
            actual_pred = self.predict_at_frame(play_data, release_frame, actual_x, actual_y)
            actual_prob = actual_pred['completion_prob']
        except:
            actual_prob = np.nan
        
        return {
            'optimal_x': optimal_x,
            'optimal_y': optimal_y,
            'optimal_prob': max_prob,
            'actual_x': actual_x,
            'actual_y': actual_y,
            'actual_prob': actual_prob
        }


# ==============================================================================
# DATA LOADING FUNCTIONS
# ==============================================================================

def load_supplementary_data(possession_team: str = None) -> pl.DataFrame:
    """Load supplementary data with completion labels and route information."""
    df = pl.read_csv(
        SUPPLEMENTARY_DATA_PATH,
        null_values=["NA", "nan", "N/A", "NaN", ""],
    )
    
    # Filter to plays with pass_result
    df = df.filter(pl.col("pass_result").is_not_null())
    
    # Filter by possession team if specified
    if possession_team is not None:
        df = df.filter(pl.col("possession_team") == possession_team)
        print(f"Filtered to {possession_team} plays: {len(df)} plays")
    
    # Convert pass_result to binary completion
    df = df.with_columns([
        pl.when(pl.col("pass_result") == "C")
        .then(1)
        .otherwise(0)
        .cast(pl.Int32)
        .alias("completion")
    ])
    
    # Handle missing route types
    df = df.with_columns([
        pl.col("route_of_targeted_receiver")
        .fill_null("UNKNOWN")
        .alias("route_type")
    ])
    
    # Select columns including play context
    columns_to_select = ["game_id", "play_id", "completion", "route_type"]

    # Add play context columns if they exist
    context_columns = ["down", "quarter", "week", "yards_to_go", "yards_gained", "expected_points_added",
                      "possession_team", "team_coverage_man_zone"]
    for col in context_columns:
        if col in df.columns:
            columns_to_select.append(col)
    
    df = df.select(columns_to_select)
    
    print(f"Loaded supplementary data: {len(df)} plays")
    print(f"Completion rate: {df['completion'].mean():.1%}")
    
    return df


def load_input_data() -> pl.DataFrame:
    """Load input tracking data (snap to release)."""
    csv_pattern = str(INPUT_DATA_DIR / "input_*.csv")
    df = pl.read_csv(csv_pattern, null_values=["NA", "nan", "N/A", "NaN", ""])
    print(f"Loaded input data: {len(df)} rows")
    print(f"Unique plays: {df.n_unique(['game_id', 'play_id'])}")
    return df


def convert_tracking_to_cartesian(tracking_df: pl.DataFrame) -> pl.DataFrame:
    """Convert polar coordinates to Cartesian."""
    return (
        tracking_df.with_columns(
            dir_adjusted=((pl.col("dir") - 90) * -1) % 360,
            o_adjusted=((pl.col("o") - 90) * -1) % 360,
        )
        .with_columns(
            vx=pl.col("s") * pl.col("dir_adjusted").radians().cos(),
            vy=pl.col("s") * pl.col("dir_adjusted").radians().sin(),
            ox=pl.col("o_adjusted").radians().cos(),
            oy=pl.col("o_adjusted").radians().sin(),
        )
        .drop(["dir_adjusted", "o_adjusted"])
    )



def standardize_tracking_directions(tracking_df: pl.DataFrame) -> pl.DataFrame:
    """Standardize play directions to always moving left to right."""
    return tracking_df.with_columns(
        x=pl.when(pl.col("play_direction") == "right").then(pl.col("x")).otherwise(120 - pl.col("x")),
        y=pl.when(pl.col("play_direction") == "right").then(pl.col("y")).otherwise(53.3 - pl.col("y")),
        vx=pl.when(pl.col("play_direction") == "right").then(pl.col("vx")).otherwise(-1 * pl.col("vx")),
        vy=pl.when(pl.col("play_direction") == "right").then(pl.col("vy")).otherwise(-1 * pl.col("vy")),
        ox=pl.when(pl.col("play_direction") == "right").then(pl.col("ox")).otherwise(-1 * pl.col("ox")),
        oy=pl.when(pl.col("play_direction") == "right").then(pl.col("oy")).otherwise(-1 * pl.col("oy")),
        ball_land_x=pl.when(pl.col("play_direction") == "right").then(pl.col("ball_land_x")).otherwise(120 - pl.col("ball_land_x")),
        ball_land_y=pl.when(pl.col("play_direction") == "right").then(pl.col("ball_land_y")).otherwise(53.3 - pl.col("ball_land_y")),
    ).drop("play_direction")


def prepare_play_dataframe(
    tracking_df: pl.DataFrame,
    game_id: int,
    play_id: int,
    supp_df: pl.DataFrame
) -> pd.DataFrame:
    """
    Prepare single play data for prediction.
    
    Returns pandas DataFrame with route_type added.
    """
    # Filter to this play
    play_df = tracking_df.filter(
        (pl.col('game_id') == game_id) &
        (pl.col('play_id') == play_id)
    )
    
    # Get route type from supplementary data
    route_info = supp_df.filter(
        (pl.col('game_id') == game_id) &
        (pl.col('play_id') == play_id)
    )
    
    if len(route_info) == 0:
        route_type = 'UNKNOWN'
    else:
        route_type = route_info['route_type'].iloc[0] if hasattr(route_info['route_type'], 'iloc') else route_info['route_type'][0]
    
    # Convert to pandas and add route_type
    play_pd = play_df.to_pandas()
    play_pd['route_type'] = route_type
    
    return play_pd


# ==============================================================================
# RTS EVALUATION FUNCTIONS
# ==============================================================================

def compute_tsv(
    play_data: pd.DataFrame,
    predictor: CounterfactualCompletionPredictor,
    actual_target_x: float,
    actual_target_y: float,
    release_frame: int
) -> Dict:
    """
    Compute Timing Success Value (TSV).
    
    Returns:
        dict with TSV metrics
    """
    # Analyze timing across all input frames
    timing_results = predictor.analyze_play_timing(
        play_data=play_data,
        target_x=actual_target_x,
        target_y=actual_target_y
    )
    
    if len(timing_results) == 0:
        return {
            'CP_peak': np.nan,
            'CP_actual_timing': np.nan,
            'optimal_frame': np.nan,
            'optimal_time_since_snap': np.nan,
            'TSV': np.nan
        }
    
    # Find peak
    CP_peak = timing_results['completion_prob'].max()
    optimal_idx = timing_results['completion_prob'].idxmax()
    optimal_frame = timing_results.loc[optimal_idx, 'frame_id']
    optimal_time = timing_results.loc[optimal_idx, 'time_since_snap']
    
    # Get actual
    actual_rows = timing_results[timing_results['frame_id'] == release_frame]
    if len(actual_rows) == 0:
        CP_actual_timing = np.nan
    else:
        CP_actual_timing = actual_rows['completion_prob'].values[0]
    
    # Calculate TSV
    TSV = CP_actual_timing - CP_peak
    
    return {
        'CP_peak': CP_peak,
        'CP_actual_timing': CP_actual_timing,
        'optimal_frame': optimal_frame,
        'optimal_time_since_snap': optimal_time,
        'TSV': TSV
    }


def compute_psv(
    play_data: pd.DataFrame,
    predictor: CounterfactualCompletionPredictor,
    actual_target_x: float,
    actual_target_y: float,
    release_frame: int,
    search_radius: float,
    grid_size: float
) -> Dict:
    """
    Compute Placement Success Value (PSV).
    
    Returns:
        dict with PSV metrics
    """
    # Define search space
    x_min = max(10, actual_target_x - search_radius)
    x_max = min(110, actual_target_x + search_radius)
    
    # Analyze spatial grid
    spatial_results = predictor.analyze_play_spatial(
        play_data=play_data,
        release_frame=release_frame,
        x_range=(x_min, x_max),
        y_range=(5, 48),
        grid_size=grid_size
    )
    
    # Extract metrics
    CP_optimal = spatial_results['optimal_prob']
    optimal_x = spatial_results['optimal_x']
    optimal_y = spatial_results['optimal_y']
    CP_actual_placement = spatial_results['actual_prob']
    
    # Calculate spatial error
    spatial_error = np.sqrt(
        (optimal_x - actual_target_x)**2 +
        (optimal_y - actual_target_y)**2
    )
    
    # Calculate PSV
    PSV = CP_actual_placement - CP_optimal
    
    return {
        'CP_optimal': CP_optimal,
        'CP_actual_placement': CP_actual_placement,
        'optimal_x': optimal_x,
        'optimal_y': optimal_y,
        'spatial_error': spatial_error,
        'PSV': PSV
    }


def compute_rss(tsv_metrics: Dict, psv_metrics: Dict) -> Dict:
    """
    Compute Route Synchronization Score (RSS).
    
    Returns:
        dict with RSS metrics
    """
    TSV = tsv_metrics['TSV']
    PSV = psv_metrics['PSV']
    
    # Synchronization Value
    SV = (TSV + PSV) / 2
    
    # Difficulty Multiplier
    CP_peak = tsv_metrics['CP_peak']
    CP_optimal = psv_metrics['CP_optimal']
    CP_optimal_avg = (CP_peak + CP_optimal) / 2
    DM = 1.0 / (CP_optimal_avg + 0.1)
    
    # Final RSS
    RSS = SV * DM
    
    return {
        'SV': SV,
        'DM': DM,
        'RSS': RSS
    }


def evaluate_single_play(
    game_id: int,
    play_id: int,
    tracking_df: pl.DataFrame,
    supp_df: pl.DataFrame,
    predictor: CounterfactualCompletionPredictor,
    config: Dict
) -> Dict:
    """
    Evaluate a single play and return RSS metrics.
    
    Returns:
        dict with all metrics, or None if play fails
    """
    try:
        # Prepare play data
        play_data = prepare_play_dataframe(tracking_df, game_id, play_id, supp_df)
        
        if len(play_data) == 0:
            return None
        
        # Extract play metadata
        release_frame = play_data['frame_id'].max()
        snap_frame = play_data['frame_id'].min()
        actual_target_x = play_data['ball_land_x'].iloc[0]
        actual_target_y = play_data['ball_land_y'].iloc[0]
        route_type = play_data['route_type'].iloc[0]

        # Extract passer and targeted receiver names
        passer_rows = play_data[play_data['player_role'] == 'Passer']
        passer_name = passer_rows['player_name'].iloc[0] if len(passer_rows) > 0 else np.nan

        receiver_rows = play_data[play_data['player_role'] == 'Targeted Receiver']
        receiver_name = receiver_rows['player_name'].iloc[0] if len(receiver_rows) > 0 else np.nan
        
        #    Get play context from supplementary data
        play_info = supp_df.filter(
            (pl.col('game_id') == game_id) &
            (pl.col('play_id') == play_id)
        )
        
        if len(play_info) == 0:
            return None
        
        # Extract play context (handle both Polars DataFrame and Series)
        def safe_extract(df, col, default=np.nan):
            if col not in df.columns:
                return default
            val = df[col]
            if hasattr(val, 'iloc'):
                return val.iloc[0] if len(val) > 0 else default
            elif hasattr(val, '__getitem__'):
                return val[0] if len(val) > 0 else default
            return default
        
        actual_completion = safe_extract(play_info, 'completion', np.nan)
        down = safe_extract(play_info, 'down', np.nan)
        quarter = safe_extract(play_info, 'quarter', np.nan)
        week = safe_extract(play_info, 'week', np.nan)
        yards_to_go = safe_extract(play_info, 'yards_to_go', np.nan)
        yards_gained = safe_extract(play_info, 'yards_gained', np.nan)
        expected_points_added = safe_extract(play_info, 'expected_points_added', np.nan)
        possession_team = safe_extract(play_info, 'possession_team', 'UNK')
        team_coverage_man_zone = safe_extract(play_info, 'team_coverage_man_zone', 'UNK')
        
        # Compute TSV
        tsv_metrics = compute_tsv(
            play_data, predictor, actual_target_x, actual_target_y, release_frame
        )

        # Compute temporal error (frames and seconds off from optimal timing)
        if not np.isnan(tsv_metrics.get('optimal_frame', np.nan)):
            temporal_error_frames = release_frame - tsv_metrics['optimal_frame']
            temporal_error_seconds = temporal_error_frames / 10.0
        else:
            temporal_error_frames = np.nan
            temporal_error_seconds = np.nan

        # Compute PSV
        psv_metrics = compute_psv(
            play_data, predictor, actual_target_x, actual_target_y, release_frame,
            config['spatial_search_radius'], config['spatial_grid_size']
        )

        # Compute RSS
        rss_metrics = compute_rss(tsv_metrics, psv_metrics)
        
        # Combine all results
        # Combine all results
        result = {
            'game_id': game_id,
            'play_id': play_id,
            'passer': passer_name,
            'targeted_receiver': receiver_name,
            'week': week,
            'quarter': quarter,
            'possession_team': possession_team,
            'down': down,
            'yards_to_go': yards_to_go,
            'team_coverage_man_zone': team_coverage_man_zone,
            'yards_gained': yards_gained,
            'expected_points_added': expected_points_added,
            'route_type': route_type,
            'actual_completion': actual_completion,
            'actual_x': actual_target_x,
            'actual_y': actual_target_y,
            'release_frame': release_frame,
            'snap_frame': snap_frame,
            'actual_time_since_snap': (release_frame - snap_frame) / 10.0,
            'temporal_error_frames': temporal_error_frames,
            'temporal_error_seconds': temporal_error_seconds,
        }
        
        # Add TSV metrics
        result.update(tsv_metrics)
        
        # Add PSV metrics
        result.update(psv_metrics)
        
        # Add RSS metrics
        result.update(rss_metrics)
        
        return result
        
    except Exception as e:
        print(f"Error evaluating play {game_id}-{play_id}: {e}")
        return None


def batch_evaluate_plays(
    plays_list: List[Tuple[int, int]],
    tracking_df: pl.DataFrame,
    supp_df: pl.DataFrame,
    predictor: CounterfactualCompletionPredictor,
    config: Dict
) -> pd.DataFrame:
    """
    Evaluate a batch of plays and return results DataFrame.
    
    Args:
        plays_list: List of (game_id, play_id) tuples
        tracking_df: Tracking data
        supp_df: Supplementary data
        predictor: Counterfactual predictor
        config: Configuration dict
    
    Returns:
        DataFrame with all results
    """
    results = []
    failed_plays = []
    
    checkpoint_dir = OUTPUT_DIR / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    
    for idx, (game_id, play_id) in enumerate(tqdm(plays_list, desc="Evaluating plays")):
        result = evaluate_single_play(
            game_id, play_id, tracking_df, supp_df, predictor, config
        )
        
        if result is not None:
            results.append(result)
        else:
            failed_plays.append((game_id, play_id))
        
        # Checkpoint
        if (idx + 1) % config['checkpoint_frequency'] == 0:
            checkpoint_path = checkpoint_dir / f"progress_{idx+1}.pkl"
            pd.DataFrame(results).to_pickle(checkpoint_path)
            
            elapsed = time.time() - start_time
            plays_per_sec = (idx + 1) / elapsed
            remaining = len(plays_list) - (idx + 1)
            est_remaining = remaining / plays_per_sec if plays_per_sec > 0 else 0
            
            print(f"\nCheckpoint {idx+1}/{len(plays_list)}")
            print(f"  Success: {len(results)}, Failed: {len(failed_plays)}")
            print(f"  Elapsed: {elapsed/60:.1f} min, Est. remaining: {est_remaining/60:.1f} min")
    
    # Save failed plays log
    if failed_plays:
        failed_df = pd.DataFrame(failed_plays, columns=['game_id', 'play_id'])
        failed_df.to_csv(OUTPUT_DIR / 'failed_plays.csv', index=False)
        print(f"\nFailed plays: {len(failed_plays)} (logged to failed_plays.csv)")
    
    return pd.DataFrame(results)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function."""
    
    print("="*80)
    print("ROUTE SYNCHRONIZATION SCORE (RSS) EVALUATION PIPELINE")
    print("="*80)
    print(f"\nStarting evaluation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {CONFIG['device']}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # =========================================================================
    # 1. LOAD MODEL
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: LOADING MODEL")
    print("="*80)
    
    print(f"Loading model from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=CONFIG['device'], weights_only=False)
    
    route_to_id = checkpoint['route_to_id']
    num_routes = checkpoint['num_routes']
    
    print(f"Routes loaded: {num_routes}")
    
    # Initialize model
    model = RouteAwareCompletionModel(
        num_routes=num_routes,
        route_embedding_dim=32,
        player_feature_dim=256,
        num_transformer_layers=4,
        num_attention_heads=8,
        dropout=0.1
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(CONFIG['device'])
    model.eval()
    
    print("✓ Model loaded successfully")
    
    # Create predictor
    predictor = CounterfactualCompletionPredictor(model, route_to_id, CONFIG['device'])
    print("✓ Predictor initialized")
    
    # =========================================================================
    # 2. LOAD DATA
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: LOADING DATA")
    print("="*80)
    
    print(f"Loading supplementary data (filtering to {CONFIG['possession_team']})...")
    supp_df = load_supplementary_data(possession_team=CONFIG['possession_team'])    
    print("\nLoading tracking data...")
    tracking_df = load_input_data()
    
    print("\nConverting to cartesian coordinates...")
    tracking_df = convert_tracking_to_cartesian(tracking_df)
    
    print("Standardizing play directions...")
    tracking_df = standardize_tracking_directions(tracking_df)
    
    print("✓ Data loaded and preprocessed")
    
    # =========================================================================
    # 3. FILTER PLAYS
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: FILTERING PLAYS")
    print("="*80)
    
    # Get unique plays
    unique_plays = tracking_df.select(['game_id', 'play_id']).unique()
    
    # Join with supplementary data to get only plays that exist in supp_df
    # (which is already filtered to the specified team)
    unique_plays = unique_plays.join(
        supp_df.select(['game_id', 'play_id']),
        on=['game_id', 'play_id'],
        how='inner'
    )
    
    # Filter by route type if specified
    if CONFIG['route_types'] is not None:
        print(f"Filtering to route types: {CONFIG['route_types']}")
        filtered_supp = supp_df.filter(
            pl.col('route_type').is_in(CONFIG['route_types'])
        )
        unique_plays = unique_plays.join(
            filtered_supp.select(['game_id', 'play_id']),
            on=['game_id', 'play_id'],
            how='inner'
        )
    
    plays_list = [(row['game_id'], row['play_id']) for row in unique_plays.iter_rows(named=True)]
    
    # Limit number of plays if specified
    if CONFIG['max_plays'] is not None:
        plays_list = plays_list[:CONFIG['max_plays']]
        print(f"Limiting to first {CONFIG['max_plays']} plays for testing")
    
    print(f"Total plays to evaluate: {len(plays_list)}")
    
    # =========================================================================
    # 4. EVALUATE PLAYS
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: EVALUATING PLAYS")
    print("="*80)
    print(f"Configuration:")
    print(f"  Spatial search radius: {CONFIG['spatial_search_radius']} yards")
    print(f"  Spatial grid size: {CONFIG['spatial_grid_size']} yards")
    print(f"  Checkpoint frequency: {CONFIG['checkpoint_frequency']} plays")
    print()
    
    # Estimate time
    est_seconds_per_play = 60 if CONFIG['device'] == 'cpu' else 15
    est_total_seconds = len(plays_list) * est_seconds_per_play
    est_hours = est_total_seconds / 3600
    print(f"Estimated time: {est_hours:.1f} hours ({est_seconds_per_play}s per play)")
    print()
    
    results_df = batch_evaluate_plays(
        plays_list, tracking_df, supp_df, predictor, CONFIG
    )
    
    print(f"\n✓ Evaluation complete: {len(results_df)} plays successfully evaluated")
    
    # =========================================================================
    # 5. SAVE RESULTS
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: SAVING RESULTS")
    print("="*80)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save full results
    csv_path = OUTPUT_DIR / f'rss_results_{timestamp}.csv'
    pkl_path = OUTPUT_DIR / f'rss_results_{timestamp}.pkl'
    
    results_df.to_csv(csv_path, index=False)
    results_df.to_pickle(pkl_path)
    
    print(f"✓ Results saved:")
    print(f"  CSV: {csv_path}")
    print(f"  PKL: {pkl_path}")
    
    # Generate summary statistics
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    summary = results_df.groupby('route_type').agg({
        'RSS': ['count', 'mean', 'std', 'min', 'max'],
        'TSV': ['mean', 'std'],
        'PSV': ['mean', 'std'],
        'SV': ['mean', 'std'],
        'DM': ['mean', 'std'],
        'actual_completion': 'mean',
        'expected_points_added': 'mean'
    }).round(4)
    
    summary_path = OUTPUT_DIR / f'rss_summary_{timestamp}.csv'
    summary.to_csv(summary_path)
    
    print(summary)
    print(f"\n✓ Summary saved: {summary_path}")
    
    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"Total plays evaluated: {len(results_df)}")
    print(f"Mean RSS: {results_df['RSS'].mean():.4f}")
    print(f"Mean TSV: {results_df['TSV'].mean():.4f}")
    print(f"Mean PSV: {results_df['PSV'].mean():.4f}")
    print(f"Mean Spatial Error: {results_df['spatial_error'].mean():.2f} yards")
    print(f"Completion Rate: {results_df['actual_completion'].mean():.1%}")
    
    print("\n" + "="*80)
    print(f"EVALUATION COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()