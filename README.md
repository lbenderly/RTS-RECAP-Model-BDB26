# Measuring the Riddle of a "Perfect Pass" through Route Timing Synchronization

**By Lindsay Fleishman, Liam Benderly, & Matthew Griffith**
University Track submission for the 2025 NFL Big Data Bowl (University of Georgia)

---

## Overview

In modern NFL offenses, milliseconds separate a highlight-reel touchdown from a drive-killing incompletion. We introduce the **Route Timing Synchronization (RTS) Score**, a player-level metric that quantifies the alignment between quarterback release timing and receiver route progression. Using positional tracking data, we transform what was once an intangible art form into a measurable science.

Inspired by duos like Patrick Mahomes and Travis Kelce, the RTS Score captures the chemistry and timing perfection that makes certain QB-receiver pairs unstoppable.

## Key Contributions

**RECAP Model (Release Expected Completion & Placement)**
- Transformer-based completion probability model trained on release frames
- Achieves **70% accuracy** and **74% AUC-ROC** on held-out test data
- Incorporates receiver trajectory features, route-type embeddings, and momentum encoders
- Enables counterfactual analysis for coaching applications

**RTS Score (0-100 scale)**
- **Temporal Component**: Measures timing precision by comparing actual release to optimal release window
- **Spatial Component**: Evaluates ball placement relative to optimal target location
- Route-specific grading scaled within each route type

## Model Architecture

Our model leverages the SportsTrackingTransformer architecture with key innovations:
- 4-layer Transformer encoder with 8 attention heads
- Route-type embeddings to capture route-specific patterns
- Receiver momentum encoder quantifying trajectory alignment
- Combined prediction head fusing player interactions, route context, and Next Gen Stats features

Training exclusively on release frames (~2,100 examples) optimizes for the specific decision point where quarterbacks commit to throw timing and placement.

## Repository Structure

```
├── code/
│   ├── recap_model_code.ipynb              # Main model training notebook
│   ├── completion_probability_visualizer.py # Completion probability visualization
│   ├── model_demonstration_visualizer.py    # Model demonstration tools
│   ├── qb_wr_chemistry_analysis.py          # QB-WR pair analysis
│   ├── rss_evaluation_pipeline.py           # RTS evaluation pipeline
│   ├── rts_postprocessing.py                # RTS score calculation
│   └── rts_validation_analysis.py           # Validation metrics
├── data/                                     # Raw and processed tracking data
├── results/                                  # Model outputs and RTS leaderboards
├── visualizations/                           # Play breakdowns and film analysis
└── recap_model.pt                            # Trained RECAP model weights
```


## Data Source

NFL Player Tracking Data (2023 season)

## Acknowledgments

We would like to thank the University of Georgia Football Team's coaches, players, and sports scientists for their insight on the value of RTS. Special thanks to Smit Bajaj, Cole Jacobson, and Sam Bruchhaus for their advice and inspiration on the technical and visual aspects of this project.

---
