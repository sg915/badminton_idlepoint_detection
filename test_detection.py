#!/usr/bin/env python3
"""
Test script for Badminton Point Detection System
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_results(csv_file='point_detection_results.csv'):
    """Analyze the detection results"""
    df = pd.read_csv(csv_file)
    
    print("=== Badminton Point Detection Analysis ===")
    print(f"Total frames analyzed: {len(df)}")
    print(f"Video duration: {df['timestamp'].iloc[-1]:.1f} seconds")
    
    # State distribution
    state_counts = df['state'].value_counts()
    print(f"\nState Distribution:")
    for state, count in state_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {state.replace('_', ' ').title()}: {count} frames ({percentage:.1f}%)")
    
    # Point statistics
    points_df = df[df['point_number'] > 0]
    if not points_df.empty:
        unique_points = points_df['point_number'].nunique()
        print(f"\nPoint Statistics:")
        print(f"  Total points detected: {unique_points}")
        
        # Calculate point durations
        point_durations = []
        for point_num in points_df['point_number'].unique():
            point_frames = points_df[points_df['point_number'] == point_num]
            duration = len(point_frames)
            point_durations.append(duration)
        
        if point_durations:
            avg_duration = np.mean(point_durations)
            min_duration = np.min(point_durations)
            max_duration = np.max(point_durations)
            print(f"  Average point duration: {avg_duration:.1f} frames ({avg_duration/30:.1f} seconds)")
            print(f"  Shortest point: {min_duration} frames ({min_duration/30:.1f} seconds)")
            print(f"  Longest point: {max_duration} frames ({max_duration/30:.1f} seconds)")
    
    # Movement analysis
    print(f"\nMovement Analysis:")
    print(f"  Average movement: {df['movement'].mean():.4f}")
    print(f"  Max movement: {df['movement'].max():.4f}")
    print(f"  Movement std: {df['movement'].std():.4f}")
    
    # Create visualization
    create_visualization(df)

def create_visualization(df):
    """Create visualization of the results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Movement over time
    ax1.plot(df['timestamp'], df['movement'], 'b-', alpha=0.7, linewidth=0.8)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Movement')
    ax1.set_title('Movement Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Highlight point playing periods
    point_periods = df[df['state'] == 'point_playing']
    if not point_periods.empty:
        ax1.scatter(point_periods['timestamp'], point_periods['movement'], 
                   c='red', s=20, alpha=0.8, label='Point Playing')
        ax1.legend()
    
    # State over time
    state_numeric = df['state'].map({'idle': 0, 'point_playing': 1})
    ax2.fill_between(df['timestamp'], 0, state_numeric, 
                     where=(state_numeric == 1), color='green', alpha=0.7, 
                     label='Point Playing')
    ax2.fill_between(df['timestamp'], 0, state_numeric, 
                     where=(state_numeric == 0), color='red', alpha=0.7, 
                     label='Idle')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('State')
    ax2.set_title('Point Detection Over Time')
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('detection_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved as 'detection_analysis.png'")

if __name__ == "__main__":
    analyze_results()
