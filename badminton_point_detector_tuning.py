#!/usr/bin/env python3
"""
Badminton Point Detection System - Tuning Version
Detects when a point is being played vs when players are idling
with enhanced temporal smoothing and additional tuning parameters
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import argparse
import os
from collections import deque
import time

class TuningBadmintonPointDetector:
    def __init__(self, movement_threshold=0.005, min_point_duration=30, 
                 smoothing_window=10, idle_grace_period=15, point_grace_period=10,
                 movement_boost_factor=1.0, context_window=5, debug=False):
        """
        Initialize the tuning point detector
        
        Args:
            movement_threshold: Minimum movement to consider as active play
            min_point_duration: Minimum frames to consider a valid point
            smoothing_window: Number of frames for movement smoothing
            idle_grace_period: Frames to wait before switching to idle
            point_grace_period: Frames to wait before switching to point
            movement_boost_factor: Multiplier for movement sensitivity (1.0 = normal)
            context_window: Number of recent frames to consider for context
            debug: Enable debug output
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Detection parameters
        self.movement_threshold = movement_threshold
        self.min_point_duration = min_point_duration
        self.smoothing_window = smoothing_window
        self.idle_grace_period = idle_grace_period
        self.point_grace_period = point_grace_period
        self.movement_boost_factor = movement_boost_factor
        self.context_window = context_window
        self.debug = debug
        
        # State tracking
        self.previous_landmarks = None
        self.movement_history = deque(maxlen=smoothing_window)
        self.current_state = 'idle'
        self.point_start_frame = None
        self.point_count = 0
        
        # Enhanced state tracking for smoothing
        self.state_transition_counter = 0
        self.pending_state = None
        self.last_movement_values = deque(maxlen=context_window)
        self.state_history = deque(maxlen=10)  # Track recent states
        
        # Debug tracking
        self.debug_info = []
        
    def calculate_movement(self, current_landmarks, previous_landmarks):
        """Calculate movement between current and previous pose landmarks"""
        if previous_landmarks is None or current_landmarks is None:
            return 0.0
        
        try:
            total_movement = 0.0
            valid_points = 0
            
            # Calculate movement for key body points
            key_points = [
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_ELBOW,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                self.mp_pose.PoseLandmark.LEFT_WRIST,
                self.mp_pose.PoseLandmark.RIGHT_WRIST,
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_HIP,
                self.mp_pose.PoseLandmark.LEFT_KNEE,
                self.mp_pose.PoseLandmark.RIGHT_KNEE,
                self.mp_pose.PoseLandmark.LEFT_ANKLE,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE
            ]
            
            for point in key_points:
                try:
                    curr = current_landmarks[point]
                    prev = previous_landmarks[point]
                    
                    # Calculate Euclidean distance
                    distance = np.sqrt((curr.x - prev.x)**2 + (curr.y - prev.y)**2)
                    total_movement += distance
                    valid_points += 1
                except:
                    continue
            
            # Apply movement boost factor
            movement = (total_movement / max(valid_points, 1)) * self.movement_boost_factor
            return movement
            
        except Exception as e:
            return 0.0
    
    def detect_point_state_with_smoothing(self, movement):
        """Detect if a point is being played with enhanced temporal smoothing"""
        # Add movement to history
        self.movement_history.append(movement)
        self.last_movement_values.append(movement)
        
        # Calculate average movement over smoothing window
        avg_movement = np.mean(list(self.movement_history))
        
        # Determine desired state based on movement
        if avg_movement > self.movement_threshold:
            desired_state = 'point_playing'
        else:
            desired_state = 'idle'
        
        # Apply enhanced smoothing logic
        return self.apply_enhanced_smoothing(desired_state, avg_movement)
    
    def apply_enhanced_smoothing(self, desired_state, avg_movement):
        """Apply enhanced smoothing to prevent false state transitions"""
        
        # If we're already in the desired state, reset counters
        if desired_state == self.current_state:
            self.state_transition_counter = 0
            self.pending_state = None
            return self.current_state
        
        # If this is a new desired state, start counting
        if self.pending_state != desired_state:
            self.pending_state = desired_state
            self.state_transition_counter = 1
        else:
            self.state_transition_counter += 1
        
        # Determine grace period based on current state and movement context
        grace_period = self.calculate_grace_period(desired_state, avg_movement)
        
        # Check if we've waited long enough
        if self.state_transition_counter >= grace_period:
            # Confirm the state change
            old_state = self.current_state
            self.current_state = self.pending_state
            self.state_transition_counter = 0
            self.pending_state = None
            
            # Handle state transition for point counting
            self.handle_state_transition(old_state, self.current_state)
            
            if self.debug:
                print(f"State change: {old_state} -> {self.current_state} (grace: {grace_period})")
            
            return self.current_state
        else:
            # Keep current state, but return it with smoothing info
            if self.debug and self.state_transition_counter % 5 == 0:
                print(f"Pending {self.pending_state} ({self.state_transition_counter}/{grace_period})")
            return self.current_state
    
    def calculate_grace_period(self, desired_state, avg_movement):
        """Calculate dynamic grace period based on context"""
        if self.current_state == 'point_playing' and desired_state == 'idle':
            # Switching from point to idle - use idle grace period
            grace_period = self.idle_grace_period
            
            # Additional logic: if movement is very low but we're in a point,
            # and recent movement was high, extend grace period (brief pause in point)
            if avg_movement < self.movement_threshold * 0.5:
                recent_high_movement = any(m > self.movement_threshold * 1.5 for m in list(self.last_movement_values)[-3:])
                if recent_high_movement:
                    grace_period = min(grace_period * 2, 30)  # Extend grace period
                    
        elif self.current_state == 'idle' and desired_state == 'point_playing':
            # Switching from idle to point - use point grace period
            grace_period = self.point_grace_period
            
            # Additional logic: if movement is high but we just ended a point,
            # require more confirmation (walking between points)
            if self.point_count > 0 and self.point_start_frame is None:
                recent_low_movement = any(m < self.movement_threshold * 0.8 for m in list(self.last_movement_values)[-3:])
                if recent_low_movement:
                    grace_period = min(grace_period * 1.5, 20)  # Require more confirmation
                    
            # Additional check: if we've been idle for a long time, be more sensitive
            if len(self.state_history) > 0:
                recent_idle_frames = sum(1 for state in list(self.state_history)[-5:] if state == 'idle')
                if recent_idle_frames >= 4:  # Been idle for most of recent frames
                    grace_period = max(grace_period * 0.7, 5)  # Be more sensitive
        else:
            grace_period = 5  # Default short grace period
        
        return grace_period
    
    def handle_state_transition(self, old_state, new_state):
        """Handle state transitions for point counting"""
        if new_state == 'point_playing' and old_state == 'idle':
            # Point started
            self.point_count += 1
            self.point_start_frame = time.time()  # Track start time
            print(f"Point #{self.point_count} started")
            
        elif new_state == 'idle' and old_state == 'point_playing':
            # Point ended
            if self.point_start_frame:
                duration = time.time() - self.point_start_frame
                print(f"Point #{self.point_count} ended (duration: {duration:.1f}s)")
                self.point_start_frame = None
    
    def process_video(self, video_path, output_path=None, results_path='tuning_point_detection_results.csv'):
        """Process video and detect points with enhanced smoothing"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing: {width}x{height} @ {fps}fps, {total_frames} frames")
        print(f"Movement threshold: {self.movement_threshold}")
        print(f"Movement boost factor: {self.movement_boost_factor}")
        print(f"Idle grace period: {self.idle_grace_period} frames")
        print(f"Point grace period: {self.point_grace_period} frames")
        print(f"Debug mode: {self.debug}")
        
        # Setup video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Results storage
        results_data = []
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            movement = 0.0
            if results.pose_landmarks:
                # Calculate movement from previous frame
                movement = self.calculate_movement(
                    results.pose_landmarks.landmark, 
                    self.previous_landmarks
                )
                
                # Update previous landmarks
                self.previous_landmarks = results.pose_landmarks.landmark
                
                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Detect point state with enhanced smoothing
            detected_state = self.detect_point_state_with_smoothing(movement)
            
            # Update state history
            self.state_history.append(detected_state)
            
            # Create visualization
            self.draw_visualization(frame, detected_state, movement, frame_count)
            
            # Store results
            results_data.append({
                'frame': frame_count,
                'timestamp': frame_count / fps,
                'movement': movement,
                'state': detected_state,
                'point_number': self.point_count if detected_state == 'point_playing' else 0,
                'transition_counter': self.state_transition_counter,
                'pending_state': self.pending_state,
                'avg_movement': np.mean(list(self.movement_history)) if self.movement_history else 0
            })
            
            # Write frame to output video
            if out:
                out.write(frame)
            
            frame_count += 1
            
            # Progress update
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed
                print(f"Processed {frame_count}/{total_frames} frames ({fps_actual:.1f} fps)")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        
        # Save results
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(results_path, index=False)
        
        print(f"\nProcessing complete!")
        print(f"Total points detected: {self.point_count}")
        print(f"Results saved to: {results_path}")
        if output_path:
            print(f"Output video saved to: {output_path}")
        
        return results_df
    
    def draw_visualization(self, frame, state, movement, frame_number):
        """Draw enhanced visualization on the frame"""
        height, width = frame.shape[:2]
        
        # State-specific colors
        if state == 'point_playing':
            color = (0, 255, 0)  # Green
            text = "POINT PLAYING"
            bg_color = (0, 200, 0)
        else:
            color = (0, 0, 255)  # Red
            text = "IDLE"
            bg_color = (0, 0, 200)
        
        # Draw background rectangle
        cv2.rectangle(frame, (10, 10), (400, 120), bg_color, -1)
        cv2.rectangle(frame, (10, 10), (400, 120), (255, 255, 255), 2)
        
        # Draw state text
        cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw movement value
        cv2.putText(frame, f"Movement: {movement:.4f}", (20, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw frame info
        cv2.putText(frame, f"Frame: {frame_number}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw smoothing info
        if self.pending_state:
            pending_text = f"Pending: {self.pending_state} ({self.state_transition_counter})"
            cv2.putText(frame, pending_text, (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Draw boost factor
        cv2.putText(frame, f"Boost: {self.movement_boost_factor:.1f}", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw point counter
        if self.point_count > 0:
            cv2.putText(frame, f"Points: {self.point_count}", (width - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def main():
    parser = argparse.ArgumentParser(description='Tuning Badminton Point Detection System')
    parser.add_argument('--input', '-i', required=True, help='Input video file')
    parser.add_argument('--output', '-o', help='Output video file')
    parser.add_argument('--results', '-r', default='tuning_point_detection_results.csv', 
                       help='Results CSV file')
    parser.add_argument('--threshold', '-t', type=float, default=0.005, 
                       help='Movement threshold for point detection')
    parser.add_argument('--min-duration', '-d', type=int, default=30, 
                       help='Minimum point duration in frames')
    parser.add_argument('--smoothing', '-s', type=int, default=10, 
                       help='Smoothing window size')
    parser.add_argument('--idle-grace', type=int, default=15, 
                       help='Frames to wait before switching to idle')
    parser.add_argument('--point-grace', type=int, default=10, 
                       help='Frames to wait before switching to point')
    parser.add_argument('--boost', type=float, default=1.0, 
                       help='Movement boost factor (1.0 = normal)')
    parser.add_argument('--context', type=int, default=5, 
                       help='Context window size for recent movement analysis')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug output')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        return
    
    # Create tuning detector
    detector = TuningBadmintonPointDetector(
        movement_threshold=args.threshold,
        min_point_duration=args.min_duration,
        smoothing_window=args.smoothing,
        idle_grace_period=args.idle_grace,
        point_grace_period=args.point_grace,
        movement_boost_factor=args.boost,
        context_window=args.context,
        debug=args.debug
    )
    
    print("Starting tuning badminton point detection...")
    print(f"Input: {args.input}")
    if args.output:
        print(f"Output: {args.output}")
    print(f"Results: {args.results}")
    
    # Process video
    results_df = detector.process_video(args.input, args.output, args.results)
    
    if results_df is not None:
        # Print summary
        state_counts = results_df['state'].value_counts()
        print("\nDetection Summary:")
        for state, count in state_counts.items():
            percentage = (count / len(results_df)) * 100
            print(f"{state.replace('_', ' ').title()}: {count} frames ({percentage:.1f}%)")
        
        # Print point statistics
        points_df = results_df[results_df['point_number'] > 0]
        if not points_df.empty:
            unique_points = points_df['point_number'].nunique()
            print(f"\nTotal unique points detected: {unique_points}")
            
            # Calculate average point duration
            point_durations = []
            for point_num in points_df['point_number'].unique():
                point_frames = points_df[points_df['point_number'] == point_num]
                duration = len(point_frames)
                point_durations.append(duration)
            
            if point_durations:
                avg_duration = np.mean(point_durations)
                print(f"Average point duration: {avg_duration:.1f} frames ({avg_duration/30:.1f} seconds at 30fps)")

if __name__ == "__main__":
    main()
