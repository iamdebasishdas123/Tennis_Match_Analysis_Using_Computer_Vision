from utils import (read_video,
                   save_video)
from tracking import player_trackers
from tracking import ball_tracker
from courtline  import CourtLineDetector
from mini_court import MiniCourt
import numpy as np
import torch
import cv2
import pandas as pd
from copy import deepcopy
import constants
from utils.utils_bbox import mesure_distance_between_two_points
from utils.conversion import convert_pixel_distance_to_meters
from utils.player_stats_draw import draw_player_stats
from utils.utils_bbox import get_center_of_bbox
from tracking import BounceDetector
import cv2


def draw_points_on_the_video(all_frames, frame_numbers, ball_position, color=(0, 0, 255)):
    frames = all_frames.copy()

    for frame_no in frame_numbers:
        # Draw the circle permanently on all subsequent frames after `frame_no`
        for i in range(frame_no, len(frames)):  # Start from `frame_no` and go to the end
            frame = frames[i]
            ball_x, ball_y = ball_position[frame_no]  # Use the position for the specified frame
            # Mini Court ball positions when bonce the  
            
            cv2.circle(frame, (int(ball_x), int(ball_y)), 10, color, -1)

    return frames

def draw_points_mini_courts(all_frames, frame_numbers, ball_positions, color=(0,0,0)):
    frames = all_frames.copy()
    ball_position_mini_court = [item[1] for item in ball_positions]
    k=0

    for frame_no in frame_numbers:
        
        # Draw the circle permanently on all subsequent frames after `frame_no`
        for i in range(frame_no, len(frames)):  # Start from `frame_no` and go to the end
            frame = frames[i]
            ball_x, ball_y = ball_position_mini_court[k]  # Use the position for the specified frame
            
            # Mini Court ball positions when bonce the  
            cv2.circle(frame, (int(ball_x), int(ball_y)), 10, color, -1)
        k=k+1

    return frames
    
   

def main():
    # Read the input video and get its frames
    input_video_path = 'input_videos/input_video1.mp4'
    video_frames = read_video(input_video_path)
    
    # Detect players in the video frames
    player_tracker=player_trackers(model_path=r"models\yolov8x.pt")
    ball_tracking=ball_tracker(model_path=r"models\Detect_tennis_ball.pt")
    
    
    player_detections=player_tracker.detect_frames(video_frames,read_from_stub=True,stub_path="tracker_stubs/player_tracking_stub1.pkl")
    ball_detector=ball_tracking.detect_frames(video_frames,read_from_stub=True,stub_path="tracker_stubs/ball_tracking_stub1.pkl")
    ball_detector=ball_tracking.interpolate_ball_positions(ball_detector)
    # print("ball_detector", ball_detector)

    
    # Court point detection
    courtline_model_path=r"models\keypoints_model.pt"
    courtline_detector=CourtLineDetector(courtline_model_path)
    courtline_keypoints=courtline_detector.predict(video_frames[0])
    
    # choose player 
    player_detections=player_tracker.choose_and_filter_players(courtline_keypoints, player_detections)
    
    # MiniCourt
    mini_court = MiniCourt(video_frames[0]) 
    
    #DETECT THE bALL SHORT 
    ball_shot_frames=ball_tracking.get_ball_shot_frames(ball_detector)
 
    
    filter_shot_frames = [ball_shot_frames[0]] + [ball_shot_frames[i] for i in range(1, len(ball_shot_frames)) if ball_shot_frames[i] - ball_shot_frames[i - 1] > 10]
    print("ball shots frame",filter_shot_frames)
    
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, 
                                                                                                          ball_detector,
                                                                                                          courtline_keypoints)
    
    # print("ball_position",len(ball_mini_court_detections))
    # print("player_position",len(player_mini_court_detections))
    # data=[item[1] for item in ball_mini_court_detections]
    # print("ball_unique_position",len(set(data)))
    
    
    player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
    } ]
    
    for ball_shot_ind in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind+1]
        ball_shot_time_in_seconds = (end_frame-start_frame)/24 # 24fps

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = mesure_distance_between_two_points(ball_mini_court_detections[start_frame][1],
                                                           ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters( distance_covered_by_ball_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           ) 

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * 3.6

        # player who the ball
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min( player_positions.keys(), key=lambda player_id: mesure_distance_between_two_points(player_positions[player_id],
                                                                                                 ball_mini_court_detections[start_frame][1]))

        # opponent player speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = mesure_distance_between_two_points(player_mini_court_detections[start_frame][opponent_player_id],
                                                                player_mini_court_detections[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters( distance_covered_by_opponent_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           ) 

        speed_of_opponent = distance_covered_by_opponent_meters/ball_shot_time_in_seconds * 3.6

        current_player_stats= deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_number_of_shots']

    # # Bounce Detectors
    bounce_detector = BounceDetector(path_model=r"models\ctb_regr_bounce.cbm")
    ball_track=[ get_center_of_bbox(x[1]) for x in ball_detector]
    
    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]
    
    bounces = bounce_detector.predict(x_ball, y_ball)
    bounce_frames=list(bounces)
    bounce_frames.sort()
    print(" actual bounces",bounce_frames)
    filtered_bounce_frames_ = [bounce_frames[0]] + [
    bounce_frames[i] for i in range(1, len(bounce_frames)) if bounce_frames[i] - bounce_frames[i - 1] > 10]
    # print("Bounces:", filtered_bounce_frames)
    filtered_bounce_frames = [
        bounce for bounce in filtered_bounce_frames_
        if all(abs(bounce - hit) > 5 for hit in filter_shot_frames)
    ]
 
    print("Bounce filter:", filtered_bounce_frames)
    

    
    
    
    # ball Postion on bounce Frame 
    ball_track_on_bounce_frames = [ball_detector[i] for i in filtered_bounce_frames]
    player_track_bounce_frame= [player_detections[i] for i in filtered_bounce_frames]
    
    palyer_position, ball_position =mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_track_bounce_frame,ball_track_on_bounce_frames,courtline_keypoints)
    # print("ball_position",ball_position)
    
    
    
    

    
    # Draw output
    
    #draw bounding boxes 
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracking.draw_bboxes(output_video_frames, ball_detector)
    output_video_frames = courtline_detector.draw_keypoints_on_video(output_video_frames, courtline_keypoints)
    
    # Draw Hit ball postion and Bounce Ball Position in video frame 
    output_video_frames = draw_points_on_the_video(output_video_frames, filter_shot_frames, ball_track, color=(0, 255, 0))
    output_video_frames = draw_points_on_the_video(output_video_frames, filtered_bounce_frames, ball_track, color=(255, 0, 0))
    output_video_frames = draw_points_mini_courts(output_video_frames,filtered_bounce_frames,ball_position)
    

    
    # Mini Court 
    
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections, color=(255,0,0)) 
    
    
    #Player Stats
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)   
    
    
    # draw number of frame in the output video
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0 , 255), 2)
    
 
    # Save the output video   
    save_video(output_video_frames, 'output_videos/output_video1.avi')
    
    
if __name__ == '__main__':
    main()