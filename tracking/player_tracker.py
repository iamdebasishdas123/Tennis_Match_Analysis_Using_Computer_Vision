from ultralytics import YOLO
import cv2
import pickle 
import numpy as np
import os
import pandas as pd
import sys
sys.path.append('../')
from utils import get_center_of_bbox, mesure_distance_between_two_points

class player_trackers:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
    def choose_and_filter_players(self,court_key_points, player_detections):
        player_detection_frist_frame = player_detections[0]
        chosen_player_detections = self.choose_player_detections(court_key_points,player_detection_frist_frame)
        fliter_player_detections = []
        for player_detection in player_detections:
            fliter_player_dict={ track_id: bbox for track_id, bbox in player_detection.items() if track_id in chosen_player_detections}
            fliter_player_detections.append(fliter_player_dict)
        return fliter_player_detections
    
    def choose_player_detections(self,court_key_points,player_dict):
        distances=[]
        for track_id, bbox in player_dict.items():
            player_center=get_center_of_bbox(bbox)
            
            min_distance=float('inf')
            for i in range(0,len(court_key_points),2):
                court_key_point=(court_key_points[i],court_key_points[i+1])
                distance =mesure_distance_between_two_points(player_center,court_key_point)
                
                if distance < min_distance:
                    min_distance=distance
            distances.append((track_id,min_distance))
            
        # sort Distence 
        distances.sort(key=lambda x: x[1])
        # choose the first 2 player track 
        choose_player=[distances[0][0], distances[1][0]]
        
        return choose_player
        

        
    def detect_frames(self,frames,read_from_stub=False,stub_path=None):

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections


        player_detections=[]
        for frame in frames:
            player_dict=self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)


        return player_detections

            
        
        
    
    def detect_frame(self, frame):
        results=self.model.track(frame,persist=True)
        id_name_dict=self.model.names
        
        player_dict={}
        
        for box in results[0].boxes:
            track_id=int(box.id.tolist()[0])
            result=box.xyxy.tolist()[0]
            object_class_id=box.cls.tolist()[0]
            object_class_name=id_name_dict[object_class_id]
            if object_class_name == "person":
                player_dict[track_id]=result
                
        return player_dict

    def draw_bboxes(self,video_frames, player_detections):
            output_video_frames = []
            for frame, player_dict in zip(video_frames, player_detections):
                # Draw Bounding Boxes
                for track_id, bbox in player_dict.items():
                    x1, y1, x2, y2 = bbox
                    cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 0, 255), 2)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (120, 0, 255), 2)
                output_video_frames.append(frame)
            
            return output_video_frames
