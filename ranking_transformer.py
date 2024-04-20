
import os
import pandas as pd

import hopsworks
from opensearchpy import OpenSearch

import logging


class Transformer(object):
    
    def __init__(self):
        # Connect to Hopsworks
        project = hopsworks.connection().get_project()
        self.fs = project.get_feature_store()
        
        # Retrieve the 'videos' feature view
        self.videos_fv = self.fs.get_feature_view(
            name="videos", 
            version=1,
        )
        
        # Get list of feature names for videos
        self.video_features = [feat.name for feat in self.videos_fv.schema]
        
        # Retrieve the 'users' feature view
        self.users_fv = self.fs.get_feature_view(
            name="users", 
            version=1,
        )

        # Retrieve the 'candidate_embeddings' feature view
        self.candidate_index = self.fs.get_feature_view(
            name="candidate_embeddings", 
            version=1,
        )

        # Retrieve ranking model
        mr = project.get_model_registry()
        model = mr.get_model(
            name="ranking_model", 
            version=1,
        )
        
        # Extract input schema from the model
        input_schema = model.model_schema["input_schema"]["columnar_schema"]
        
        # Get the names of features expected by the ranking model
        self.ranking_model_feature_names = [feat["name"] for feat in input_schema]
            
    def preprocess(self, inputs):
        # Extract the input instance
        inputs = inputs["instances"][0]

        # Extract customer_id from inputs
        user_id = inputs["user_id"]
        
        # Search for candidate items
        neighbors = self.candidate_index.find_neighbors(
            inputs["query_emb"], 
            k=100,
        )
        neighbors = [neighbor[0] for neighbor in neighbors]
        
        # Get IDs of items already bought by the customer
        already_seen_videos_ids = self.fs.sql(
            f"SELECT video_id from interactions_1 WHERE user_id = '{user_id}'"
        ).values.reshape(-1).tolist()
        
        # Filter candidate items to exclude those already bought by the customer
        video_id_list = [
            video_id
            for video_id 
            in neighbors 
            if video_id
            not in already_seen_videos_ids
        ]
        video_id_df = pd.DataFrame({"video_id" : video_id_list})
        
        # Retrieve Article data for candidate items
        videos_data = [
            self.videos_fv.get_feature_vector({"video_id": video_id}) 
            for video_id 
            in video_id_list
        ]

        videos_df = pd.DataFrame(
            data=videos_data, 
            columns=self.video_features,
        )
        
        # Join candidate items with their features
        ranking_model_inputs = video_id_df.merge(
            videos_df, 
            on="video_id", 
            how="inner",
        )        
        
        # Add customer features
        user_features = self.users_fv.get_feature_vector(
            {"user_id": user_id}, 
            return_type="pandas",
        )
        
        ranking_model_inputs["user_id"] = user_features['age'].values[0]   
        ranking_model_inputs["gender"] = user_features["gender"].values[0] 
        ranking_model_inputs["age"] = user_features["age"].values[0] 
        ranking_model_inputs["country"] = user_features["country"].values[0] 
        
        # Select only the features required by the ranking model
        ranking_model_inputs = ranking_model_inputs[self.ranking_model_feature_names]
                
        return { 
            "inputs" : [{"ranking_features": ranking_model_inputs.values.tolist(), "video_ids": video_id_list}]
        }

    def postprocess(self, outputs):
        # Extract predictions from the outputs
        preds = outputs["predictions"]
        
        # Merge prediction scores and corresponding article IDs into a list of tuples
        ranking = list(zip(preds["scores"], preds["video_ids"]))
        
        # Sort the ranking list by score in descending order
        ranking.sort(reverse=True)
        
        # Return the sorted ranking list
        return { 
            "ranking": ranking,
        }
