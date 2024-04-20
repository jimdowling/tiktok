
import os
import joblib
import numpy as np

import logging

class Predict(object):
    
    def __init__(self):
        self.model = joblib.load(os.environ["ARTIFACT_FILES_PATH"] + "/ranking_model.pkl")

    def predict(self, inputs):
        # Extract ranking features and article IDs from the inputs
        features = inputs[0].pop("ranking_features")
        video_ids = inputs[0].pop("video_ids")
        
        # Log the extracted features
        logging.info("predict -> " + str(features))

        # Predict probabilities for the positive class
        scores = self.model.predict_proba(features).tolist()
        
        # Get scores of positive class
        scores = np.asarray(scores)[:,1].tolist() 

        # Return the predicted scores along with the corresponding article IDs
        return {
            "scores": scores, 
            "video_ids": video_ids,
        }
