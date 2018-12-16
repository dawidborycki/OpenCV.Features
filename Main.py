# Imports
import cv2 as opencv
from Camera import CameraCapture
from Features import FeatureMatching

# Helper functions
def runLenaImageSample():
    # Init
    feature_matching = FeatureMatching()

    # Find features in Lena image
    feature_matching.findFeaturesInLenaImage()
    
    # Find features in rotated and scaled Lena image
    translation = [100,-100]
    deg_rotation = -10
    scale = 1.25
    feature_matching.findFeaturesInLenaImage(
        "Lena features (transformed)", translation, deg_rotation, scale)

    # Match features
    feature_matching.matchLenaFeatures(
        translation, deg_rotation, scale)    

    # Wait for user to press the key.
    # Otherwise images will not be displayed
    opencv.waitKey()

def runCameraSample():
    camera_capture = CameraCapture()    
    camera_capture.startPreview()

    # Release resources
    camera_capture.release()

if __name__ == "__main__":
    use_camera = True

    if(use_camera):            
        runCameraSample()
    else:
        runLenaImageSample()
        
