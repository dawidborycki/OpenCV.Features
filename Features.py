# Imports
import cv2 as opencv
import numpy as np
import math
import Common as common

class FeatureMatching:      
    # Fields
    template = None

    def __init__(self):
        self.feature_detector = opencv.AKAZE_create()       
        #self.feature_detector = opencv.ORB_create()

        self.feature_matcher = opencv.DescriptorMatcher_create(
            opencv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    def displayFeatures(self, window_caption, image, key_points):
        # Draw keypoints
        image_with_features = opencv.drawKeypoints(image, key_points, None, 
            flags = opencv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS | opencv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
       
        # Display image
        opencv.imshow(window_caption, image_with_features)        

    def findFeatures(self, image):		
        return self.feature_detector.detectAndCompute(image, None)

    def translateRotateAndScaleImage(self, image, translation, deg_rotation, scale):
        # Get image dimensions
        (rows, cols, channel_count) = image.shape

        # Set transformation center
        transformation_center = (cols / 2, rows / 2)        

        # Get rotation matrix
        transformation_matrix = opencv.getRotationMatrix2D(transformation_center, deg_rotation, scale)        

        # Add translation
        transformation_matrix[0, 2] += translation[0]  
        transformation_matrix[1, 2] += translation[1]

        # Transform an image
        return opencv.warpAffine(image, transformation_matrix, None)

    def findFeaturesInLenaImage(self, window_caption="Lena features",      
                                translation=(0,0), deg_rotation=0, scale=1):
        # Read image from file
        lena_image = opencv.imread(common.lena_image_path)

        # Transform an image
        lena_image_transformed = self.translateRotateAndScaleImage(lena_image, translation, deg_rotation, scale)

	    # Find features
        key_points, descriptor = self.findFeatures(lena_image_transformed)

	    # Display features
        self.displayFeatures(window_caption, lena_image_transformed, key_points)

    def matchLenaFeatures(self, translation=(0,0), deg_rotation=0, scale=1):
        # Read image from file
        lena_image = opencv.imread(common.lena_image_path)

        # Transform an image
        lena_image_transformed = self.translateRotateAndScaleImage(lena_image, translation, deg_rotation, scale)

        # Compute features
        lena_key_pts, lena_desc = self.findFeatures(lena_image)
        lena_transformed_key_pts, lena_transformed_desc = self.findFeatures(lena_image_transformed)

        # Match features
        matches = self.feature_matcher.match(lena_desc, lena_transformed_desc, None)

        # Filter matches
        matches = self.filterMatches(matches)

        # Display matches
        image_with_matches = opencv.drawMatches(lena_image, lena_key_pts, 
            lena_image_transformed, lena_transformed_key_pts, matches, None)
        opencv.imshow("Match result", image_with_matches)

        # Register transformed image
        self.registerImage(lena_image, lena_image_transformed, 
                            matches, lena_key_pts, lena_transformed_key_pts)

    def filterMatches(self, matches, top_matches_count=15):
        matches.sort(key = lambda x: x.distance, reverse = False)
        
        return matches[:top_matches_count]

    def registerImage(self, template_image, test_image, matches, 
                        template_image_key_pts, test_image_key_pts):
        
        # Get key points of matched descriptors
        template_pts = [template_image_key_pts[match.queryIdx].pt 
                        for match in matches]
        test_pts = [test_image_key_pts[match.trainIdx].pt 
                    for match in matches]

        # Find homography
        H = opencv.findHomography(np.array(test_pts), np.array(template_pts))
        H = H[0]
        
        # Calculate and pring angle of rotation
        self.printAngleOfRotation(H)

        # Correct test image
        registered_image = opencv.warpPerspective(test_image, H, None)

        # Display results
        opencv.imshow("Original image", template_image)
        opencv.imshow("Transformed image", test_image)
        opencv.imshow("Registered image", registered_image)
        opencv.imshow("Weighted sum", opencv.addWeighted(
            registered_image, 0.5, template_image, 0.5, 0.0))

    def printAngleOfRotation(self, H):
        # Matrix inversion
        H = np.linalg.inv(H)

        # Get elements at 00 and 01
        h01 = H[0][1]
        h00 = H[0][0]

        # Calculate angle of rotation and express it in degrees
        deg_rotation_angle = math.atan2(h01, h00) * 180 / math.pi        

        # Print angle of rotation to the console
        print("Detected angle of rotation: {:.2f}".format(deg_rotation_angle))

    ## Object tracking
    def setTemplate(self, current_camera_frame, user_rectangle):
        if(current_camera_frame is not None):
            self.template = current_camera_frame[user_rectangle[1]:user_rectangle[3], 
                user_rectangle[0]:user_rectangle[2]]
                    
            # Get and store template feature descriptors
            # They will be used later in the 'trackTamplate' method
            self.template_key_pts, self.template_desc = self.findFeatures(
                self.template)

            # Display the template with features
            self.displayFeatures(common.template_preview_window_name, 
                                 self.template, self.template_key_pts)
            
    def hasTemplate(self):
        return self.template is not None

    def clearTemplate(self):
        self.template = None    

    def trackTemplate(self, current_camera_frame):
        if(self.template is not None):
            # Perform feature matching
            H = self.performFeatureMatching(current_camera_frame)
                        
            # Display tracking result
            return self.drawTrackingResult(H, current_camera_frame)
        else:
            return None

    def performFeatureMatching(self, current_camera_frame):
        # Compute current frame feature descriptors
        current_frame_key_pts, current_frame_desc = self.findFeatures(current_camera_frame)

        # Display features
        self.displayFeatures("Features", current_camera_frame, current_frame_key_pts)

        # Match features
        matches = self.feature_matcher.match(self.template_desc, current_frame_desc, None)

        # Filter matches
        matches = self.filterMatches(matches)
                    
        # Find and return homography
        return self.findHomography(current_frame_key_pts, matches)

    def findHomography(self, current_frame_key_pts, matches):
        # Get key points of matched descriptors
        template_pts = [self.template_key_pts[match.queryIdx].pt for match in matches]
        current_frame_pts = [current_frame_key_pts[match.trainIdx].pt for match in matches]

        # Find and then return homography
        H = opencv.findHomography(np.array(current_frame_pts), np.array(template_pts))        
        return H[0]

    def drawTrackingResult(self, H, current_camera_frame):
        # Get template dimensions
        height, width, _ = self.template.shape
            
        # Prepare template rectangle
        template_rect = np.float32([[0, 0], [0, height], 
                                    [width, height], [width, 0]]);

        # Transform rectangle using homography        
        template_rect_transformed = opencv.perspectiveTransform(
            template_rect.reshape(-1,1,2), np.linalg.inv(H))        
        
        # Draw rectangle
        opencv.polylines(current_camera_frame, [np.int32(template_rect_transformed)], True, 
                        common.green, common.rectangle_thickness)     
                
        # Return frame with rectangle
        return current_camera_frame
