# Imports
import Common as common
import cv2 as opencv
from Features import FeatureMatching

class CameraCapture:
    # Preview 
    is_preview_active = False        

    # Reference rectangle selected by user
    user_rectangle = None

    # Tracking via feature matching
    feature_matching = FeatureMatching()

    def __init__(self):
        # Init camera
        self.camera_capture = opencv.VideoCapture(0)

        # Get fps and calculate delay
        fps = self.camera_capture.get(opencv.CAP_PROP_FPS)
        self.ms_delay = int(1000 / fps)

        # Initialize windows
        opencv.namedWindow(common.preview_window_name, opencv.WINDOW_AUTOSIZE)

        # Set mouse callback
        opencv.setMouseCallback(common.preview_window_name, self.onMouseMove)
        
    def startPreview(self):
        self.is_preview_active = True        
        
        while(self.is_preview_active):  
            self.captureAndDisplayFrame()

    def stopPreview(self):        
        self.is_preview_active = False
    
    def captureAndDisplayFrame(self):
        # Read frame from the camera
        (capture_status, self.current_camera_frame) = self.camera_capture.read()

        # Verify capture status
        if(capture_status):            
            # Check, whether reference frame was set. If so, track the template.
            # Otherwise draw user rectangle
            if self.feature_matching.hasTemplate():
                self.current_camera_frame = self.feature_matching.trackTemplate(
                    self.current_camera_frame)            
            else:
                self.drawUserRectangle()

            # Display the captured frame
            opencv.imshow(common.preview_window_name, self.current_camera_frame)
            
            # Check, whether user pressed 'q' key
            if(opencv.waitKey(self.ms_delay) == common.quit_key):
                self.stopPreview()
        else:
            # Print error to the console
            print(common.capture_failed)

    def onMouseMove(self, event, x, y, flags, user_data):
        # User pressed left mouse button and started drawing the rectangle
        if(event == opencv.EVENT_LBUTTONDOWN):    
            self.feature_matching.clearTemplate()
            self.mouse_start_pos = (x, y)

        # User is drawing the rectangle. 
        # We store coordinates of this rectangle in the user_rectangle field
        elif(flags & opencv.EVENT_FLAG_LBUTTON):
            if(self.mouse_start_pos is not None):
                min_pos = (min(self.mouse_start_pos[0], x), 
                           min(self.mouse_start_pos[1], y))

                max_pos = (max(self.mouse_start_pos[0], x), 
                           max(self.mouse_start_pos[1], y))

                self.user_rectangle = (min_pos[0], min_pos[1], 
                                       max_pos[0], max_pos[1])

        # User has finished drawing the rectangle
        elif(event == opencv.EVENT_LBUTTONUP):
            self.mouse_start_pos = None
            self.feature_matching.setTemplate(
                self.current_camera_frame, self.user_rectangle)

        # User resets the rectangle
        elif(event == opencv.EVENT_RBUTTONDOWN):
            self.feature_matching.clearTemplate()
            self.user_rectangle = None

    def drawUserRectangle(self):
        if(self.user_rectangle is not None):
            top_left_corner = (self.user_rectangle[0], 
                               self.user_rectangle[1])

            bottom_right_corner = (self.user_rectangle[2], 
                                   self.user_rectangle[3])

            opencv.rectangle(self.current_camera_frame, 
                             top_left_corner, bottom_right_corner, 
                             common.yellow, common.rectangle_thickness)

    def release(self):
        # Stop and release camera preview
        self.stopPreview()
        self.camera_capture.release()
        
        # Release windows
        opencv.destroyAllWindows()        
