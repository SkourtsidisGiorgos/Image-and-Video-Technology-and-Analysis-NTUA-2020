import cv2 as cv
import numpy as np
from skimage import util,filters
from skimage.morphology import disk

#-------------------------------------------------------------------------------#
#                Παραμετροποίηση των αλγορίθμων                                 #        
#-------------------------------------------------------------------------------#

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners = 300, qualityLevel = 0.05, minDistance = 4, blockSize = 7,
                      useHarrisDetector = False)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize = (30,30), maxLevel = 1, criteria = (cv.TERM_CRITERIA_EPS | 
                                                              cv.TERM_CRITERIA_COUNT, 6, 0.07))

#-------------------------------------------------------------------------------#
#                Ερώτημα 1ο: Download video και frame resize                    #        
#-------------------------------------------------------------------------------#                                                             

# We renamed the video to an easier name
cap = cv.VideoCapture("ss.mp4")

# Green
color = (0, 255, 0)

# Read 1st frame
ret, first_frame = cap.read()

# Salt & Pepper θόρυβος στο 1st frame (ucomment to apply)
#first_frame = util.random_noise(first_frame, mode='s&p')

#cv.imshow('- Show image in window',first_frame) 
#cv.waitKey(0) 
#cv.destroyAllWindows() # destroys the window showing image

print('Original Dimensions : ',first_frame.shape)

scale_percent = 50 # percent of original size
width = int(first_frame.shape[1] * scale_percent / 100)
height = int(first_frame.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
first_frame = cv.resize(first_frame, dim, interpolation = cv.INTER_AREA)
 
print('Resized Dimensions : ',first_frame.shape)

#-------------------------------------------------------------------------------#
#                Ερώτημα 2o: Eπιλογή χρωματικού χώρου                           #        
#-------------------------------------------------------------------------------#  
# Converts frame to grayscale because we only need the luminance channel for detecting edges
# - less computationally expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

#prev_gray = util.random_noise(prev_gray, mode='s&p')

#-------------------------------------------------------------------------------#
#                Ερώτημα 3o: Harris και Shi-Tomasi                              #        
#-------------------------------------------------------------------------------#  
# Finds the strongest corners in the first frame by Shi-Tomasi / Harris method -
# we will track the optical flow for these corners
prev = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)



# Creates an image filled with zero intensities with the same dimensions as the frame 
#- for later drawing purposes
mask = np.zeros_like(first_frame)

j = 0



#-------------------------------------------------------------------------------#
#                    Ερώτημα 4o:  Lucas-Kanade                                  #        
#-------------------------------------------------------------------------------#  
while(cap.isOpened()):
    
    j = j+1
    
    ret, frame = cap.read()
    
    scale_percent = 50 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    frame = cv.resize(frame, dim, interpolation = cv.INTER_AREA)
    
#-------------------------------------------------------------------------------#
#    Ερώτημα 5o:  Τροποποίηση κώδικα για την παρακολούθηση αντικειμένων         #        
#-------------------------------------------------------------------------------#  

    # Every 30 frames apply edges detection
    if j==30 :
                
        j=0

        # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
        prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
       
        
        # Finds the strongest corners in the  frame by Shi-Tomasi / Harris method - we will track the optical flow for these corners
       
        prev = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)



# Make the frames grayscale
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#-------------------------------------------------------------------------------#
#               Ερώτημα 6o:  Salt and Pepper noise                              #        
#-------------------------------------------------------------------------------#  
    # Uncomment to apply noise
    # frame = util.random_noise(frame, mode='s&p') 

#-------------------------------------------------------------------------------#
#               Ερώτημα 7o:  Denoising with Median Filter                       #        
#-------------------------------------------------------------------------------#  
    # Uncomment to apply denoising
    # frame = filters.rank.median(salt_and_pepper_noise, disk(3))

    # Calculates sparse optical flow by Lucas-Kanade method
    next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)

    # Selects good feature points for previous position
    good_old = prev[status == 1]

    # Selects good feature points for next position
    good_new = next[status == 1]

    # Draws the optical flow tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        # a, b = coordinates of new point
        a, b = new.ravel()

        # a, b = coordinates of old point
        c, d = old.ravel()

        # Draws line between new and old position with green color and 2 thickness
        mask = cv.line(mask, (a, b), (c, d), color, 2)

        # Draws filled circle (thickness of -1) at new position with green color and radius of 3
        frame = cv.circle(frame, (a, b), 3, color, -1)

    # Overlays the optical flow tracks on the original frame
    output = cv.add(frame, mask)

    # Updates previous frame
    prev_gray = gray.copy()

    # Updates previous good feature points
    prev = good_new.reshape(-1, 1, 2)

    # Opens a new window and displays the output frame
    cv.imshow("sparse optical flow", output)

    # Frames are read by intervals of 32 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(32) & 0xFF == ord('q'):
        break

# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()
