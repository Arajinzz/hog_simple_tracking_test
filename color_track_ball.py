import numpy as np
import cv2
from hog_hist import HOG, normSSD
import os

# DATASET PATH
input_folder = 'Atelier/BallR/'

# RGB OR GRAY
isGray = True

# HYPERPARAMETRES
# initial Rectangle Position
x = 220
y = 130

# how many bins in the histogram
bins = 18
bin_size = 360 // bins

# rectangle size
sizey = 28
sizex = 28

# search area and stride
sArea = 20
st = 4

# READ FRAMES
frames = []
for img in os.listdir(input_folder):
    if(img == 'Thumbs.db'):
        continue
    
    if isGray:
        img = cv2.imread(input_folder + img, 0)
    else:
        img = cv2.imread(input_folder + img)
    
    frames.append(img)



def getBestMatch(img, searchArea, lasthist, x, y, strides):
    errors = []
    
    bL, eL = max(sizey, y-searchArea), min(img.shape[0]-sizey, y+searchArea)
    bC, eC = max(sizex, x-searchArea), min(img.shape[1]-sizex, x+searchArea)

    normalized_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    for i in range(bL, eL, strides):
        for j in range(bC, eC, strides):
            block = normalized_img[i-sizey:i+sizey, j-sizex:j+sizex]
            hist = HOG(block, bins, bin_size)
            errors.append(((j, i), normSSD(hist, lasthist), hist))

    best_error = min(errors, key=lambda s: s[1])

    return best_error
    

matches_memory = []

# first hog histogram
first_normalized_frame = cv2.normalize(frames[0], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
block = first_normalized_frame[y-sizey:y+sizey, x-sizex:x+sizex]
temp_hist = HOG(block, bins, bin_size)

for i in range(1, len(frames)):
    frame = frames[i]
    best_match = getBestMatch(frame, sArea, temp_hist, x, y, st)
    
    if best_match:
        x, y = best_match[0]
        matches_memory.append((best_match[1], best_match[2]))
    
    # UPDATE the temp_hist every 30 frames
    if i % 30 == 0:
        temp_hist = min(matches_memory, key=lambda s: s[0])[1]
        matches_memory = []
    
    cv2.rectangle(frame, (x-sizex, y-sizey), (x+sizex, y+sizey), (0, 0, 0), 1)
    cv2.imshow('Original', frame)

    if cv2.waitKey(25) == ord('q'):
        break


cv2.destroyAllWindows()

