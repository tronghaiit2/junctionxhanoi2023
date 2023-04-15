from SuperGluePretrainedNetwork.matching_tool import MatchingTool
import cv2
import time
import numpy as np

  
def Left_index(points):
      
    '''
    Finding the left most point
    '''
    minn = 0
    for i in range(1,len(points)):
        if points[i][0] < points[minn][0]:
            minn = i
        elif points[i][0] == points[minn][0]:
            if points[i][1] > points[minn][1]:
                minn = i
    return minn
  
def orientation(p, q, r):
    '''
    To find orientation of ordered triplet (p, q, r). 
    The function returns following values 
    0 --> p, q and r are collinear 
    1 --> Clockwise 
    2 --> Counterclockwise 
    '''
    val = (q[1] - p[1]) * (r[0] - q[0]) - \
          (q[0] - p[0]) * (r[1] - q[1])
  
    if val == 0:
        return 0
    elif val > 0:
        return 1
    else:
        return 2
def convexHull(points, n):
      
    # There must be at least 3 points 
    if n < 3:
        return
  
    # Find the leftmost point
    l = Left_index(points)
  
    hull = []
      
    '''
    Start from leftmost point, keep moving counterclockwise 
    until reach the start point again. This loop runs O(h) 
    times where h is number of points in result or output. 
    '''
    p = l
    q = 0
    while(True):
          
        # Add current point to result 
        hull.append(p)
  
        '''
        Search for a point 'q' such that orientation(p, q, 
        x) is counterclockwise for all points 'x'. The idea 
        is to keep track of last visited most counterclock- 
        wise point in q. If any point 'i' is more counterclock- 
        wise than q, then update q. 
        '''
        q = (p + 1) % n
  
        for i in range(n):
              
            # If i is more counterclockwise 
            # than current q, then update q 
            if(orientation(points[p], 
                           points[i], points[q]) == 2):
                q = i
  
        '''
        Now q is the most counterclockwise with respect to p 
        Set p as q for next iteration, so that q is added to 
        result 'hull' 
        '''
        p = q
  
        # While we don't come to first point
        if(p == l):
            break
    # Print Result 
    a = []
    for each in hull:
        b = []
        b.append(points[each][0])
        b.append(points[each][1])
        a.append(b)
    return a
def draw(image,pts):
    
    # Window name in which image is
    # displayed
    window_name = 'Image'
    
    # Polygon corner points coordinates
    
    
    # pts = pts.reshape((-1, 1, 2))
    
    isClosed = True
    
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 2
    
    # Using cv2.polylines() method
    # Draw a Blue polygon with
    # thickness of 1 px
    image = cv2.polylines(image, np.int32([pts]),
                        isClosed, color, thickness)
    
    # Displaying the image
    while(1):
        
        cv2.imshow('image', image)
        if cv2.waitKey(20) & 0xFF == 27:
            break
            
    cv2.destroyAllWindows()
if __name__ == '__main__':
    img1_path = r'test_img\src_2.png'
    img2_path = r'test_img\dest_2.png'
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    a = MatchingTool()
    start = time.time()
    
    points = a.getMatchPoint(img1,img2)
    # a.visualizeImages(img1,img2)
    after_resize_1 = np.array([[kp[0]*img1.shape[1]/a.resize[0], kp[1]*img1.shape[0]/a.resize[1]] for kp in points[0]])
    after_resize_2 = np.array([[kp[0]*img2.shape[1]/a.resize[0], kp[1]*img2.shape[0]/a.resize[1]] for kp in points[1]])
    x = np.array(convexHull(after_resize_1,len(after_resize_1)))
    # y = np.array(convexHull(after_resize_2,len(after_resize_2)))
    # kp2 = np.array([[kp[0]*croped_dest_img.shape[1]/self.matching.resize[0], kp[1]*croped_dest_img.shape[0]/self.matching.resize[1]] for kp in kp2])
    for p in after_resize_1:
        img1 = cv2.circle(img1, (int(p[0]),int(p[1])), radius=3, color=(0, 0, 255), thickness=-1)
    # for p in after_resize_2:
    #     img2 = cv2.circle(img2, (int(p[0]),int(p[1])), radius=3, color=(0, 0, 255), thickness=-1)
    print(x)
    draw(img1,x)
    # draw(img2,y)
 
    
    end = time.time()
    print(end - start)
    