import cv2
import time
import numpy as np
from sklearn.cluster import DBSCAN
import time
import json
from SuperGluePretrainedNetwork.demo_superglue import SuperGlueInference
import threading
import os

# from sklearn.cluster import DBSCAN


class OverlapProcess():
    def __init__(self):
        self.dbscan = DBSCAN(eps=0.06, min_samples=2)


    def getPoints(self, image1, image2, is_draw=True):
        points = self.matching.getMatchPoint(image1, image2)
        final_points_1 = np.array([[kp[0]*image1.shape[1]/self.matching.resize[0],
                                  kp[1]*image1.shape[0]/self.matching.resize[1]] for kp in points[0]])
        final_points_2 = np.array([[kp[0]*image2.shape[1]/self.matching.resize[0],
                                  kp[1]*image2.shape[0]/self.matching.resize[1]] for kp in points[1]])
        if is_draw:
            print(final_points_1)
            print(image1.shape)
            for e in final_points_1:
                image1 = cv2.circle(image1, (int(e[0]), int(
                    e[1])), radius=3, color=(0, 0, 255), thickness=-1)
            for e in final_points_2:
                image2 = cv2.circle(image2, (int(e[0]), int(
                    e[1])), radius=3, color=(0, 0, 255), thickness=-1)
        return final_points_1, final_points_2

    def getGlobalPolygon(self, points, image, is_draw=True):
        x = np.array(self.convexHull(points))
        if is_draw:
            image = cv2.polylines(image, np.int32([x]), True, (255, 0, 0), 2)
        return x

    def Left_index(self, points):
        minn = 0
        for i in range(1, len(points)):
            if points[i][0] < points[minn][0]:
                minn = i
            elif points[i][0] == points[minn][0]:
                if points[i][1] > points[minn][1]:
                    minn = i
        return minn

    def orientation(self, p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - \
            (q[0] - p[0]) * (r[1] - q[1])

        if val == 0:
            return 0
        elif val > 0:
            return 1
        else:
            return 2

    def convexHull(self, points):
        n = len(points)
        # There must be at least 3 points
        if n < 3:
            return

        # Find the leftmost point
        l = self.Left_index(points)

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
                if(self.orientation(points[p],
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

    def draw(self, image):
        while(1):

            cv2.imshow('image', image)
            if cv2.waitKey(20) & 0xFF == 27:
                break

        cv2.destroyAllWindows()

convex_storage = {}
num_completed = 0

def findConvex(names, frames, overlap_process, id):
    global convex_storage
    global num_completed
    for i,e in enumerate(frames):
        cv2.imwrite(f"SuperGluePretrainedNetwork/assets/AI_Track_{id}/{names[i]}.png", e)
    
    superglue = SuperGlueInference(input=f"SuperGluePretrainedNetwork/assets/AI_Track_{id}/", output_dir=f"dump_demo_sequence_{id}",
                                    resize=[320, 240], match_threshold=0.045, force_cpu=False, keypoint_threshold=0.015)
    # cv2.imshow("frame_0", frames[0])
    match_points = superglue.process()
    total_match_point = []
    for e in match_points:
        total_match_point += e[0]
    frames[id] = cv2.resize(frames[id], tuple(superglue.resize))
    convex = overlap_process.getGlobalPolygon(total_match_point, frames[id])
    return convex

def write_result_to_file(result, out_folder, resize=[320, 240]):
    os.mkdir(out_folder)
    cam_files = []
    for i in range(len(result[0])):
        f = open(f"{out_folder}/CAM_{i + 1}.txt", "a")
        cam_files.append(f)
    
    for i, frame in enumerate(result):
        for j, file in enumerate(cam_files):
            #Normalize!!!!!!
            origin_shape= frame[j][2]
            print("Origin: ", origin_shape)
            temp = []
            flip_frame = np.flip(frame[j][0], axis=0)
            flip_frame = np.insert(flip_frame, 0, flip_frame[-1],0)
            print(flip_frame)
            for p in flip_frame.tolist():
                temp.append(p[0]*origin_shape[1]/resize[0])
                temp.append(p[1]*origin_shape[0]/resize[1])
            
            file.write(f"frame_{i + 1}.jpg, {temp}, {1/frame[j][1]}\n".replace("[","(").replace("]", ")"))      

    for f in cam_files:
        f.close()  

def solveProblem(videos, scence):
    overlap_process = OverlapProcess()
    video_readers = []
    for e in videos:
        video_readers.append(cv2.VideoCapture(e))

    result_txt = []
    is_break = False
    index = 0
    
    while True:
        frame_txt = []
        frames = []
        origin_shape = []
        for e in video_readers:
            ret, frame = e.read()
            if frame is None:
                # e.release()
                is_break = True
                break
            frames.append(frame)
            origin_shape.append(frame.shape)
            print("Frame shape", frame.shape)
        if is_break:
            break
        index += 1
        print("Num Frame: ", index)
        names_0 = ["a", "b", "c", "d"]
        names_1 = ["b", "a", "c", "d"]
        names_2 = ["c", "b", "a", "d"]
        names_3 = ["d", "b", "c", "a"]
        # thread_0 = threading.Thread(target=findConvex, args=(names_0, frames, overlap_process, 0))
        # thread_1 = threading.Thread(target=findConvex, args=(names_1, frames, overlap_process, 1))
        # thread_0.start()
        # thread_1.start()
        # if 2 < len(frames) :
        #     thread_2 = threading.Thread(target=findConvex, args=(names_2, frames, overlap_process, 2))
        #     thread_2.start()

        # elif len(frames) > 3:
        #     thread_3 = threading.Thread(target=findConvex, args=(names_3, frames, overlap_process, 3))
        #     thread_3.start()

        # while num_completed < len(frames):
        #     pass
        start = time.perf_counter()
        convex = findConvex(names_0, frames, overlap_process, 0)
        end = time.perf_counter()
        frame_txt.append([convex, end-start, origin_shape[0]])
        print("Time: ",end - start)
        
        start = time.perf_counter()
        convex = findConvex(names_1, frames, overlap_process, 1)
        end = time.perf_counter()
        frame_txt.append([convex, end-start, origin_shape[1]])
        print("Time: ",end - start)

        if len(frames)>2:
            start = time.perf_counter()
            convex = findConvex(names_2, frames, overlap_process, 2)
            end = time.perf_counter()
            frame_txt.append([convex, end-start, origin_shape[2]])
            print("Time: ",end - start)     
        if len(frames)>3:
            start = time.perf_counter()
            convex = findConvex(names_3, frames, overlap_process, 3)
            end = time.perf_counter()
            frame_txt.append([convex, end-start, origin_shape[3]])
            print("Time: ",end - start)
        
        # if len(frames) == 4:
        #     top_row = cv2.hconcat([frames[0],frames[1]])
        #     bottom_row = cv2.hconcat([frames[2],frames[3]])
        #     final_image = cv2.vconcat([top_row,bottom_row])
            
            
        # if len(frames) == 3:
        #     black_image = np.zeros((320,240,3),dtype=np.uint8)
        #     top_row = cv2.hconcat([frames[0],frames[1]])
        #     bottom_row = cv2.hconcat([frames[2],black_image])
        #     final_image = cv2.vconcat([top_row,bottom_row])
            
        # if len(frames) == 2:
        #     black_image = np.zeros((320,240,3),dtype=np.uint8)
        #     top_row = cv2.hconcat([frames[0],frames[1]])
        #     final_image = top_row

        # cv2.imshow('Images',final_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows(
        # if cv2.waitKey(0) == ord('q'):
            # break
        result_txt.append(frame_txt)
        
    print("AAA",result_txt)
    write_result_to_file(result_txt, f"{scence}")




if __name__ == '__main__':
    # videos = [r"C:\Users\nguye\Downloads\videos-20230416T033520Z-001\videos\scene_dynamic_cam_01\CAM_1.mp4",
    #           r"C:\Users\nguye\Downloads\videos-20230416T033520Z-001\videos\scene_dynamic_cam_01\CAM_2.mp4",
    #           r"C:\Users\nguye\Downloads\videos-20230416T033520Z-001\videos\scene_dynamic_cam_01\CAM_2.mp4",
    #           r"C:\Users\nguye\Downloads\videos-20230416T033520Z-001\videos\scene_dynamic_cam_01\CAM_2.mp4"]
    # solveProblem(videos)
    scences = os.listdir(r"C:\Users\nguye\Downloads\videos-20230416T033520Z-001\videos")
    for i,scence in enumerate(scences):
        try:
            if i > 9:
                break
            print(i, scence)
            videos = os.listdir("C:/Users/nguye/Downloads/videos-20230416T033520Z-001/videos/"+scence)
            videos = list(map(lambda x: "C:/Users/nguye/Downloads/videos-20230416T033520Z-001/videos/"+scence + "/" + x, videos))
            print(videos)
            solveProblem(videos, scence)
        except:
            pass