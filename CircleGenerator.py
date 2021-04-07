import cv2
import numpy as np
import math
import time


THRESHOLD = 252

global points
points = []


def draw_circle(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDBLCLK:
        points.append([x, y])
        print("click")


def choose_reference_points_in_first_frame(path):

    cap = cv2.VideoCapture(path)

    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output", 900, 700)
    cv2.setMouseCallback("output", draw_circle)

    ret, frame = cap.read()

    while True:

        cv2.imshow("output", frame)
        if cv2.waitKey() == ord('q'):

            break

    cap.release()
    cv2.destroyAllWindows()

'''
def get_vectors(contour, k, cX, cY, log = False):

    vector_ends = []
    max_point = 0
    for i in range(0, len(contour)):
        if (len(contour[i]) > max_point):
            max_point = len(contour[i])
            t = contour[i]

    for i in range(0, len(t) - 1):

        alpha = 0
        f = True
        while (alpha < k) and f:

            phi = 2 * 3.14159265 * alpha / k + degree
            coef = math.tan(phi)

            if log:

                print("point1 =", (t[i][0][1] - cY)/(t[i][0][0] - cX))
                print("tan = " , coef)
                print("point2 = ", (t[i + 1][0][1] - cY)/(t[i + 1][0][0] - cX))
                print("\n")

            # divizion by zero
            if ((t[i][0][1] - cY)/(t[i][0][0] - cX) > coef) and ((t[i + 1][0][1] - cY)/(t[i + 1][0][0] - cX) < coef):


                vector_ends.append(t[i][0])
                f = False

            alpha += 1

    return vector_ends
'''


def get_vectors_2(contour, k, cX, cY, log = False):

    ends = []
    vector_ends = [[] for i in range(0, k)]

    '''
    max_point = 0
    for i in range(0, len(contour)):
        if (len(contour[i]) > max_point):
            max_point = len(contour[i])
            t = contour[i]
    '''
    t = contour

    degree = 1.5 / k
    for i in range(0, len(t) - 1):

        alpha = 0
        f = True
        while (alpha < k) and f:

            phi = 2 * 3.14159265 * alpha / k + degree
            coef = math.tan(phi)

            if log:

                print("point1 =", (t[i][0][1] - cY)/(t[i][0][0] - cX))
                print("tan = " , coef)
                print("point2 = ", (t[i + 1][0][1] - cY)/(t[i + 1][0][0] - cX))
                print("\n")

            # divizion by zero
            try:
                if (t[i][0][0] == cX) or (t[i + 1][0][0] == cX):

                    i += 1
                    f = False

                elif ((t[i][0][1] - cY)/(t[i][0][0] - cX) > coef) and ((t[i + 1][0][1] - cY)/(t[i + 1][0][0] - cX) < coef):

                    if (t[i][0][1] > cY):
                        vector_ends[alpha].append(t[i][0])

                    else:
                        vector_ends[alpha + int(k/2)].append(t[i][0])

                    f = False

            except:
                print("except")
                print(i)
                print(len(t))

            alpha += 1


    for i in range(0, len(vector_ends)):

        t = [0, 0]
        for j in range(0, len(vector_ends[i])):
            t += vector_ends[i][j]

        if (len(vector_ends[i]) != 0):
            t = t / len(vector_ends[i])

        if (len(vector_ends[i])):
            ends.append([int(t[0]), int(t[1])])
        else:
            ends.append([])

    for i in range(0, len(ends)):

        if ends[i] == []:

            index_high = i + 1
            index_high = index_high % (len(ends))
            while (index_high != i) and (ends[index_high] == []):

                index_high += 1
                index_high = index_high % (len(ends))

            index_low = i - 1
            index_low = index_low % (len(ends))
            while (index_low % (len(ends)) != i) and (ends[index_low] == []):
                index_low -= 1
                index_high = index_high % (len(ends))

            ends[i] = [int((ends[index_low][0] + ends[index_high][0])/2), int((ends[index_low][1] + ends[index_high][1])/2)]

    return ends


def show_with_contours(path, show=True, log=False, blacked=False, vectors=False,  n_vectors=6, sleep = 0, timing = True):

    cap = cv2.VideoCapture(path)
    ret = True

    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output", 900, 700)

    while ret:

        # read frame
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = hsvImage
        # convert image in binary
        _, thresh = cv2.threshold(thresh, THRESHOLD, 255, cv2.THRESH_BINARY_INV)


        # find and draw contour
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (255, 0, 0), 3, cv2.LINE_AA, hierarchy, 1)


        # find and draw center
        for contour in contours:

            if (len(contour) > n_vectors):

                M = cv2.moments(contour)
                cX = int(M["m10"] / (M["m00"] + 0.0001))
                cY = int(M["m01"] / (M["m00"] + 0.0001))
                cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)

                start_time = time.perf_counter()

                vector_ends = get_vectors_2(contour, n_vectors, cX, cY)

                if time:

                    print("time = ", time.perf_counter() - start_time)

                if log:

                    print("vector of points = ", vector_ends)

                if vectors:

                    """
                    for i in range(0, len(vector_ends)):
                        print(i, " : ", vector_ends[i])
                    """

                    for p in vector_ends:

                        # draw vectors from center
                        print(vector_ends)
                        cv2.line(frame, (cX, cY), (p[0], p[1]), (0, 128, 0), 3)

                    for i in range(0, len(vector_ends) - 1):

                        # draw m-param contour
                        cv2.line(frame, (vector_ends[i][0], vector_ends[i][1]), (vector_ends[i + 1][0], vector_ends[i + 1][1]), (0, 0, 255), 3)


                    l = len(vector_ends)
                    if (l > 0):
                        cv2.line(frame, (vector_ends[0][0], vector_ends[0][1]), (vector_ends[l - 1][0], vector_ends[l - 1][1]), (0, 0, 255), 3)

        if show:
            if blacked:

                cv2.imshow("output", thresh)

            else:

                cv2.imshow('output', frame)

            if cv2.waitKey(sleep) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker():
    def __init__(self):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()


    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]


    def update(self, inputCentroids):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(inputCentroids) == 0:
            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        # loop over the bounding box rectangles
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()
            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue
                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

                # compute both the row and column index we have NOT yet
                # examined
                unusedRows = set(range(0, D.shape[0])).difference(usedRows)
                unusedCols = set(range(0, D.shape[1])).difference(usedCols)

                # in the event that the number of object centroids is
                # equal or greater than the number of input centroids
                # we need to check and see if some of these objects have
                # potentially disappeared
                if D.shape[0] >= D.shape[1]:
                    # loop over the unused row indexes
                    for row in unusedRows:
                        # grab the object ID for the corresponding row
                        # index and increment the disappeared counter
                        objectID = objectIDs[row]

                        for col in unusedCols:
                            self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.objects


def show_with_contours_and_tracking(path, tracker, show=True, log=False, blacked=False, vectors=False, track = True,  n_vectors=6, sleep = 0, timing = True):

    cap = cv2.VideoCapture(path)
    ret = True

    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output", 900, 700)

    while ret:

        # read frame
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = hsvImage
        # convert image in binary
        _, thresh = cv2.threshold(thresh, THRESHOLD, 255, cv2.THRESH_BINARY_INV)


        # find and draw contour
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (255, 0, 0), 3, cv2.LINE_AA, hierarchy, 1)

        centroids = []
        for contour in contours:

            if (len(contour) > n_vectors):

                M = cv2.moments(contour)
                cX = int(M["m10"] / (M["m00"] + 0.0001))
                cY = int(M["m01"] / (M["m00"] + 0.0001))
                centroids.append([cX, cY])

        objects = tracker.update(centroids)

        # find and draw center
        for contour in contours:

            if (len(contour) > n_vectors):

                M = cv2.moments(contour)
                cX = int(M["m10"] / (M["m00"] + 0.0001))
                cY = int(M["m01"] / (M["m00"] + 0.0001))
                cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)

                start_time = time.perf_counter()

                vector_ends = get_vectors_2(contour, n_vectors, cX, cY)

                if time:

                    print("time = ", time.perf_counter() - start_time)

                if log:

                    print("vector of points = ", vector_ends)

                if vectors:

                    """
                    for i in range(0, len(vector_ends)):
                        print(i, " : ", vector_ends[i])
                    """

                    for p in vector_ends:

                        # draw vectors from center
                        print(vector_ends)
                        cv2.line(frame, (cX, cY), (p[0], p[1]), (0, 128, 0), 3)

                    for i in range(0, len(vector_ends) - 1):

                        # draw m-param contour
                        cv2.line(frame, (vector_ends[i][0], vector_ends[i][1]), (vector_ends[i + 1][0], vector_ends[i + 1][1]), (0, 0, 255), 3)


                    l = len(vector_ends)
                    if (l > 0):
                        cv2.line(frame, (vector_ends[0][0], vector_ends[0][1]), (vector_ends[l - 1][0], vector_ends[l - 1][1]), (0, 0, 255), 3)

        if track:
            for (objectID, centroid) in objects.items():
                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.circle(frame, (centroid[0], centroid[1]), 5, (0, 255, 0), -1)


        if show:
            if blacked:

                cv2.imshow("output", thresh)

            else:

                cv2.imshow('output', frame)

            waitkey = cv2.waitKey(sleep)
            if waitkey == ord('q'):
                break
            elif waitkey == ord('s'):
                sleep = 0
            elif waitkey == ord('g'):
                sleep = 1

    cap.release()
    cv2.destroyAllWindows()


path = r'C:\Users\user\Downloads\Telegram Desktop\TwoObjectsTest_.mp4'


tracker = CentroidTracker()
# choose_reference_points_in_first_frame(path)
# show_with_contours(path, vectors=True, log=False, n_vectors=24, sleep=1, show=True)
# show_with_contours_and_tracking(path, tracker, n_vectors=24, sleep=0, track= False, show=True)