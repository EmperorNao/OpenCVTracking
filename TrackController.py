import cv2
import time
from MParametrization import get_vectors
from Tracker import CentroidTracker


THRESHOLD = 200

global points
points = []


def add_point(event, x, y, flags, param):
    """
    callback function that append coordinates for point to global array points

    :param event:
    :param x:
    :param y:
    :param flags:
    :param param:
    :return:
    """

    if event == cv2.EVENT_LBUTTONDBLCLK:

        points.append([x, y])


def choose_reference_points_to_frame(window_name, frame):

    cv2.setMouseCallback(window_name, add_point())

    while True:

        cv2.imshow(window_name, frame)
        if cv2.waitKey() == ord('q'):

            break

    cv2.setMouseCallback(window_name, lambda *args: None)
    return points


# this function was_depricates
'''
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

                vector_ends = get_vectors(contour, n_vectors, cX, cY)

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
'''



# TODO
#  1) remake function and make her ables to stop video and choose contours in time
#  2) remake callback without global variable

def track(path,
                                    show=True,
                                    track=True,
                                    log=False,
                                    blacked=False,
                                    vectors=False,
                                    n_vectors=6,
                                    sleep = 0,
                                    timing = False,
                                    track_log = True):

    tracker = CentroidTracker()
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

                vector_ends = get_vectors(contour, n_vectors, cX, cY)

                if timing:

                    print("time = ", time.perf_counter() - start_time)

                if log:

                    print("vector of points = ", vector_ends)

                if vectors:

                    for p in vector_ends:

                        # draw vectors from center
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

            if track_log:

                print("Number of trackable objects:", len(objects.items()))

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