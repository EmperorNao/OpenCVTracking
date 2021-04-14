import math


def get_vectors(contour, n_vectors, c_x, c_y, log=False):
    """
    Function to parameterize contour as n_vectors

    :param contour: OpenCV contour
    :param n_vectors: number of vector for parametrization
    :param c_x: x-center coordinate
    :param c_y: y-center coordinate
    :param log: print in console some logs
    :return: return a list of size n_vectors
    """

    vector_ends = [[] for i in range(0, n_vectors)]
    # bias for normal comparison
    degree = 1.5 / n_vectors

    # for each point
    for i in range(0, len(contour) - 1):

        alpha = 0
        f = True
        # iterate over all degrees while won't find which satisfied
        while (alpha < n_vectors) and f:

            phi = 2 * 3.14159265 * alpha / n_vectors + degree
            coef = math.tan(phi)

            if log:

                print("point1 =", (contour[i][0][1] - c_y)/(contour[i][0][0] - c_x))
                print("tan = ", coef)
                print("point2 = ", (contour[i + 1][0][1] - c_y)/(contour[i + 1][0][0] - c_x))
                print("\n")

            try:

                if (contour[i][0][0] == c_x) or (contour[i + 1][0][0] == c_x):

                    i += 1
                    f = False

                elif ((contour[i][0][1] - c_y)/(contour[i][0][0] - c_x) > coef) and \
                        ((contour[i + 1][0][1] - c_y)/(contour[i + 1][0][0] - c_x) < coef):

                    if contour[i][0][1] > c_y:
                        vector_ends[alpha].append(contour[i][0])

                    else:
                        vector_ends[alpha + int(n_vectors/2)].append(contour[i][0])

                    f = False

            except():
                pass

            alpha += 1

    ends = []
    # for all vectors find mean
    for i in range(0, len(vector_ends)):

        t = [0, 0]
        for j in range(0, len(vector_ends[i])):
            t += vector_ends[i][j]

        if len(vector_ends[i]):
            t = t / len(vector_ends[i])
            ends.append([int(t[0]), int(t[1])])

        else:
            ends.append([])

    # fill empty points as mean of closest in sequence
    for i in range(0, len(ends)):

        if not ends[i]:

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

            ends[i] = [int((ends[index_low][0] + ends[index_high][0])/2),
                       int((ends[index_low][1] + ends[index_high][1])/2)]

    return ends
