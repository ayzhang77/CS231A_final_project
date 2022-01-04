"""
CS231A Final Project: Go Board Reconstruction from a Single Image
By Amy Zhang

Takes as input a single image of a Go board and simultaneously reconstructs the game board and stone placement.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans


"""
Given two lines, l1 and l2, defined by 4 points, returns the intersection of the two lines.
"""
def get_intersection(l1, l2):
    x1 = l1[0]
    y1 = l1[1]
    x2 = l1[2]
    y2 = l1[3]

    x3 = l2[0]
    y3 = l2[1]
    x4 = l2[2]
    y4 = l2[3]

    det = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)

    xnum = (x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)
    ynum = (x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)

    return int(xnum / det), int(ynum / det)


"""
Performs perspective transformation on input image given 4 selected corner points. CLICK IN ORDER
TOP LEFT, TOP RIGHT, BOTTOM LEFT, BOTTOM RIGHT
"""
def perspective_transformation(a, b, img1):
    pts1 = np.float32([[a[0], b[0]], [a[1], b[1]], [a[2], b[2]], [a[3], b[3]]])
    pts2 = np.float32([[0, 0], [1000, 0], [0, 1000], [1000, 1000]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img1 = cv2.warpPerspective(img1, matrix, (1000, 1000))
    return img1


"""
Converts img1 to grayscale and performs thresholding and blurring. Uses Laplacian edge detector
and the fast line detector to get the main lines in the processed img. Uses detected lines to 
get corners of grid.
"""
def get_board_lines(img1, final_grid_img):
    # convert to grayscale, FOR GRID
    gray_temp = img1.copy()
    gray_img = cv2.cvtColor(gray_temp, cv2.COLOR_BGR2GRAY)

    # threshold image
    # anything above threshold becomes white, anything below threshold becomes black
    threshold = 90
    gray_thresh_img = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)[1]

    # remove Gaussian noise
    blur_param = 11
    blurred_img = cv2.GaussianBlur(gray_thresh_img, (blur_param, blur_param), 0)

    # use Laplacian edge detector (second order using Sobel derivatives)
    laplacian_16s_1 = cv2.Laplacian(blurred_img, cv2.CV_16S, ksize=1)
    abs_laplacian16s_1 = np.absolute(laplacian_16s_1)
    edges = np.uint8(abs_laplacian16s_1)

    cv2.imwrite("edges5.png", edges)

    # fast line detector from cv2 library
    fld = cv2.ximgproc.createFastLineDetector(_length_threshold=10, _distance_threshold=1.414213562, _canny_th1=25.0,
                                              _canny_th2=50.0, _canny_aperture_size=3, _do_merge=True)
    # (x1, y1, x2, y2), point 1 = start, point 2 = end, lines are directed so that brighter side is on their left
    lines = fld.detect(edges)
    blank_image = np.zeros((img1.shape[0], img1.shape[1], 3), np.uint8)
    detected_line_segmenets_img = fld.drawSegments(blank_image, lines)
    lines = lines[:, 0]
    cv2.imwrite("lines5.png", detected_line_segmenets_img)

    # get corners of grid
    top_left = tuple(min(lines[:, 0:2], key=sum))
    bottom_right = tuple(max(lines[:, 0:2], key=sum))
    top_right = tuple([bottom_right[0], top_left[1]])
    bottom_left = tuple([top_left[0], bottom_right[1]])

    # draw corners on final grid image
    cv2.circle(final_grid_img, top_left, radius=10, color=(0, 0, 255), thickness=4)
    cv2.circle(final_grid_img, bottom_right, radius=10, color=(0, 0, 255), thickness=4)
    cv2.circle(final_grid_img, top_right, radius=10, color=(0, 0, 255), thickness=4)
    cv2.circle(final_grid_img, bottom_left, radius=10, color=(0, 0, 255), thickness=4)

    return final_grid_img, lines, top_left, bottom_right, top_right, bottom_left


"""
From detected lines, extracts all horizontal lines within set threshold and performs kMean
clustering to cluster horizontal lines in 19 main clusters which serve as the 19 horizontal 
grid lines of the board.
"""
def get_horizontal_grid_lines(lines, stones_img, final, top_left, bottom_right, top_right, bottom_left):
    # manually extract horizontal lines out of detected lines
    h_thresh_dy = 10
    h_thresh_len = 22
    horiz_lines = [lines[i] for i in range(len(lines)) if
                   lines[i][3] - h_thresh_dy <= lines[i][1] <= lines[i][3] + h_thresh_dy
                   and lines[i][0] - lines[i][2] > h_thresh_len]
    horiz_lines = np.reshape(horiz_lines, (len(horiz_lines), 4))

    # cluster horizontal lines based off of p1 y value
    h_kmeans = KMeans(n_clusters=19, random_state=None).fit(horiz_lines[:, 1].reshape(-1, 1))

    # plot clustered horizontal grid lines
    h_grid_lines = []
    for center in h_kmeans.cluster_centers_:
        cv2.line(stones_img, (top_left[0], int(center)), (top_right[0], int(center)), (0, 255, 0), 4)
        h_grid_lines.append([top_left[0], int(center), top_right[0], int(center)])
        cv2.line(final, (top_left[0], int(center)), (top_right[0], int(center)), (0, 255, 0), 4)
    h_grid_lines = np.reshape(h_grid_lines, (len(h_grid_lines), 4))
    h_grid_lines = h_grid_lines[h_grid_lines[:, 1].argsort()]

    return h_grid_lines, stones_img, final


"""
From detected lines, extracts all vertical lines within set threshold and performs kMean
clustering to cluster vertical lines in 19 main clusters which serve as the 19 vertical 
grid lines of the board.
"""
def get_vertical_grid_lines(lines, stones_img, final, top_left, bottom_right, top_right, bottom_left):
    # manually extract vertical lines out of detected lines
    v_thresh_dx = 10
    v_thresh_len = 22
    vert_lines = [lines[i] for i in range(len(lines)) if
                  lines[i][2] - v_thresh_dx <= lines[i][0] <= lines[i][2] + v_thresh_dx
                  and lines[i][1] - lines[i][3] > v_thresh_len]
    vert_lines = np.reshape(vert_lines, (len(vert_lines), 4))

    # cluster vertical lines based off of p1 x value
    v_kmeans = KMeans(n_clusters=19, random_state=None).fit(vert_lines[:, 0].reshape(-1, 1))

    # plot vertical grid lines
    v_grid_lines = []
    for center in v_kmeans.cluster_centers_:
        cv2.line(stones_img, (int(center), top_left[1]), (int(center), bottom_right[1]), (0, 255, 0), 4)
        cv2.line(final, (int(center), top_left[1]), (int(center), bottom_right[1]), (0, 255, 0), 4)
        v_grid_lines.append([int(center), top_left[1], int(center), bottom_right[1]])
    v_grid_lines = np.reshape(v_grid_lines, (len(v_grid_lines), 4))
    v_grid_lines = v_grid_lines[v_grid_lines[:, 0].argsort()]

    return v_grid_lines, stones_img, final


"""
Uses reconstructed grid lines from grid detection pipeline to extract stone data. Looks
at the intersections of all of the lines and aggregates pixel data of areas around all the 
intersections and clusters pixel data into 3 clusters: white, black and empty (blue)
"""
def get_stones(h_grid_lines, v_grid_lines, color_thresh_img, stones_img):
    # aggregating intersection data for each intersection of detected grid lines
    intersections = []
    stone_regions = []
    for hline in h_grid_lines:
        for vline in v_grid_lines:
            x, y = get_intersection(hline, vline)
            intersection = (x, y)
            intersections.append(intersection)

            # separating out circular radius around intersection
            mask = np.zeros(img1.shape, np.uint8)
            radius = 10
            mask = cv2.circle(mask, intersection, radius, (255, 255, 255), -1)
            region = np.where(mask == (255, 255, 255))
            values_from_original = color_thresh_img[region[0], region[1]]
            average_of_radius = np.mean(values_from_original, axis=0)

            stone_regions.append(average_of_radius)

    # clustering intersection area pixel data into 3 clusters
    np.reshape(stone_regions, (len(stone_regions), 3))
    stone_kmeans = KMeans(n_clusters=3, random_state=None).fit(stone_regions)
    stone_labels = np.array(stone_kmeans.labels_)

    label_map = {}
    black_thresh = 60
    white_thresh = 200

    for i in range(len(stone_kmeans.cluster_centers_)):
        if np.count_nonzero(stone_kmeans.cluster_centers_[i] < black_thresh) == 3:
            label_map[i] = tuple((0, 0, 0))
            continue
        if np.count_nonzero(stone_kmeans.cluster_centers_[i] > white_thresh) == 3:
            label_map[i] = tuple((255, 255, 255))
            continue
        label_map[i] = "empty"

    # plotting labeled stones on final stone image
    for i in range(len(stone_labels)):
        if label_map[stone_labels[i]] != "empty":
            cv2.circle(stones_img, intersections[i], radius=15, color=label_map[stone_labels[i]], thickness=4)

    return stones_img




if __name__ == "__main__":
    # load image
    img1 = cv2.imread('./image_data/6.jpg', 1)

    # select corners of game board in input image
    a = []
    b = []
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            a.append(x)
            b.append(y)
            cv2.circle(img1, (x, y), 1, (0, 0, 255), thickness=-1)
            cv2.putText(img1, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 0), thickness=1)
            cv2.imshow("image", img1)
            print(x, y)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img1)
    cv2.waitKey(0)

    # rectify image using selected 4 points
    img1 = perspective_transformation(a, b, img1)
    cv2.imwrite('rectified_image5.jpg', img1)

    final_grid_img = img1.copy()

    # threshold on color image, make black and white more intense
    white = [255, 255, 255]
    black = [0, 0, 0]
    white_thresh = [160, 160, 160]
    black_thresh = [50, 50, 50]
    img1[np.where((img1 > white_thresh).all(axis=2))] = white
    img1[np.where((img1 < black_thresh).all(axis=2))] = black

    cv2.imwrite('processed6.jpg', img1)

    # color_thresh_img is for color reference for STONE RECOGNITION, turns background of board blue
    color_thresh_img = img1.copy()
    color_thresh_img[np.where((np.logical_and(color_thresh_img < white, color_thresh_img > black)).all(axis=2))] = [255, 0, 0]
    stones_img = np.ones((img1.shape[0], img1.shape[1], 3), np.uint8)
    stones_img[:, :, :] = (100, 100, 100)
    #cv2.imwrite('color_thresh5.jpg', color_thresh_img)

    # GRID DETECTION PIPELINE
    final_grid_img, lines, top_left, bottom_right, top_right, bottom_left = get_board_lines(img1, final_grid_img)
    h_grid_lines, stones_img, final_grid_img = get_horizontal_grid_lines(lines, stones_img, final_grid_img, top_left, bottom_right, top_right, bottom_left)
    v_grid_lines, stones_img, final_grid_img = get_vertical_grid_lines(lines, stones_img, final_grid_img, top_left, bottom_right, top_right, bottom_left)
    cv2.imwrite('board_final5.jpg', final_grid_img)

    # STONE DETECTION PIPELINE
    stones_img = get_stones(h_grid_lines, v_grid_lines, color_thresh_img, stones_img)
    cv2.imwrite('stones_final5.jpg', stones_img)







