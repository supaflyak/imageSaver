import cv2 as cv
import argparse
import os
import datetime
import numpy as np


total_rectangle = 9


class HandInfo:
    def __init__(self, hand_coord, fingers):
        self.hand_coord = hand_coord
        self.fingers = fingers


def draw_rect(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    # x_coords = [6, 9, 12]
    # y_coords = [9, 10, 11]
    # scale_factor = 20

    x_coords = [6, 8, 10]
    y_coords = [9, 10, 11]
    scale_factor = 20

    hand_rect_one_x = np.array(
        [x_coords[0] * rows / 20, x_coords[0] * rows / 20, x_coords[0] * rows / 20,
         x_coords[1] * rows / 20, x_coords[1] * rows / 20, x_coords[1] * rows / 20,
         x_coords[2] * rows / 20, x_coords[2] * rows / 20, x_coords[2] * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [y_coords[0] * cols / 20, y_coords[1] * cols / 20, y_coords[2] * cols / 20,
         y_coords[0] * cols / 20, y_coords[1] * cols / 20, y_coords[2] * cols / 20,
         y_coords[0] * cols / 20, y_coords[1] * cols / 20, y_coords[2] * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame


def hist_masking(frame, hist):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    dst = cv.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (21, 21))
    cv.filter2D(dst, -1, disc, dst)

    ret, thresh = cv.threshold(dst, 120, 255, cv.THRESH_BINARY)

    # thresh = cv.dilate(thresh, None, iterations=5)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=7)

    thresh = cv.merge((thresh, thresh, thresh))

    return cv.bitwise_and(frame, thresh)


def contours(hist_mask_image):
    gray_hist_mask_image = cv.cvtColor(hist_mask_image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray_hist_mask_image, 0, 255, 0)
    cont, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return cont


def centroid(max_contour):
    moment = cv.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None


def manage_image_opr(frame, hand_hist, debug_flag=False):
    hist_mask_image = hist_masking(frame, hand_hist)
    # hist_mask_image = cv.erode(hist_mask_image, None, iterations=2)
    hist_mask_image = cv.dilate(hist_mask_image, None, iterations=2)

    # fill in the holes of the masked image
    fill_mask = cv.cvtColor(hist_mask_image, cv.COLOR_RGB2GRAY)
    _, fill_mask = cv.threshold(fill_mask, 1, 255, cv.THRESH_BINARY_INV)
    fill_mask = cv.floodFill(fill_mask,
                             np.zeros((hist_mask_image.shape[0] + 2, hist_mask_image.shape[1] + 2), np.uint8),
                             seedPoint=(0, 0),
                             newVal=255)[1]
    _, fill_mask = cv.threshold(fill_mask, 1, 255, cv.THRESH_BINARY_INV)
    fill_mask[fill_mask == 255] = 1
    final_mask = np.zeros_like(hist_mask_image)
    final_mask[:, :, 0] += fill_mask
    final_mask[:, :, 1] += fill_mask
    final_mask[:, :, 2] += fill_mask
    hist_mask_image = final_mask * frame
    _, hand_data_cascade = find_hands_cascade(hist_mask_image, hand_cascade)
    _, hand_data_mask = find_hands_mask(hist_mask_image)
    hand_data = hand_data_mask

    cv.imshow("mask", final_mask * 255)
    cv.imshow("what it look like", hist_mask_image)

    contour_list = []
    centroids = []
    gray = cv.cvtColor(hist_mask_image, cv.COLOR_RGB2GRAY)
    _, thresh = cv.threshold(gray, 120, 255, cv.THRESH_BINARY)
    for hand_location in hand_data:
        roi_mask = np.zeros_like(thresh)
        roi_mask[hand_location.hand_coord[1]:hand_location.hand_coord[1] + hand_location.hand_coord[3],
                 hand_location.hand_coord[0]:hand_location.hand_coord[0] + hand_location.hand_coord[2]] = 1
        masked_hand = np.bitwise_and(roi_mask, thresh)
    #     contour_list.append(cv.findContours(masked_hand, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE))
    # for contours in contour_list:
    #     if len(contours) > 0:
    #         max_cont = max(contours[0], key=cv.contourArea)
    #         centroids.append(centroid(max_cont))

    if debug_flag:
        for hand in hand_data:
            frame = cv.rectangle(frame, (hand.hand_coord[0], hand.hand_coord[1]),
                                 (hand.hand_coord[0] + hand.hand_coord[2], hand.hand_coord[1] + hand.hand_coord[3]),
                                 (0, 0, 255), 3)
            frame = cv.putText(frame, str(hand.fingers), (hand.hand_coord[0], hand.hand_coord[1]),
                               cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), thickness=2)
        for center in centroids:
            frame = cv.circle(frame, center, 3, [128,128,128], 3)

    return frame, hand_data


def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    hand_hist = cv.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv.normalize(hand_hist, hand_hist, 0, 255, cv.NORM_MINMAX)


def img_str(im_counter):
    num = str(im_counter)
    while num.__len__() < 4:
        num = "0" + num
    return num


# Takes masked color image and returns all of the bounding boxes of hands
def find_hands_cascade(image, hand_cas):
    hand_data = []
    gray = cv.cvtColor(image.copy(), cv.COLOR_RGB2GRAY)
    _, thresh = cv.threshold(gray, 120, 255, cv.THRESH_BINARY)
    # cv.imshow("gray", gray)
    # cv.imshow("thresh", thresh)
    hand = hand_cas.detectMultiScale(thresh, 1.1, 5)
    # masked_hands = np.zeros_like(gray)
    if hand.__len__() > 0:
        for (x, y, w, h) in hand:
            ones_mask = np.zeros_like(gray)
            ones_mask[y:y+h, x:x+w] = 1
            masked_image = ones_mask * thresh
            # masked_hands = np.bitwise_or(masked_hands, masked_image)
            # contours, _ = cv.findContours(masked_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            # cv.drawContours(image, contours, -1, (0, 0, 255), 2)
            # cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
            hand_data.append(HandInfo((x, y, w, h), count_fingers(masked_image)))
    # cv.imshow("masks", masked_hands)
    return image, hand_data


def find_hands_mask(image):
    hand_data = []
    gray = cv.cvtColor(image.copy(), cv.COLOR_RGB2GRAY)
    _, thresh = cv.threshold(gray, 0, 255, 0)
    _, markers = cv.connectedComponents(thresh)
    # markers = markers + 1
    max = markers.max()
    if markers.max() == 0:
        max = 1
    cv.imshow("markers", markers * 255 / max)
    if markers.max() > 0:
        for hand_num in range(1, markers.max()):
            if np.count_nonzero(markers == hand_num) > 20:
                ones_mask = np.zeros_like(gray)
                ones_mask[markers == hand_num] = 1
                masked_image = ones_mask * thresh
                # masked_hands = np.bitwise_or(masked_hands, masked_image)
                # contours, _ = cv.findContours(masked_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                # cv.drawContours(image, contours, -1, (0, 0, 255), 2)
                # cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
                contour, _ = cv.findContours(ones_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                rect = cv.boundingRect(contour[0])
                hand_data.append(HandInfo(rect, count_fingers(masked_image)))
    # cv.imshow("masks", masked_hands)
    return image, hand_data


# takes a theshold image of hands and the corresponding HandInfo and returns the number of fingers for each hand
def count_fingers(image_with_hand):
    contour, _ = cv.findContours(image_with_hand, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contour) > 0:
        # Find largest contour
        maxIndex = 0
        maxArea = 0
        for i in range(len(contour)):
            cnt = contour[i]
            area = cv.contourArea(cnt)
            if area > maxArea:
                maxArea = area
                maxIndex = i

        contour = contour[i]
        hull = cv.convexHull(contour, returnPoints=False)
        if len(hull) > 3:
            defects = cv.convexityDefects(contour, hull)
            cnt = 0
            if type(defects) != type(None):
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(contour[s, 0])
                    end = tuple(contour[e, 0])
                    far = tuple(contour[f, 0])
                    angle = calculate_angle(far, start, end)

                    # Ignore the defects which are small and wide
                    # Probably not fingers
                    if d > 10000 and angle <= np.pi / 2:
                        cnt += 1
                return True, cnt
            return False, 0


def calculate_angle(far, start, end):
    """Cosine rule"""
    a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
    c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
    angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
    return angle


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="save images")
    parser.add_argument('--fps', metavar='fps', type=int, default=30)
    parser.add_argument('--file', metavar='save file path', type=str,
                        default=os.path.join(os.path.curdir,
                        os.path.abspath("default_savepath")))
    parser.add_argument('--overwrite', type=bool, default=False)

    global hand_cascade
    hand_cascade = cv.CascadeClassifier("Hand_haar_cascade.xml")
    args = parser.parse_args()
    file_path = args.file
    fps = .001 / args.fps
    overwrite_images = args.overwrite

    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    # capture and recording variables
    cap = cv.VideoCapture(0)
    print("press Space Bar to start and stop capturing")
    start_saving = False
    im_counter = 0

    # histogram variables
    global hand_hist
    is_hand_hist_created = False

    debug_flag = False

    if not overwrite_images:
        im_counter = os.listdir(file_path).__len__()

    while cap.isOpened():
        pressed_key = cv.waitKey(1)
        _, frame = cap.read()
        start_time = datetime.datetime.now().microsecond
        frame_count = img_str(im_counter)
        # frame, hand_data = find_hands(frame, hand_cascade)
        if is_hand_hist_created:
            frame, hand_info = manage_image_opr(frame, hand_hist, debug_flag=debug_flag)
        else:
            frame = draw_rect(frame)

        #save image
        if start_saving:
            if len(hand_info) == 1:
                y = hand_info[0].hand_coord[1]
                x = hand_info[0].hand_coord[0]
                w = hand_info[0].hand_coord[2]
                h = hand_info[0].hand_coord[3]
                cv.imwrite
                cv.imwrite(os.path.join(file_path, str(frame_count) + ".png"), frame[y:y+h, x:x+h,:])
                im_counter += 1

        # write onto the image
        frame = cv.putText(frame, frame_count, (frame.shape[0] - 50, 50),
                           cv.FONT_HERSHEY_COMPLEX, 1, color=(128, 255, 128), thickness=2)

        cv.imshow("frame", frame)
        if pressed_key:
            key = pressed_key & 0xFF
            if key == ord('z') and not is_hand_hist_created:
                is_hand_hist_created = True
                hand_hist = hand_histogram(frame)
            if key == ord('d'):
                debug_flag = not debug_flag
            if key == ord(' '):
                start_saving = not start_saving
            if key == ord('q'):
                break
    cv.destroyAllWindows()
    cap.release()
    quit(1)
