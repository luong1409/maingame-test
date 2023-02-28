import cv2
import numpy as np

# frame_width = 640
# frame_height = 480

# def empty(a):
#     pass

# cv2.namedWindow("Parameters")
# cv2.resizeWindow("Parameters", 640, 240)
# cv2.createTrackbar("Threshold1", "Parameters", 248, 255, empty)
# cv2.createTrackbar("Threshold2", "Parameters", 173, 255, empty)

# def vconcat_resize(img_list, scale=2, interpolation=cv2.INTER_CUBIC):
#     height, width = img_list[0].shape[:2]
    
#     new_height = scale * height
#     new_width = scale * width
#     # resizing images
#     resized_img_list = []
#     for i, img in enumerate(img_list):
#         if len(img.shape) < 3:
#             new_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#             new_img = cv2.resize(src=new_img, dsize=(new_width, new_height), interpolation=interpolation)
#             resized_img_list.append(new_img)
#         else:
#             new_img = cv2.resize(src=img, dsize=(new_width, new_height), interpolation=interpolation)
#             resized_img_list.append(new_img)
#     # return final image
#     return cv2.vconcat(resized_img_list)

def get_contours(img):
    contours, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    rects = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 900:
            peri = cv2.arcLength(curve=cnt, closed=True)
            approx = cv2.approxPolyDP(curve=cnt, epsilon=0.02 * peri, closed=True)
            # print(approx)
            x, y, w, h = cv2.boundingRect(array=approx)
            # cv2.rectangle(img=img_contour, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=5)
            rects.append((x, y, w, h))
    return rects


def detect_square_hero(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    img_blur = cv2.GaussianBlur(img, (7, 7), 1)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

    threshold1 = 228
    threshold2 = 220

    img_canny = cv2.Canny(img_gray, threshold1=threshold1, threshold2=threshold2)
    kernel = np.ones((5,5))
    img_dil = cv2.dilate(img_canny, kernel=None, iterations=1)
    
    rects = get_contours(img_dil)
    
    return rects


    # img_stack = vconcat_resize(img_list=[img, img_canny, img_dil, img_contour], scale=4)

#     cv2.imshow("Result", img_stack)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
# cv2.destroyAllWindows()