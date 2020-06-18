import cv2
import os
import random
import math
import numpy as np
from copy import deepcopy
from copy import copy

feature_points_num = 10
threshold = 3
matrix_size = 3
window_size = 3
LEFT_BASE = 0
RIGTH_BASE = 1
avg_points = []

def getFeatures(img):
    global sift
    temp_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    temp_kp, temp_des = sift.detectAndCompute(temp_img, None)
    return temp_kp, temp_des

def matchFeatures(des1, des2):
    global match
    matches = match.knnMatch(des2, des1, k=2)
    index = 0

    while index < len(matches):
        if matches[index][0].distance * 1.3 >= matches[index][1].distance:
            matches.pop(index)
        else:
            matches[index] = matches[index][0]
            index += 1
    return matches

def getH(initial_points, projected_points):
    A = []
    b = []
    for i in range(len(initial_points)):
        x, y, z = initial_points[i][0], initial_points[i][1], initial_points[i][2]
        u, v, w = projected_points[i][0], projected_points[i][1], projected_points[i][2]
        A.append([-x, -y, -z, 0, 0, 0, u * x, u * y])
        A.append([0, 0, 0, -x, -y, -z, v * x, v * y])
        b.extend([-u, -v])
    H = list(np.linalg.lstsq(A, b, rcond=-1)[0])
    H.append(1)
    H = np.mat([[H[i * 3 + j] for j in range(3)] for i in range(3)])
    return H

def getInliners(kp1, kp2, matches, H):
    inliners = []
    for match in matches:
        predict_value = np.array(np.dot(H, kp2[match.queryIdx].pt + (1,)))[0]
        for i in range(3):
            predict_value[i] = predict_value[i] / predict_value[2]
        real_value = kp1[match.trainIdx].pt + (1,)
        error = np.sqrt(np.sum(np.square(np.array(predict_value) - np.array(real_value))))
        if error < threshold:
            inliners.append(match)
    return inliners

def ranSAC(matches, kp1, kp2):
    global feature_points_num, threshold
    # 判断共线点数
    """
    while True:
        indexs = random.sample(range(len(matches)), feature_points_num)
        initial_points = [kp2[matches[indexs[i]].queryIdx].pt + (1,) for i in range(4)]
        inSameLine = 0
        for i in range(4):
            temp_index = [0, 1, 2, 3]
            temp_index.pop(i)
            lines = [[initial_points[j][k] - initial_points[j+1][k] for k in range(2)] for j in range(2)]
            if (np.linalg.matrix_rank(np.mat(lines)) != 2):
                inSameLine += 1
        if inSameLine < feature_points_num - 4:
            break
    """
    H_best = None
    I_most = 0
    iter_times = int(math.log(0.01) / math.log(1 - pow(0.9, feature_points_num)))
    threshold = 3
    best_inliners = []
    for k in range(iter_times):
        indexs = random.sample(range(len(matches)), feature_points_num)
        initial_points = [kp2[matches[indexs[i]].queryIdx].pt + (1,) for i in range(feature_points_num)]
        projected_points = [kp1[matches[indexs[i]].trainIdx].pt + (1,) for i in range(feature_points_num)]
        H = getH(initial_points, projected_points)
        inliners = getInliners(kp1, kp2, matches, H)
        if len(inliners) <= 10:
            continue
        initial_points = [kp2[match.queryIdx].pt + (1,) for match in inliners]
        projected_points = [kp1[match.trainIdx].pt + (1,) for match in inliners]
        H = getH(initial_points, projected_points)
        inliners = getInliners(kp1, kp2, matches, H)
        if len(inliners) > I_most:
            H_best = H
            I_most = len(inliners)
            best_inliners = inliners
    return H_best, best_inliners
"""
# version2
def fillIndex(start, end, index):
    if start+2>end:
        return
    if index[start] == index[end]:
        mid = int((start + end) / 2)
        index[mid] = 1 - index[start]
        step = (index[mid] - index[start]) / (mid - start)
        for i in range(start + 1, mid):
            index[i] = index[i - 1] + step
        step = (index[end] - index[mid]) / (end - mid)
        for i in range(mid + 1, end):
            index[i] = index[i - 1] + step
    else:
        step = index[end] - index[start] / (end - start)
        for i in range(start + 1, end):
            index[i] = index[i - 1] + step
def getBlending(img_raw, img_result,aaaaa):
    rows, cols = img_result.shape[:2]
    result = np.zeros([rows, cols, 3], np.uint8)
    for row in range(0, rows):
        # make a row_index for each line with marks
        col_index = []
        for col in range(0, cols):
            #  img_raw \ img_result |     0       1
            #  ------------------------------------------------------
            #        0              |     -1      0
            #        1              |     1       0.5
            if col >= img_raw.shape[1] or not img_raw[row, col].any(): # 旧图无
                if img_result[row, col].any(): # 新图有
                    col_index.append(0)
                else:
                    col_index.append(1)
            else:
                if img_result[row, col].any(): # 新图有
                    col_index.append(0.5)
                else:
                    col_index.append(0)
        col = 0
        while (col < cols):
            if col_index[col] == 0.5:
                start = col 
                while(col < cols and col_index[col] == 0.5):
                    col += 1
                end = col-1
                fillIndex(start, end, col_index)
            else:
                col += 1
        for col in range(0, cols):
            if col_index[col] == -1:
                result[row, col] = np.array([0, 0, 0])
            elif col >= img_raw.shape[1]:
                result[row, col] = img_result[row, col]
            else:
                alpha = col_index[col]
                result[row, col] = np.clip(img_raw[row, col] * alpha + img_result[row, col] * (1 - alpha), 0, 255)
    left, right = 0, 0
    for col in range(0, result.shape[:2][1]):
        if result[:, col].any():  # 合成图的最左端
            left = col
            break
    for col in range(result.shape[:2][1] - 1, 0, -1):
        if result[:, col].any():  # 合成图的最右端
            right = col
            break
    for row in range(0, result.shape[:2][0]):
        if result[row, :].any():  # 合成图的最左端
            top = row
            break
    for row in range(result.shape[:2][0] - 1, 0, -1):
        if result[row, :].any():  # 合成图的最左端
            bottom = col
            break
    result = result[top:bottom, left: right]
    return result
"""
# version1

def judgeBlack(row, col, img):
    global matrix_size, avg_points
    if col <= matrix_size:
        lp = 0
        rp = col + matrix_size
    elif col >= img.shape[1] - matrix_size - 1:
        lp = col - matrix_size
        rp = img.shape[1] - 1
    else:
        lp = col - matrix_size
        rp = col + matrix_size

    if row <= matrix_size:
        up = 0
        bp = row + matrix_size
    elif row >= img.shape[0] - matrix_size - 1:
        up = row - matrix_size
        bp = img.shape[0] - 1
    else:
        up = row - matrix_size
        bp = row + matrix_size

    for pixels in img[up:bp, lp:rp]:
        if np.array([0, 0, 0]) in pixels or 0 in pixels:
            avg_points.append([row, col])
            return True
    return False

def getAvg(point, res):
    global matrix_size
    row, col = point[0], point[1]
    if col <= window_size:
        lp = 0
        rp = col + window_size
    elif col >= res.shape[1] - window_size - 1:
        lp = col - window_size
        rp = res.shape[1] - 1
    else:
        lp = col - window_size
        rp = col + window_size

    if row <= window_size:
        up = 0
        bp = row + window_size
    elif row >= res.shape[0] - window_size - 1:
        up = row - window_size
        bp = res.shape[0] - 1
    else:
        up = row - window_size
        bp = row + window_size

    avg = np.array([0, 0, 0])
    size = 0
    for pixels in res[up:bp, lp:rp]:
        for pixel in pixels:
            if pixel.any():
                size += 1
                avg = avg + pixel
    avg = np.clip(avg / size, 0, 255)
    return avg

def extendList(points, res):
    global window_size
    result = []
    for point in points:
        global window_size
        row, col = point[0], point[1]
        if col <= window_size:
            lp = 0
            rp = col + window_size
        elif col >= res.shape[1] - window_size - 1:
            lp = col - window_size
            rp = res.shape[1] - 1
        else:
            lp = col - window_size
            rp = col + window_size

        if row <= window_size:
            up = 0
            bp = row + window_size
        elif row >= res.shape[0] - window_size - 1:
            up = row - window_size
            bp = res.shape[0] - 1
        else:
            up = row - window_size
            bp = row + window_size

        for i in range(up, bp + 1):
            for j in range(lp, rp + 1):
                result.append((i, j))

    return list(set(result))

def getBlending(img_raw, img_result, base):
    rows, cols = img_raw.shape[:2]
    for col in range(0, cols):
        if img_raw[:, col].any() and img_result[:, col].any():  # 开始重叠的最左端
            left = col
            break
    for col in range(cols - 1, 0, -1):
        if img_raw[:, col].any() and img_result[:, col].any():  # 重叠的最右一列
            right = col
            break
    res = np.zeros([rows, cols, 3], np.uint8)
    avg_points = []
    for row in range(0, rows):
        for col in range(0, cols):
            srcImgLen = float(abs(col - left))
            testImgLen = float(abs(col - right))
            alpha = srcImgLen / (srcImgLen + testImgLen)
            if not img_raw[row, col].any():  # 如果没有原图，用旋转的填充
                res[row, col] = img_result[row, col]
            elif not img_result[row, col].any():
                res[row, col] = img_raw[row, col]
            else:
                if judgeBlack(row, col, img_result):
                    avg_points.append((row, col))
                    res[row, col] = img_raw[row, col]
                    continue
                # cv2.circle(img1111, (col, row), 1, (0, 255, 0), -1)
                if base == RIGTH_BASE:
                    res[row, col] = np.clip(img_raw[row, col] * alpha + img_result[row, col] * (1 - alpha), 0, 255)
                else:
                    res[row, col] = np.clip(img_raw[row, col] * (1 - alpha) + img_result[row, col] * alpha, 0, 255)
    # print(len(avg_points))
    # avg_points = extendList(avg_points, img_result)
    # print(len(avg_points))
    img_result[0:img_raw.shape[0], 0:img_raw.shape[1]] = res
    # for point in avg_points:
    #    img_result[point[0], point[1]] = getAvg(point, img_result)
    for col in range(0, img_result.shape[:2][1]):
        if img_result[:, col].any():  # 合成图的最左端
            left = col
            break
    for col in range(img_result.shape[:2][1] - 1, 0, -1):
        if img_result[:, col].any():  # 合成图的最右端
            right = col
            break
    img_result = img_result[:, left: right]
    return img_result

def contain(src, u, v):
    return u>=0 and u<src.shape[0]-1 and v>=0 and v<src.shape[1]-1

def pointSampling(src, u, v):
    u = int(u+0.5)
    v = int(v+0.5)
    if not contain(src, u, v):
        return deepcopy(src[0,0])
    return deepcopy(src[u,v])

def toFloat(a):
    return np.asarray(a, dtype=np.float)

def toInt(a):
    return np.asarray(a, dtype=np.uint8)

def bilinear(a, b, w):
    a = toFloat(a)
    b = toFloat(b)
    return a*w + b*(1-w);

def triangleFiltering(src, u, v):
    x = [int(u),int(u+1)]
    y = [int(v),int(v+1)]
    a = bilinear(src[x[0],y[0]], src[x[0],y[1]], u-x[0])
    b = bilinear(src[x[1],y[0]], src[x[1],y[1]], u-x[0])
    # print (a,b)
    return toInt(bilinear(a, b, v-y[0]))

def warp(src, H, shape):
    H = np.linalg.inv(H)
    dst = np.ndarray((shape[0],shape[1],3),dtype=np.uint8)
    if H.shape != (3,3):
        print ("wrong homnography")
        print (H)
        exit(0)
    for x in range(shape[1]):
        for y in range(shape[0]):
            v = (H[0,0]*x + H[0,1]*y + H[0,2]) / (H[2,0]*x + H[2,1]*y + H[2,2])
            u = (H[1,0]*x + H[1,1]*y + H[1,2]) / (H[2,0]*x + H[2,1]*y + H[2,2])
            # t = H.dot(np.array([x,y,1]))
            # t = t / t[2]
            # v, u = t[0], t[1]
            if contain(src, u, v):
                dst[y,x] = pointSampling(src, u, v)
                #dst[y,x] = triangleFiltering(src, u, v)
    return dst

def count(a):
    # count numbers in ndarray
    unique, counts = np.unique(a, return_counts=True)
    return dict(zip(unique, counts))

def combineImg(img1, img2, base):
    global sift, match, avg_points

    # kp1, des1 = sift.detectAndCompute(img1, None)
    # kp2, des2 = sift.detectAndCompute(img2, None)
    # matches = matchFeatures(des1, des2)
    # H, inliners = ranSAC(matches, kp1, kp2)
    # img_result = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
    # cv2.imshow('1', img_result)
    # cv2.waitKey(0)
    # exit(0)
    
    # if base == RIGTH_BASE:
        # img1 = cv2.copyMakeBorder(img1, 0, 0, img2.shape[1], 0, cv2.BORDER_CONSTANT, value=0)
    img1 = cv2.copyMakeBorder(img1, img1.shape[1], img1.shape[0], img1.shape[1], img1.shape[0], cv2.BORDER_CONSTANT, value=0)
    img2 = cv2.copyMakeBorder(img2, img1.shape[1], img1.shape[0], img1.shape[1], img1.shape[0], cv2.BORDER_CONSTANT, value=0)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    matches = matchFeatures(des1, des2)
    H, inliners = ranSAC(matches, kp1, kp2)
    # matches = matchFeatures(des2, des1)
    # H, inliners = ranSAC(matches, kp2, kp1)
    print (H)
    H = np.asarray(H)
    # if base == RIGTH_BASE:
        # img_result = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
    img_result = warp(img2, H, (3*img1.shape[0], 3*img1.shape[1]))
    # else:
        # img_result = cv2.warpPerspective(img2, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    # img_result = warp(img2, H, (img1.shape[0], img1.shape[1] + img2.shape[1]))

    # cv2.imshow('1', img_result)
    # cv2.waitKey(0)
    # exit(0)
    img_result = getBlending(img1, img_result, base)
    return img_result

def getImgs():
    imgs = []
    base = 'source'
    dir_path = input('Please input file name:')
    dir_path = os.path.join(base,dir_path)
    files = os.listdir(dir_path)
    for file in files:
        imgs.append(cv2.imread(os.path.join(dir_path, file)))
    return imgs

if __name__ == "__main__":
    match = cv2.BFMatcher()
    sift = cv2.xfeatures2d.SIFT_create()
    imgs = getImgs()
    # 从中央开始合成图片
    img_num = len(imgs)
    if img_num % 2 == 1:
        base_img = imgs[int(img_num / 2)]
        for i in range(int(img_num / 2)):
            base_img = combineImg(base_img, imgs[int(img_num / 2) - 1 - i], RIGTH_BASE)
            base_img = combineImg(base_img, imgs[int(img_num / 2) + 1 + i], LEFT_BASE)
    else:
        base_img = combineImg(imgs[int(img_num / 2) - 1], imgs[int(img_num / 2)], LEFT_BASE)
        for i in range(int(img_num / 2) - 1):
            base_img = combineImg(base_img, imgs[int(img_num / 2) - 2 - i], RIGTH_BASE)
            base_img = combineImg(base_img, imgs[int(img_num / 2) + 1 + i], LEFT_BASE)

    # 从最左开始合成图片
    """
    base_img = combineImg(imgs[0], imgs[1], LEFT_BASE)
    for i in range(2, len(imgs)):
        base_img = combineImg(base_img, imgs[i], LEFT_BASE)
    """
    cv2.imshow("original_image_drawMatches.jpg", base_img)
    cv2.waitKey(0)
#    cv2.imwrite(dir_path + ".jpg", base_img)

#draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                   singlePointColor = None,
#                   flags = 2)
#img3 = cv2.drawMatches(img2, kp2, img1, kp1, inliners, None, **draw_params)
#cv2.imshow("Match Image", img3)
