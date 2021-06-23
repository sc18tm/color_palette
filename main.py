import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
import copy
import time
from IPython.display import Image, display
from colornamer import get_color_from_rgb
from skimage import io, segmentation, color

from sklearn.cluster import KMeans
from PIL import Image, ImageDraw

import pandas as pd
import numpy as np
import scipy
from scipy.cluster.hierarchy import linkage,dendrogram,fcluster


def generate_paths(dir_original, dir_segmented):
        # ファイル名を取得
    paths_original = glob.glob(dir_original + "/*")
    paths_segmented = glob.glob(dir_segmented + "/*")

    if len(paths_original) == 0 or len(paths_segmented) == 0:
        raise FileNotFoundError("Could not load images.")
    # 教師画像の拡張子を.pngに書き換えたものが読み込むべき入力画像のファイル名になります
    filenames = list(map(lambda path: path.split(os.sep)[-1].split(".")[0], paths_segmented))
    paths_original = list(map(lambda filename: dir_original + "/" + filename + ".jpg", filenames))

    return paths_original, paths_segmented

def image_generator(file_paths, init_size=(256,256), normalization=True, antialias=False):

    for file_path in file_paths:
        if file_path.endswith(".png") or file_path.endswith(".jpg"):
            # open a image
            image = Image.open(file_path)
            # to square
            image = crop_to_square(image)
            # resize by init_size
            if init_size is not None and init_size != image.size:
                if antialias:
                    image = image.resize(init_size, Image.ANTIALIAS)
                else:
                    image = image.resize(init_size)
            # delete alpha channel
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image = np.asarray(image)
            if normalization:
                image = image / 255.0
            yield image

def crop_to_square(image):
    size = min(image.size)
    left, upper = (image.width - size) // 2, (image.height - size) // 2
    right, bottom = (image.width + size) // 2, (image.height + size) // 2
    return image.crop((left, upper, right, bottom))

def create_colorpalette(img_src, img_mask,file_name):
    segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=0.7, k=80, min_size=5)
    segment = segmentator.processImage(img_src)
    RGB_gs = [[0] * 3] * (segment.max()+1)#各領域のRGB情報
    count_gs = [0] * (segment.max()+1)#各領域のpixel数
    for i in range(256):
        for j in range(256):
            RGB_gs[segment[i][j]] += img_src[i][j]
            count_gs[segment[i][j]] += 1
    for i in range(len(RGB_gs)):
        RGB_gs[i] = RGB_gs[i]/count_gs[i]

    for i in range(256):
        for j in range(256):
            img_src[i][j] = RGB_gs[segment[i][j]]

    im_rgb = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
    img_masked = cv2.bitwise_and(im_rgb, im_rgb, mask=img_mask )
    img_masked_rgb = cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB)
    io.imsave("./middle/masked_input/"+file_name,img_masked_rgb)

    RGB_in_mask = [[0] * 3] * (segment.max()+1)#mask内の各領域のRGB情報
    count_in_mask = [0] * (segment.max()+1)#mask内の各領域のpixel数

    for i in range(256):
        for j in range(256):
            if img_masked_rgb[i][j].sum() != 0:
                count_in_mask[segment[i][j]]+=1
    del_counter = 0
    RGB_in_mask = copy.deepcopy(RGB_gs)
    for i in range(len(count_in_mask)):
        if(count_in_mask[i] == 0):
            del RGB_in_mask[i-del_counter]
            del_counter += 1
    for i in range(del_counter):
        count_in_mask.remove(0)

#マスク内にある各領域の合計ピクセル数->count_in_mask
#マスク内にある各領域のRGBの種類のリスト->RGB_in_mask
    Z = linkage(RGB_in_mask, method='complete', metric='euclidean')
    pd.DataFrame(Z)
    clusters = fcluster(Z,130,criterion= "distance")

    RGB_cl = [[0] * 3] * clusters.max()#代表色のRGB
    hist = [0] * clusters.max()#各代表色の割合
    count_cl = [0] * clusters.max()#各代表色のpixel数

    cutoff_rate = 0.08

    for i in range(len(clusters)):
        count_cl[clusters[i]-1] +=1
        hist[clusters[i]-1] += count_in_mask[i]
        RGB_cl[clusters[i]-1] += RGB_in_mask[i]
    for i in range(len(hist)):
        if hist[i] < sum(hist)*cutoff_rate:
            hist[i] = 0.0
    hist_sum = sum(hist)
    for i in range(clusters.max()):
        RGB_cl[i] = RGB_cl[i]/count_cl[i]
        hist[i] = hist[i]/hist_sum

    before_sort_hist = copy.deepcopy(hist)
    sorted_RGB = []
    for i in range(len(hist)):
        for j in range(1, len(hist)-i, 1):
            if(hist[j-1]<hist[j]):
                hist[j],hist[j-1] = hist[j-1],hist[j]

    for i in range(len(hist)):
        for j in range(len(hist)):
            if hist[i] == before_sort_hist[j]:
                sorted_RGB.append(RGB_cl[j])
    for i in range(len(sorted_RGB)):
        sorted_RGB[i][0],sorted_RGB[i][2] = sorted_RGB[i][2],sorted_RGB[i][0]
    plot_colors(hist,sorted_RGB,file_name)

def plot_colors(hists,rgbs,file_name):
    W = 500
    H = 100
    start = 0
    end = 0
    im = Image.new('RGB', (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    for i in range(len(hists)):
        r = int(rgbs[i][2])
        g = int(rgbs[i][1])
        b = int(rgbs[i][0])
        end += hists[i]
        draw.rectangle((start*W,0,end*W,W),fill=(r,g,b))
        start = end
    #file_name = "colorpalatte" + file_name
    im.save('./output/'+file_name, quality=95)

if __name__ == '__main__':
    dir_src = "./src"
    dir_pred = "./res_unet_22cat_3cat_ss"

    paths_src,paths_pred = generate_paths(dir_src,dir_pred)
    src_images,pred_images = [],[]

    init_size = (256,256)

    for image in image_generator(paths_src, init_size, antialias=True,normalization=False):
        src_images.append(image)

    for image in image_generator(paths_pred, init_size, normalization=False):
        pred_images.append(image)

    src_images = np.asarray(src_images, dtype=np.uint8)
    pred_images = np.asarray(pred_images, dtype=np.uint8)
#maskは服以外を0、服を1にする
    pred_images = np.where(pred_images == 1, 0, pred_images)
    pred_images = np.where(pred_images == 2, 1, pred_images)
    for (src_image, pred_image,filename) in zip(src_images, pred_images,paths_src):
        filename = os.path.basename(filename)
        create_colorpalette(src_image, pred_image,filename)
