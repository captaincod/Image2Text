import cv2.cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
from skimage.measure import label, regionprops


def extract_features(image):
    features = []
    # Next, Previous, First_Child, Parent
    _, hierarchy = cv2.cv2.findContours(image, cv2.cv2.RETR_CCOMP, cv2.cv2.CHAIN_APPROX_SIMPLE)
    ext_cnt = 0
    int_cnt = 0
    for i in range(len(hierarchy[0])):
        if hierarchy[0][i][-1] == -1:
            ext_cnt += 1
        elif hierarchy[0][i][-1] == 0:
            int_cnt += 1
    features.extend([ext_cnt, int_cnt])
    labeled = label(image)
    region = regionprops(labeled)[0]
    filling_factor = region.area / region.bbox_area
    features.append(filling_factor)
    centroid = np.array(region.local_centroid) / np.array(region.image.shape)
    features.extend(centroid)
    features.append(region.eccentricity)
    return features


def distance(px1, px2):
    return ((px1[0] - px2[0]) ** 2 + (px1[1] - px2[1]) ** 2) ** 0.5


def image_to_text(image, knn):
    text = ""
    binary = image.copy()
    binary[binary > 0] = 1
    labeled = label(binary)
    regions = regionprops(labeled)
    sorted_regions = {}
    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox
        sorted_regions[min_col] = [region]
    sorted_regions = dict(sorted(sorted_regions.items()))
    prev_key = 0
    has_space = []
    for key in sorted_regions:
        if key - prev_key < 9:
            sorted_regions[key].append(sorted_regions[prev_key][0])
            sorted_regions[prev_key] = []
        else:
            if key - prev_key > 105:
                has_space.append(key)
            prev_key = key

    text = ""
    for key in sorted_regions:
        if not sorted_regions[key]:
            text += ' '
        elif len(sorted_regions[key]) == 2:
            text += 'i'
        else:
            if key in has_space:
                text += ' '
            region = sorted_regions[key][0]
            min_row, min_col, max_row, max_col = region.bbox
            test_symbol = extract_features(image[min_row:max_row, min_col:max_col])
            test_symbol = np.array(test_symbol, dtype="f4").reshape(1, 6)
            ret, results, neighbours, dist = knn.findNearest(test_symbol, 3)
            text += chr(int(ret))
            # plt.imshow(image[min_row:max_row, min_col:max_col])
            # plt.show()
    return text


train_dir = Path("out") / "train"
train_data = defaultdict(list)

for path in sorted(train_dir.glob("*")):
    if path.is_dir():
        for img_path in path.glob("*.png"):
            symbol = path.name[-1]
            gray = cv2.cv2.imread(str(img_path), 0)
            binary = gray.copy()
            binary[binary > 0] = 1
            train_data[symbol].append(binary)

features_array = []
responses = []
for i, symbol in enumerate(train_data):
    for img in train_data[symbol]:
        features = extract_features(img)
        features_array.append(features)
        responses.append(ord(symbol))

features_array = np.array(features_array, dtype="f4")
responses = np.array(responses)

knn = cv2.cv2.ml.KNearest_create()
knn.train(features_array, cv2.cv2.ml.ROW_SAMPLE, responses)

for img_number in range(0, 6):
    image = cv2.cv2.imread(f"out/{img_number}.png", cv2.cv2.IMREAD_GRAYSCALE)
    print(image_to_text(image, knn))
    plt.imshow(image)
    plt.show()
