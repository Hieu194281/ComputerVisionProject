import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import kmeans, vq
import cv2
import numpy as np
import os
import time

# Step 1: Tìm keypoints và descriptors cho ảnh

train_path = 'coil-100-train-test/train'
classes_name = os.listdir(train_path)  # lay het  file trong thư mục

# list chứa đường dẫn tới các ảnh và class tương ứng
image_paths = []
image_classes = []
class_id = 0


def imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]


for class_name in classes_name:
    dir = os.path.join(train_path, class_name)
    class_path = imlist(dir)
    image_paths += class_path
    image_classes += [class_id] * len(class_path)
    class_id += 1

# list chứa các descriptor của ảnh
des_list = []
# Các thuật toán tìm keypoints
sift = cv2.SIFT_create(128)  # sift
# brisk = cv2.BRISK_create(30)
# orb = cv2.ORB_create()

# Đọc các ảnh và áp dụng sift lên ảnh
t1 = time.time()
for image_path in image_paths:
    im = cv2.imread(image_path)
    # im = cv2.resize(im, (150,150))
    # đầu vào là ảnh, đầu ra là các keypoint và các descriptors , các des là 1 vector 128 phan tu
    kpts, des = sift.detectAndCompute(im, None)
    des_list.append((image_path, des))
t2 = time.time()
print("Done feature extraction in %d seconds" % (t2 - t1))

# Step 2. Phân cụm các descriptor

# Nhóm các descriptor lại để phân cụm
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    if descriptor is not None:
        descriptors = np.vstack((descriptors, descriptor))
descriptors_float = descriptors.astype(float)  # ép kiểu sang float

# phân cụm các descriptor

# phân thành k cụm, giá trị voc trả về là các centroids
k = 100
t3 = time.time()
# k là so cum, 1 là true kiem tra ma tran dau vao co chưa gia tri huu
voc, variance = kmeans(descriptors_float, k, 1)
t4 = time.time()
print("Done clustering in %d seconds" % (t4 - t3))

# Step 3. Xây dựng và chuẩn hóa tập biểu đồ BoW histogram

im_features = np.zeros((len(image_paths), k), "float32")
# im_features[i][j]: số lượng cụm thứ j xuất hiện ở ảnh thứ i
for i in range(len(image_paths)):
    if des_list[i][1] is not None:
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            im_features[i][w] += 1

# chuẩn hóa histogram

stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

# Step 4. Sử dụng SVM để nhận dạng/phân loại

param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', '']}
t5 = time.time()
grid = GridSearchCV(SVC(), param_grid, refit=True,
                    verbose=3, scoring='accuracy', n_jobs=-1)
grid.fit(im_features, np.array(image_classes))
t6 = time.time()
print("Done classify in %d seconds" % (t6 - t5))

# in ra bộ tham số tốt nhât
print(grid.best_params_)
# in ra độ chính xác tốt nhất
print(grid.best_score_)

# lưu lại mô hình

joblib.dump((grid.best_estimator_, classes_name, stdSlr, k, voc),
            "sift_10011.pkl", compress=3)
