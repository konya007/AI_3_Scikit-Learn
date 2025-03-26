import pandas as pd
from sklearn.datasets import load_iris
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

current_number_question = 0
def print_question():
    global current_number_question
    current_number_question += 1
    print("-"*50)
    print("Câu {0}".format(current_number_question))

# 1. INhập dữ liệu từ tập dữ liệu iris và in ra 5 dòng đầu tiên của dữ liệu.

# Theo lý thuyết, bộ dữ liệu iris chứa 150 mẫu từ ba loài hoa iris (0: setosa, 1: versicolor, 2: virginica) và bốn thuộc tính: chiều dài và chiều rộng của đài hoa và cánh hoa.
print_question()
iris = load_iris()
print("Print 5 first items of the data")
print(iris.data[:5])

# 2. Làm thế nào để biết mỗi mẫu thuộc loại hoa nào? Làm thế nào để biết sự tương ứng giữa các loài hoa và số?
print_question()

print("In ra 5 mẫu đầu tiên thuộc loại hoa nào?")
print(iris.target[:5])
print("Tên các loài hoa tương ứng: 0 = {0}, 1 = {1}, 2 = {2}".format(iris.target_names[0], iris.target_names[1], iris.target_names[2]))

# 3. Tạo biểu đồ phân tán hiển thị ba loài hoa khác nhau bằng ba màu khác nhau; Trục X sẽ biểu diễn chiều dài của đài hoa trong khi trục y sẽ biểu diễn chiều rộng của đài hoa.
print_question()

sepal_length = iris.data[:, 0]
sepal_width = iris.data[:, 1]
plt.scatter(sepal_length[iris.target == 0], sepal_width[iris.target == 0], c='red', label='setosa')
plt.scatter(sepal_length[iris.target == 1], sepal_width[iris.target == 1], c='blue', label='versicolor')
plt.scatter(sepal_length[iris.target == 2], sepal_width[iris.target == 2], c='green', label='virginica')
plt.legend(['setosa', 'versicolor', 'virginica'])
plt.xlabel('Chiều dài của đài hoa')
plt.ylabel('Chiều rộng của đài hoa')
plt.title('Phân tán của ba loài hoa với chú thích cho Setosa')
plt.show()

# 4.Dùng giảm chiều dữ liệu, ở đây sử dụng PCA, tạo ra một chiều mới (=3, gọi là thành phần chính).
print_question()

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
X = iris.data
y = iris.target
X_r = pca.fit_transform(X)
print("Hình dạng trước khi giảm chiều: ", X.shape)
print("Hình dạng sau khi giảm chiều: ", X_r.shape)
print("5 mẫu đầu tiên sau khi giảm chiều: ")
print(X_r[:5])

#5. Dùng k-nearest neighbor để phân loại nhóm mà mỗi loài hoa thuộc về. Đầu tiên, tạo một tập huấn luyện và tập kiểm tra; với 140 mẫu sẽ được sử dụng làm tập huấn luyện, và 10 mẫu còn lại sẽ được sử dụng làm tập kiểm tra.
print_question()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=10, random_state=33)
print("Kích thước tập huấn luyện: ", X_train.shape)
print("Kích thước tập kiểm tra: ", X_test.shape)

#6. Kế tiếp, áp dụng mô hình K-nearest neighbor, thử với K=5.
print_question()

knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Kết quả dự đoán: ", y_pred)

# 7. Cuối cùng bạn có thể so sánh kết quả dự đoán với kết quả thực tế chứa trong y_test.
print_question()

print("Kết quả thực tế: ", y_test)
print("Kết quả dự đoán: ", y_pred)
print("Độ chính xác: ", accuracy_score(y_test, y_pred))

# 8. Bây giờ, bạn có thể trực quan hóa tất cả điều này bằng cách sử dụng ranh giới quyết định trong không gian được biểu diễn bằng biểu đồ phân tán 2D của các đài hoa.
print_question()

X = iris.data[:, :2] # Chỉ lấy độ dài và chiều rộng của đài hoa
y = iris.target

# Tạo lưới tọa độ
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Huấn luyện mô hình
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# Dự đoán
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Vẽ đồ thị
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
plt.xlabel('Chiều dài của đài hoa')
plt.ylabel('Chiều rộng của đài hoa')
plt.title('Ranh giới quyết định của KNN')
plt.show()

# 9. Tải tập dữ liệu diabete. Để dự đoán mô hình, chúng ta sử dụng hồi quy tuyến tính.
print_question()

from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

# 10. Đầu tiên, bạn cần phải chia tập dữ liệu thành tập dữ liệu huấn luyện (bao gồm 422 bệnh nhân đầu tiên) và tập kiểm tra (20 bệnh nhân cuối cùng).
print_question()

X_train = diabetes.data[:-20]
X_test = diabetes.data[-20:]
y_train = diabetes.target[:-20]
y_test = diabetes.target[-20:]

print("Kích thước tập huấn luyện: ", X_train.shape)
print("Kích thước tập kiểm tra: ", X_test.shape)

# 11. Bây giờ, áp dụng tập huấn luyện để dự đoán mô hình?
print_question()

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# 12. Làm thế nào để lấy ra 10 hệ số b đã được tính toán sau khi mô hình được huấn luyện?
print_question()

print("Hệ số: ", linreg.coef_)

# 13. Nếu bạn áp dụng tập kiểm tra vào dự đoán hồi quy tuyến tính, bạn sẽ nhận được một loạt các mục tiêu để so sánh với giá trị thực tế quan sát được.
print_question()

y_pred = linreg.predict(X_test)

print("Kết quả dự đoán: ", y_pred)
print("Kết quả thực tế: ", y_test)

#14. Làm thế nào để kiểm tra độ chính xác của dự đoán?
print_question()
from sklearn.metrics import mean_squared_error, r2_score

# Tính r2
r2 = r2_score(y_test, y_pred)
print("R**2: ", r2)
mse = mean_squared_error(y_test, y_pred)
print("Lỗi bình phương trung bình bình phương (MSE): ", mse) 

# 15. Bây giờ, bạn sẽ bắt đầu với hồi quy tuyến tính với một yếu tố sinh lý duy nhất, ví dụ, bạn có thể bắt đầu với độ tuổi.
print_question()

X_train_age = X_train[:, [0]]
X_test_age = X_test[:, [0]]

lr_age = LinearRegression()
lr_age.fit(X_train_age, y_train)

y_pred_age = lr_age.predict(X_test_age)
print("Kết quả dự đoán với độ tuổi: ", y_pred_age)
print("Kết quả thực tế với độ tuổi: ", y_test)

# 16. Thực tế, bạn có 10 yếu tố sinh lý trong tập dữ liệu tiểu đường. Do đó, để có một bức tranh hoàn chỉnh hơn về toàn bộ tập huấn luyện, bạn có thể thực hiện hồi quy tuyến tính cho từng yếu tố sinh lý, tạo ra 10 mô hình và xem kết quả cho từng mô hình thông qua biểu đồ tuyến tính.
print_question()

# Lặp qua 10 đặc trưng 
for i in range(10):
    X_train_feature = X_train[:, [i]]
    X_test_feature = X_test[:, [i]]

    lr_feature = LinearRegression()
    lr_feature.fit(X_train_feature, y_train)

    y_pred_feature = lr_feature.predict(X_test_feature)


    plt.figure(figsize=(10, 6))
    plt.scatter(X_test_feature, y_test, color='blue', label='Giá trị thực tế')
    plt.plot(X_test_feature, y_pred_feature, color='red', label='Giá trị dự đoán')
    plt.xlabel('Yếu tố sinh lý {0}'.format(i+1))
    plt.ylabel('Tiến trình bệnh')
    plt.title('Hồi quy tuyến tính cho yếu tố sinh lý {0}'.format(i+1))
    plt.legend()
    plt.show()

# 17. Sử dụng skicit-learn tải xuống tập dữ liệu ung thư vú của đại học Winconsin. In ra khóa của từ điển này.
print_question()

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("Các khóa trong từ điển: ", cancer.keys())

# 18. Kiểm tra hình dạng của dữ liệu. Đếm số lượng khối u “benign” và “malignant”.
print_question()

print("Kích thước dữ liệu: ", cancer.data.shape)

# Chuyển target thành Series để dễ dàng đếm
target_series = pd.Series(cancer.target)
benign_count = target_series.value_counts()[0]
malignant_count = target_series.value_counts()[1]
print("Số lượng khối u lành tính: ", benign_count)
print("Số lượng khối u ác tính: ", malignant_count)

# 19. Chia dữ liệu thành tập huấn luyện và tập kiểm tra. Sau đó, đánh giá hiệu suất của tập huấn luyện và tập kiểm tra với số lượng hàng xóm khác nhau (từ 1 đến 10). Tạo một hình ảnh trực quan hóa.
print_question()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=42)
train_score = []
test_score = []
for k in range(1, 11):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_score.append(knn.score(X_train, y_train))
    test_score.append(knn.score(X_test, y_test))

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), train_score, label='Độ chính xác tập huấn luyện', marker='o')
plt.plot(range(1, 11), test_score, label='Độ chính xác tập kiểm tra', marker='o')
plt.xlabel('Số lượng hàng xóm (K)')
plt.ylabel('Độ chính xác')
plt.title('Độ chính xác của KNN với các số lượng hàng xóm khác nhau')
plt.legend()
plt.grid()
plt.show()

# 20. Tải xuống thư viện mglearn. Sử dụng tập dữ liệu make_forge. So sánh hồi quy logistic và Linear SVC.
print_question()

import mglearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

X, y = mglearn.datasets.make_forge()

# Huấn luyện mô hình hồi quy logistic 
logreg = LogisticRegression()
logreg.fit(X, y)
print("Độ chính xác hồi quy logistic: ", logreg.score(X, y))

# Huấn luyện mô hình Linear SVC
svc = LinearSVC()
svc.fit(X, y)
print("Độ chính xác Linear SVC: ", svc.score(X, y))

# 21. Chúng ta sẽ áp dụng SVM để nhận diện hình ảnh. Tập huấn luyện của chúng ta sẽ là một nhóm hình ảnh có nhãn của khuôn mặt con người. Bây giờ hãy bắt đầu bằng cách nhập và in mô tả của nó.
print_question()

from sklearn.datasets import fetch_lfw_people
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
print("Mô tả tập dữ liệu: ", lfw_people.DESCR)
print("Kích thước hình ảnh: ", lfw_people.images.shape)
print("Số lượng hình ảnh: ", lfw_people.images.shape[0])
print("Số lượng người: ", len(lfw_people.target_names))
print("Tên người: ", lfw_people.target_names)

# 22. Nhìn vào nội dung của đối tượng faces, chúng ta có các thuộc tính sau: images, data và target.
# images: hình ảnh khuôn mặt
# data: dữ liệu hình ảnh đã được biến đổi thành một mảng 2 chiều
# target: nhãn của hình ảnh
print_question()

print("Hình ảnh khuôn mặt: ", lfw_people.images[0])
print("Dữ liệu hình ảnh: ", lfw_people.data[0])
print("Nhãn của hình ảnh: ", lfw_people.target[0])
print("Tên người: ", lfw_people.target_names[lfw_people.target[0]])
print("Kích thước hình ảnh: ", lfw_people.images[0].shape)

# 23. Trước khi học, hãy vẽ một số khuôn mặt. Vui lòng định nghĩa một hàm trợ giúp.
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Vẽ một số hình ảnh với tiêu đề"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()

plot_gallery(lfw_people.images, lfw_people.target_names[lfw_people.target], h=50, w=37)


# 24. Cài đặt SVC có các tham số quan trọng khác nhau; có lẽ tham số quan trọng nhất là kernel. Để bắt đầu, chúng ta sẽ sử dụng kernel đơn giản nhất, đó là kernel tuyến tính.
print_question()

from sklearn.svm import SVC
svc = SVC(kernel='linear', class_weight='balanced')

# 25. Trước khi tiếp tục, chúng ta sẽ chia tập dữ liệu thành tập huấn luyện và tập kiểm tra.
print_question()

X_train, X_test, y_train, y_test = train_test_split(lfw_people.data, lfw_people.target, test_size=0.25, random_state=42)
print("Kích thước tập huấn luyện: ", X_train.shape)
print("Kích thước tập kiểm tra: ", X_test.shape)
print("Số lượng hình ảnh trong tập huấn luyện: ", len(X_train))
print("Số lượng hình ảnh trong tập kiểm tra: ", len(X_test))

# 26. Và chúng ta sẽ định nghĩa một hàm để đánh giá K-fold cross-validation.
print_question()

from sklearn.model_selection import cross_val_score
def evaluate_model(model, X, y, cv=5):
    """Đánh giá mô hình bằng K-fold cross-validation"""
    scores = cross_val_score(model, X, y, cv=cv)
    print("Điểm trung bình: ", scores.mean())
    print("Độ lệch chuẩn: ", scores.std())
    return scores.mean(), scores.std()

# 27. Chúng ta cũng sẽ định nghĩa một hàm để thực hiện việc huấn luyện trên tập huấn luyện và đánh giá hiệu suất trên tập kiểm tra.
print_question()

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """Huấn luyện mô hình và đánh giá hiệu suất"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Độ chính xác trên tập kiểm tra: ", accuracy)
    return accuracy

# 28. Nếu chúng ta huấn luyện và đánh giá, bộ phân loại thực hiện hoạt động mà gần như không có lỗi. Kiểm tra điều này bằng cách sử dụng hàm evaluate_model.
print_question()

try:
    evaluate_model(svc, X_train, y_train)
    train_and_evaluate(svc, X_train, y_train, X_test, y_test)
except Exception as e:
    print("Lỗi: ", e)
    print("Có thể do không đủ bộ nhớ để chạy mô hình này.")

# ==============================================================

glasses = [
    (10, 19), (30, 32), (37, 38), (50, 59), (63, 64),
    (69, 69), (120, 121), (124, 129), (130, 139), (160, 161),
    (164, 169), (180, 182), (185, 185), (189, 189), (190, 192),
    (194, 194), (196, 199), (260, 269), (270, 279), (300, 309),
    (330, 339), (358, 359), (360, 369)
]

# 29. Sau đó, chúng ta sẽ định nghĩa một hàm từ các đoạn đó trả về một mảng mục tiêu mới đánh dấu bằng 1 cho các khuôn mặt có kính và 0 cho các khuôn mặt không có kính (các lớp mục tiêu mới của chúng ta).
print_question()

def create_glasses_target(dataset_length, glasses_ranges):
    has_glasses = np.zeros(dataset_length, dtype=int)
    
    # Mark faces that have glasses (1)
    for start, end in glasses_ranges:
        has_glasses[start:(end+1)] = 1
    
    return has_glasses

# Create the glasses target array
n_samples = len(lfw_people.target)
glasses_target = create_glasses_target(n_samples, glasses)

# Print some information about the new target array
print(f"Tổng số mẫu: {n_samples}")
print(f"Số lượng khuôn mặt có kính: {np.sum(glasses_target)}")
print(f"Số lượng khuôn mặt không có kính: {n_samples - np.sum(glasses_target)}")
print(f"Phần trăm khuôn mặt có kính: {100 * np.sum(glasses_target) / n_samples:.2f}%")

# 30. Vì vậy, chúng ta phải thực hiện việc chia lại tập huấn luyện/tập kiểm tra. Bây giờ hãy tạo một bộ phân loại SVC mới và huấn luyện nó với vector mục mới.
print_question()

svc_glasses = SVC(kernel='linear')

X_train_glasses, X_test_glasses, y_train_glasses, y_test_glasses = train_test_split(
    lfw_people.data, glasses_target, test_size=0.25, random_state=0
)

# 31. Kiểm tra hiệu suất với cross-validation. Chúng ta đạt được độ chính xác trung bình là 0.967 với cross-validation nếu chúng ta đánh giá trên tập kiểm tra của mình.
print_question()

# Evaluate the model with cross-validation
evaluate_model(svc_glasses, X_train_glasses, y_train_glasses, 5)
train_and_evaluate(svc_glasses, X_train_glasses, y_train_glasses, X_test_glasses, y_test_glasses)

# 32. Chúng ta sẽ tách tất cả các hình ảnh của cùng một người, đôi khi đeo kính và đôi khi không. Chúng ta cũng sẽ tách tất cả các hình ảnh của cùng một người, những hình ảnh có chỉ số từ 30 đến 39, huấn luyện bằng cách sử dụng các thể hiện còn lại và đánh giá trên tập hợp 10 thể hiện mới của chúng ta. Với thí nghiệm này, chúng ta sẽ cố gắng loại bỏ thực tế rằng nó đang ghi nhớ khuôn mặt, không phải các đặc điểm liên quan đến kính.
print_question()

test_indices = list(range(30, 40))
train_indices = [i for i in range(n_samples) if i not in test_indices]

# Create training and test sets
X_train_person = lfw_people.data[train_indices]
X_test_person = lfw_people.data[test_indices]
y_train_person = glasses_target[train_indices]
y_test_person = glasses_target[test_indices]

# Create and train a new SVC model
svc_person = SVC(kernel='linear')
svc_person.fit(X_train_person, y_train_person)

# Evaluate performance
y_pred_person = svc_person.predict(X_test_person)

accuracy_person = accuracy_score(y_test_person, y_pred_person)
print(f"Nhãn thực tế: {y_test_person}")
print(f"Dự đoán: {y_pred_person}")
print(f"Độ chính xác: {accuracy_person:.4f}")

# 33. Từ 10 hình ảnh, chỉ có một lỗi, vẫn là kết quả khá tốt, hãy kiểm tra xem cái nào đã bị phân loại sai. Đầu tiên, chúng ta phải định hình lại dữ liệu từ các mảng thành các ma trận 64 x 64. Sau đó vẽ với hàm print_faces của chúng tôi.
print_question()

def print_faces(images, labels, n_images=10):
    """Print faces with their classification labels
    Green border for correct classification, red for incorrect"""
    plt.figure(figsize=(15, 8))
    for i in range(min(n_images, len(images))):
        plt.subplot(2, 5, i + 1)
        # Determine border color (green for correct, red for incorrect)
        border_color = 'green' if labels[i] == y_test_person[i] else 'red'
        
        # Add colored border based on classification result
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.title(f"Predicted: {labels[i]}\nActual: {y_test_person[i]}")
        plt.xticks(())
        plt.yticks(())
        plt.gca().spines['bottom'].set_color(border_color)
        plt.gca().spines['top'].set_color(border_color) 
        plt.gca().spines['right'].set_color(border_color)
        plt.gca().spines['left'].set_color(border_color)
        plt.gca().spines['bottom'].set_linewidth(5)
        plt.gca().spines['top'].set_linewidth(5) 
        plt.gca().spines['right'].set_linewidth(5)
        plt.gca().spines['left'].set_linewidth(5)
    plt.tight_layout()
    plt.show()

# Reshape the test images to their original dimensions
# The images appear to be 50x37 based on your earlier code
test_images = [img.reshape(50, 37) for img in X_test_person]
eval_faces = [img.reshape(50, 37) for img in X_test_person] 

# Plot the results
print_faces(test_images, y_pred_person)

# Print misclassified image indices
misclassified = [i for i in range(len(y_test_person)) if y_test_person[i] != y_pred_person[i]]
print(f"Phân loại sai ở các chỉ số: {misclassified}")


# 34. Tập dữ liệu của chúng ta có thể thu được bằng cách import hàm fetch_20newsgroups từ module sklearn.datasets. Chúng ta cần chỉ định liệu chúng ta muốn import một phần hay toàn bộ tập hợp các mẫu (chúng ta sẽ import tất cả chúng).
print_question()

from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(subset='all')
print("Số lượng mẫu: ", len(newsgroups.data))
print("Số lượng lớp: ", len(newsgroups.target_names))
print("Tên các lớp: ", newsgroups.target_names)
print("Nhãn của các lớp: ", newsgroups.target[:10])

# 35. If we look at the properties of the dataset, we will find that we have the usual ones: DESCR, data, target, and target_names 
print_question()

print("Mô tả tập dữ liệu: ", newsgroups.DESCR)
print("Dữ liệu: ", newsgroups.data[:5])
print("Nhãn: ", newsgroups.target[:5])  
print(type(newsgroups.data))
print(type(newsgroups.target))
print(type(newsgroups.target_names))
print("Tên các lớp: ", newsgroups.target_names) 

# 36. Tiền xử lý dữ liệu: Trước khi bắt đầu chuyển đổi, chúng ta sẽ phải phân chia dữ liệu thành tập huấn luyện và tập kiểm tra. Dữ liệu đã được tải theo thứ tự ngẫu nhiên, vì vậy chúng ta chỉ cần chia dữ liệu thành 75% cho huấn luyện và 25% còn lại cho kiểm tra.
print_question()

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train_news, X_test_news, y_train_news, y_test_news = train_test_split(
    newsgroups.data, newsgroups.target, test_size=0.25, random_state=42
)

print(f"Số lượng mẫu huấn luyện: {len(X_train_news)}")
print(f"Số lượng mẫu kiểm tra: {len(X_test_news)}")

# 37. Nếu bạn nhìn vào module sklearn.feature_extraction.text, bạn sẽ thấy ba lớp khác nhau có thể chuyển đổi văn bản thành đặc trưng số: CountVectorizer, HashingVectorizer và TfidfVectorizer. Chúng ta sẽ tạo ba bộ phân loại khác nhau bằng cách kết hợp MultinomialNB với ba vectorizer văn bản khác nhau vừa đề cập và so sánh cái nào hoạt động tốt hơn bằng cách sử dụng các tham số mặc định.
print_question()

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Tạo ba pipeline khác nhau với ba vectorizer khác nhau
count_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB())
])

# Thêm non_negative=True vào HashingVectorizer
hash_clf = Pipeline([
    ('vect', HashingVectorizer(alternate_sign=False)),
    ('clf', MultinomialNB())
])

tfidf_clf = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

print("Đã tạo ba bộ phân loại với ba vectorizer khác nhau.")

# 38. Chúng ta sẽ định nghĩa một hàm lấy một bộ phân loại và thực hiện K-fold cross-validation trên các giá trị X và y được chỉ định.
print_question()

from sklearn.model_selection import cross_val_score
import numpy as np

def perform_cv(clf, X, y, cv=5):
    """Thực hiện K-fold cross-validation và trả về điểm trung bình và độ lệch chuẩn"""
    scores = cross_val_score(clf, X, y, cv=cv)
    print(f"Độ chính xác trung bình: {scores.mean():.3f} (±{scores.std():.3f})")
    return scores.mean(), scores.std()

# 39. Sau đó, chúng ta sẽ thực hiện five-fold cross-validation bằng cách sử dụng từng bộ phân loại.
print_question()

print("Five-fold cross-validation với CountVectorizer + MultinomialNB:")
perform_cv(count_clf, X_train_news, y_train_news, cv=5)

print("\nFive-fold cross-validation với HashingVectorizer (non_negative=True) + MultinomialNB:")
perform_cv(hash_clf, X_train_news, y_train_news, cv=5)

print("\nFive-fold cross-validation với TfidfVectorizer + MultinomialNB:")
perform_cv(tfidf_clf, X_train_news, y_train_news, cv=5)