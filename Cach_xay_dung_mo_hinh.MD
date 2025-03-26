# Các bước xây dựng mô hình

## 1. Xác định vấn đề và mục tiêu

- Xác định rõ gàng bài toán mà mô hình AI cần giải quyết (phân loại, dự đoán, nhận diện,...)
- Xác định mục tiêu cụ thể mà mô hình cần đạt được (độ chính xác, độ nhạy, độ đặc hiệu,...)

## 2. Thu thập và chuẩn bị dữ liệu

- **Thu thập dữ liệu:** Tìm nguồn dữ liệu phù hợp
- **Làm sạch dữ liệu (Tiền xử lý):** Loại bỏ những dữ liệu trùng lặp, xử lý các giá tị thiếu, chuẩn hóa định dạng.
- **Gắn nhãn dữ liệu (nếu cần):** Đặt biệt với các bài toán học có giám sát (supervised learning)
- **Chia dữ liệu:**: Thường chia thành 3 tập: Tập huấn luyện (Training set), Tập kiểm tra (Validation set) và Tập test (Test set). *Ví dụ: 70% - 20% - 10%*

## 3. Lựa chọn mô hình

- **Chọn mô hình phù hợp:** Dựa vào bài toán cụ thể để chọn mô hình phù hợp:

    - **Phân loại:** Logistic Regression, Decision Tree, Random Forest, SVM, CNN, RNN, Transformer,...
    - **Dự đoán:** Linear Regression, Polynomial Regression, SVR, MLP, LSTM, GRU, Transformer,...
    - **Nhận diện:** CNN, R-CNN, Fast R-CNN, Faster R-CNN, YOLO, SSD,...

- Nếu chưa chắc chắn, có thể thử nghiệm nhiều mô hình để so sánh kết quả.

## 4. Huấn luyện mô hình

- **Sử dụng tập huấn luyện:** Đưa dữ liệu vào mô hình để huấn luyện để 'dạy' mô hình nhận biết dữ liệu.
- **Điều chỉnh các siêu tham số (Hyperparameters)** như learning rate, batch size, số lượng epoch,...
- **Theo dõi hiệu suất** trên tập kiểm tra để tránh overfitting (quá khớp) hoặc underfitting (không khớp).

## 5. Đánh giá mô hình 
- **Chọn chỉ số đánh giá:**
    - **Độ chính xác (Accuracy):** Tỉ lệ dự đoán đúng trên tổng số dự đoán.
    - **Precision, Recall, F1-score:** Dùng trong bài toán phân loại không cân bằng.
    - **RMSE, MAE, R2-score:** Dùng trong bài toán hồi quy.
    - **AUC-ROC, AUC-PR:** Đo lường khả năng phân loại của mô hình.
-**Sử dụng tập thử nghiệm (test set):** Đánh giá hiệu suất cuối cùng của mô hình, đảm bảo không dùng tập này trong quá trình huấn luyện.

## 6. Tinh chỉnh và tối ưu hóa mô hình

- Dựa trên kết quả đánh giá điều chỉnh mô hình.
    - Thu thập thêm dữ liệu nếu cần.
    - Thay đổi kiến trúc mô hình, hoặc thử nghiệm  thuật toán khác.
- Kiểm tra lại trên tập dữ liệu thử nghiệm sau khi điều chỉnh.

## 7. Triển khai mô hình

- Đưa mô hình vào ứng dụng thực tế.
- Theo dõi hiệu suất tỏng môi trường thực tế, và cập nhật lại mô hình nếu cần thiết.