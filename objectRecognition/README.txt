Các bước set-up Project2.2:
1. Mở cmd, trỏ tới thư mục coil-100-train-test
2. Tạo 2 thư mục con: train và test
3. Với mỗi thư mục, sử dụng câu lệnh "mkdir obj1 obj2 ... obj100" để tạo ra các thư mục con tương ứng với các class của ảnh. Có thể viết 1 chương trình python cơ bản in ra chuỗi obj1 obj2 ... obj100

Các bước thực hiện:
•	Tìm ra các keypoint và descriptor tương ứng cho từng ảnh
•	Phân cụm các descriptor sử dụng thuật toán k-means để hình thành các words
•	Xây dựng và chuẩn hóa tập biểu đồ BoW histogram biểu diễn tần suất xuất hiện của các từ trong túi từ trong mỗi hình ảnh
•	Sử dụng tập biểu đồ trong việc phân loại hình ảnh

