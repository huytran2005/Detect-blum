from ultralytics import YOLO
from cv2 import imread,waitKey,destroyAllWindows,imshow,resize

# Tải mô hình YOLOv8
model = YOLO('model.pt')  # Thay 'model.pt' bằng mô hình bạn muốn sử dụng, như 'yolov8n.pt'

# Đọc hình ảnh đầu vào
img_path = 'Screenshot_2024-08-23-13-55-05-44_948cd9899890cbd5c2798760b2b95377.jpg'
img = imread(img_path)
img=resize(img,(400,500))
# Chạy mô hình YOLO trên hình ảnh
results = model(img)

# Hiển thị kết quả trên hình ảnh
for result in results:
    p=result.plot()  # plot() sẽ vẽ các bounding boxes lên hình ảnh

    imshow("YOLOv8 Detection", p)
    waitKey(0)  # Đợi người dùng nhấn phím bất kỳ để đóng cửa sổ

# Đóng tất cả các cửa sổ hiển thị
destroyAllWindows()
# Hoặc nếu bạn muốn lưu kết quả:
# results.save('output_folder')

# Trích xuất các bounding box và trả về tọa độ
coordinates = []
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]
        class_id = box.cls[0]
        coordinates.append({
            'x1': int(x1),
            'y1': int(y1),
            'x2': int(x2),
            'y2': int(y2),
            'confidence': float(confidence),
            'class_id': int(class_id)
        })

# In ra tọa độ
print(coordinates)
