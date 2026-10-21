# Kết quả sửa test và đánh giá lại RF-DETR legacy

**Cập nhật tiếp theo:** kết quả legacy trong file này giữ nguyên. Core/train bốn module và pipeline lưu `_v1`, `_v2`… đã được triển khai, kiểm tra A100 smoke thành công; xem `HUONG_DAN_TRAIN_TEST_DQK_V1.md`. Chưa có kết quả test enhanced sau huấn luyện đầy đủ.

Ngày hoàn tất: 21/10/2026, giờ Việt Nam.

## 1. Kết luận

Đã hoàn tất hai việc: sửa/kiểm chứng mapping, checkpoint load, normalize, hậu xử lý và COCO metric; chạy lại checkpoint cũ dưới tên **RFDETR_DQK_Legacy** trên đủ **1.481 ảnh test, 32 lớp** bằng Modal A100.

**mAP50=98.74%, mAP50:95=88.00%, Precision=98.00%, Recall=97.52%, F1=97.76%.** Không huấn luyện hoặc fine-tune model trong lượt này.

Đây là RFDETRMedium legacy/baseline. Checkpoint không có bốn module DARM/UFFR/DAFD/MFA; không dùng kết quả này để tuyên bố cải tiến DQK có hiệu quả.

Kết quả đầy đủ: `../Model_Improvement/Results/RESULT_20261021_021016/`.

## 2. Checkpoint và dữ liệu đã dùng

- Checkpoint cloud: volume `rfdetr-runs`, `/rfdetr_dqk_enhanced/checkpoints/best.pth`.
- SHA256 checkpoint: `c2bdde40551a54c412656d4c5e0e1710201cd45aaa1d1543ba4d367fed8f1379`.
- Kiến trúc metadata: RFDETRMedium; RF-DETR 1.10.1; best epoch index 22.
- Test data: volume `food-detection-training`, `/datasets/completed-project-5/coco/test`.
- Annotation SHA256: `d7f69d3b3933acd390fae8b65cb577e9e07d7b9b96031ab39ab74cdbc9901376`.
- 1.481 ảnh được giải mã và suy luận thành công; GT có 3.911 bounding box, 32 lớp.
- Không train lại, không thay checkpoint, không tìm ngưỡng tốt nhất trên test.

## 3. Những gì đã sửa và xác minh

### Mapping và số lớp

Lấy tên/thứ tự lớp từ metadata training trong checkpoint, đối chiếu đủ 32 tên với dataset/config. Mapping được lưu trong JSON kết quả: label model 0..31 -> COCO category ID 1..32. Slot background của custom model là 32, được loại khỏi detection; parent category ID 0 của Roboflow không phải background slot này.

Mọi final/encoder classification head đều có 33 output; runtime logits trên ảnh validation có shape `[1,300,33]`. Không dùng mặc định 90 của pretrained COCO cho bài toán này. Nhãn lạ phải gây lỗi, không bị bỏ qua âm thầm.

### Nạp trọng số

Nhánh enhanced/manual cũ đã được thay bằng adapter RFDETRMedium legacy chuyên biệt, không còn load checkpoint baseline trực tiếp vào wrapper DQK bằng strict=False.

Sau khi native constructor tạo model, adapter thực hiện load **strict=True**, rồi so sánh chính xác từng tensor đã load với tensor checkpoint:

- **510 state entries** được xác minh.
- Missing keys: **0**.
- Unexpected keys: **0**.
- Mọi tensor đã load bằng tensor checkpoint: **true**.
- Runtime version phải bằng RF-DETR version trong checkpoint, ở đây là **1.10.1**.
- Checkpoint SHA256 không đúng cấu hình thì dừng trước inference.
- Enhanced architecture chưa được xác minh sẽ bị từ chối; không fallback sang baseline/pretrained để tiếp tục xuất metric.

### Normalize và hậu xử lý

Dùng `RFDETRMedium.predict` chuẩn phiên bản 1.10.1, truyền **threshold=0.0**. Không còn nhánh ToTensor+Resize thiếu normalize hoặc max một class/query thủ công.

Một ảnh validation được dùng để kiểm tra, trước khi chạy test:

- RGB, resize 640x640 với antialias=False và normalize được so sánh bằng forward hook với transform tường minh, khớp trong dung sai float.
- Mean: `[0.485,0.456,0.406]`; std: `[0.229,0.224,0.225]`.
- Scores/labels/boxes do API predict trả về được đối chiếu với native postprocessor trên chính output core đã capture: đạt.
- Native postprocessor chọn theo sigmoid trên query-class pairs; adapter loại background, chuyển mapping theo tên lớp, clip box về ảnh gốc và giữ tối đa 100 prediction/ảnh theo giao thức chung.

### COCO metric

Thiết lập cố định: bbox COCOeval, 640, batch=1, prediction threshold 0.0, COCO maxDets `[1,10,100]`, global cap 100/ảnh; P/R/F1 micro tại conf=0.55 và IoU=0.50. AP không bị tính chỉ trên prediction trên 0.55.

Đã chạy **10/10 semantic tests**, cả local và cloud: perfect predictions, empty predictions, duplicate FP, wrong class, empty-image FP, AP độc lập ngưỡng operating point, crowd ignore, từ chối đổi protocol, mapping/background/cap/clip và từ chối label lạ.

Sau inference, lấy **146.370 COCO predictions đã lưu**, tải lại annotation và chạy chính lớp `COCOBenchmarkEvaluator` trích từ file chung **không sửa đổi** tại `E:/CNTT/NCKH/Train-V2/TEST-COMMON/common_test.py`:

- P/R/F1, mAP50, mAP50:95, mAP75, AR100 bằng nhau ở độ chính xác xuất 4 chữ số thập phân.
- Toàn bộ metric per-class của **32 lớp** bằng nhau.
- TP/FP/FN và thiết lập operating point bằng nhau tuyệt đối.
- Hash annotation được kiểm tra lại trước phép đối chiếu.

Bằng chứng: `common_evaluator_parity.json` và `semantic_checks.json` trong thư mục kết quả. Hash source script chung được ghi riêng; không giả mạo hash evaluator hoặc khẳng định protocol ID hai script giống nhau.

## 4. Metric test chính thức của lượt chạy

- Precision: **0.9800 / 98.00%**.
- Recall: **0.9752 / 97.52%**.
- F1: **0.9776 / 97.76%**.
- mAP@50: **0.9874 / 98.74%**.
- mAP@50:95: **0.8800 / 88.00%**.
- mAP@75: **0.9560 / 95.60%**.
- AR@100: **0.9177 / 91.77%**.
- Operating TP=**3.814**, FP=**78**, FN=**97**, tại conf=0.55/IoU=0.50.
- Parameters: **33.60M**.
- Latency: **30.34 ms/ảnh**; FPS: **32.96**, A100, batch=1, 10 warmup, đo native API gồm preprocess/forward/postprocess nhưng không gồm đọc/giải mã ảnh từ đĩa. Model chưa optimize inference; đây không phải tốc độ cực đại của kiến trúc.
- GFLOPs: **không đo** trong lượt chẩn đoán; không lấy số từ profiler thiếu operator để điền vào báo cáo.

COCO AP_small=-1 nghĩa là không có GT hợp lệ trong nhóm small; annotation thực tế có 0 box area<32². Không diễn giải -1 thành độ chính xác âm hoặc 0%.

Các lớp có AP50:95 thấp nhất: chicken=0.7261, okra=0.7294, chickenleg=0.8131, onion=0.8227, corn=0.8272. Xem đủ 32 lớp trong JSON/CSV kết quả; không chỉnh test theo những lớp này.

## 5. Vì sao báo cáo cũ thấp

File lịch sử `RESULT_20261019_235139/result_RFDETR_DQK_Enhanced.json` ghi mAP50=0.0024 và mAP50:95=0.0018. Lượt mới đạt 0.9874 và 0.8800 trên cùng nội dung annotation test.

Các lỗi code cũ đã xác nhận có thể làm sai phép đánh giá:

1. Checkpoint core có tên key không có prefix `net.`, trong khi wrapper yêu cầu prefix; 0/510 key checkpoint có prefix phù hợp. Strict=False chỉ in cảnh báo khiến trọng số đã train có thể không được nạp vào mạng thực thi.
2. Đánh giá checkpoint RFDETRMedium bằng wrapper có DARM mới khởi tạo và module khác chưa được train không đại diện cho mạng đã lưu.
3. Nhánh manual thiếu ImageNet normalize và dùng max-class/query thay vì native top-k query-class pairs.
4. Test/training dùng khác phiên bản RF-DETR và thiếu provenance đủ để truy nguyên checkpoint load.

**Kết luận được hỗ trợ:** checkpoint legacy hiện tại có hiệu năng test tốt khi được nạp và suy luận đúng; điểm cũ rất thấp không nên dùng làm số đo hiệu năng của checkpoint này. Không cần train lại chỉ để khôi phục việc đánh giá legacy.

**Giới hạn kết luận:** file kết quả cũ không lưu checkpoint SHA256 hoặc source snapshot hoàn chỉnh, và lượt này sửa kết hợp nhiều lỗi, không thực hiện ablation từng lỗi. Vì vậy không khẳng định chính xác tỷ lệ tác động của từng lỗi hoặc chứng minh bytes checkpoint ngày 19/10 trùng bytes hiện tại. Đây không phải phép so sánh hiệu quả thuật toán mới với cũ.

Để có kết quả **DQK enhanced thật**, vẫn cần sửa luồng train/criterion/optimizer/EMA/checkpoint để bốn module tham gia, smoke test rồi huấn luyện/fine-tune mới theo kế hoạch đã viết. Lượt này chỉ xử lý hai hạng mục test được yêu cầu.

## 6. File thay đổi và cách chạy

- `Model_Improvement/src/testing/rfdetr_legacy_adapter.py`: native inference, strict checkpoint verification, mapping và pipeline parity.
- `Model_Improvement/src/testing/common_test.py`: dùng adapter mới, kiểm tra config/hash/số ảnh/lớp, output run riêng, raw predictions/provenance, commit đúng runs volume; hỗ trợ đường dẫn Modal `/root` và package.
- `Model_Improvement/configs/models_config.json`: chỉ bật legacy, dùng đúng volume/path, pin hash checkpoint/annotation; enhanced/baseline khác đều tắt.
- `Model_Improvement/src/testing/test_legacy_evaluation.py`: semantic checks.
- `Model_Improvement/src/testing/verify_common_metric_parity.py`: đối chiếu evaluator chung với prediction đã lưu, không khởi tạo app/model của script chung.
- `Model_Improvement/src/testing/README_LEGACY.md`: hướng dẫn chạy.

Chạy từ DQK:

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m modal run Model_Improvement/src/testing/common_test.py::evaluate_my_models
```

Run cloud thành công: https://modal.com/apps/duongquockiet0511/main/ap-4pUDP3zkbLcU9kRwohpMw4

Cloud output: `/runs/dqk_legacy_evaluation/run-20261020-191054-6493fc21` trên volume `rfdetr-runs` (tên cloud dùng UTC). Local output dùng giờ Việt Nam: `RESULT_20261021_021016`.

Một lượt cloud ban đầu lỗi đường dẫn trước inference đã được sửa; lượt được báo cáo ở đây hoàn tất thành công. Không sửa file test chung ngoài DQK, không ghi đè checkpoint hoặc kết quả lịch sử. Đã xác minh SHA256 hai file source local khớp source hash được ghi trong lượt chạy thành công.
