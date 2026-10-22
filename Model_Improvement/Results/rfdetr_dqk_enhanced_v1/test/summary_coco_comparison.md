# BẢNG TỔNG HỢP SO SÁNH CÁC MÔ HÌNH OBJECT DETECTION

- **Đề tài**: Nghiên cứu khoa học - Nhận diện nguyên liệu nấu ăn
- **Thời gian**: 21/10/2026 18:48:37

| Mô hình | Precision | Recall | F1 | mAP@50 | mAP@50-95 | mAP@75 | AR@100 | Patience | GFLOPs | Parameters | Latency | FPS | Trạng thái |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RF-DETR DQK enhanced (rfdetr_dqk_enhanced_v1) | 0.9769 | 0.9729 | 0.9749 | 0.9850 | 0.8792 | 0.9578 | 0.9167 | 10 | 57.38 | 40.29 | 36.32 | 27.53 | Hoàn tất đánh giá |

<!-- complexity-estimate -->
GFLOPs: **ước lượng**, input 1×3×640×640, eval FP32; 1 FMA = 1 FLOP theo fvcore. Bao gồm DARM/DAFD/UFFR/MFA và FFT ước lượng; không còn operator chưa có bộ đếm trong trace này. Không gồm resize ảnh, hậu xử lý detection và đọc đĩa. Xem `complexity_profile.json` để biết công thức và phạm vi; không so trực tiếp với bảng dùng 2 FLOP/MAC.
