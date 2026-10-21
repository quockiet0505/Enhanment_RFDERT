# Chạy train cải tiến DQK rồi tự test và tải kết quả

## File sử dụng

Entry point mới: `Model_Improvement/modal_app/modal_dqk_train_test.py`.

Luồng chạy: cấp tên run mới -> train đủ bốn module -> chọn winner validation regular/EMA của RF-DETR -> xuất `best.pth` đủ metadata -> tạo lại model và load strict -> kiểm tra preprocessing/postprocessing trên validation -> test đúng một model enhanced trên 1.481 ảnh -> commit volume -> tải báo cáo về máy.

File cũ `modal_train_dqk_a100.py` chuyển tiếp sang pipeline mới, không còn ghi vào thư mục train cũ. Cấu hình test legacy độc lập vẫn được giữ để đối chiếu; pipeline mới tự tạo cấu hình test enhanced với đúng best của chính run vừa train.

## Lệnh train chính thức

Chạy PowerShell từ `E:/CNTT/NCKH/Train_CaiTien/DQK`:

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m modal run --detach Model_Improvement/modal_app/modal_dqk_train_test.py
```

Mặc định tối đa 80 epoch, early stopping patience=10, seed=42, 32 lớp, resolution 640, BF16, batch=16, grad accumulation=1. Cấu hình tại `Model_Improvement/configs/rfdetr_dqk.yaml`.

`--detach` giữ app cloud hoạt động nếu client bị ngắt. Khi client còn kết nối đến cuối, báo cáo tự tải về. Nếu bạn tắt terminal/máy, kết quả vẫn ở volume; tải lại bằng lệnh fetch phía dưới. Không thể tự ghi file vào máy đang tắt.

Muốn tải cả best về máy sau khi chạy xong:

```powershell
python -m modal run --detach Model_Improvement/modal_app/modal_dqk_train_test.py --download-best
```

## Cách đặt tên và vị trí lưu

Volume giữ nguyên: **rfdetr-runs**, mount tại `/runs`.

Run mới dùng tên chưa tồn tại:

```text
/runs/rfdetr_dqk_enhanced_v1/
    run_manifest.json
    checkpoints/
        best.pth
        checkpoint_best_total.pth
        checkpoint_best_ema.pth
        last.ckpt
        metrics.csv
        dqk_effective_config.json
        training_audit.json
    test/
        requested_config.json
        result_RFDETR_DQK_Enhanced.json
        predictions_RFDETR_DQK_Enhanced.json.gz
        summary_coco_comparison.csv
        all_models_per_class_comparison.csv
        ...
```

Nếu `_v1` đã tồn tại hoặc đã được một process giữ chỗ, lượt mới chọn `_v2`, `_v3`… Claim nguyên tử trong Modal Dict bảo vệ cả trường hợp hai process khởi chạy cùng lúc. Tên run lỗi cũng được giữ để tra cứu, không tái sử dụng tự động.

Đường dẫn best mới ví dụ: `/runs/rfdetr_dqk_enhanced_v1/checkpoints/best.pth`. Thư mục cũ `/runs/rfdetr_dqk_enhanced/checkpoints/` và checkpoint legacy không bị ghi đè.

Local mặc định nhận báo cáo, CSV, JSON, ảnh, prediction nén và manifest tại:

```text
Model_Improvement/Results/rfdetr_dqk_enhanced_v1/
```

Trọng số và full resume checkpoint vẫn nằm trên Modal. Chỉ `best.pth` được tải thêm khi có `--download-best`; không tự tải toàn bộ các checkpoint lớn.

## Tải lại kết quả đã lưu

Ví dụ run đã hoàn tất có tên `rfdetr_dqk_enhanced_v1`:

```powershell
python -m modal run Model_Improvement/modal_app/modal_dqk_train_test.py --fetch-run rfdetr_dqk_enhanced_v1
```

Tải kèm best:

```powershell
python -m modal run Model_Improvement/modal_app/modal_dqk_train_test.py --fetch-run rfdetr_dqk_enhanced_v1 --download-best
```

Fetch chỉ đọc volume, không train/test lại, không cấp thêm phiên bản. Có thể fetch cả run lỗi để đọc `run_manifest.json`. Trạng thái tải thành công không biến một run failed thành complete.

## Smoke test

```powershell
python -m modal run Model_Improvement/modal_app/modal_dqk_train_test.py --smoke
```

Smoke dùng `rfdetr_dqk_smoke_v1`, `_v2`… riêng, không chiếm tên `_v1` của train chính thức. Chỉ train 3 batch với batch=1, validate 1 batch, xuất/nạp lại best và kiểm tra inference trên validation. Nó không chạy test 1.481 ảnh và không tạo metric test khoa học. Không dùng điểm validation smoke để báo cáo hiệu quả cải tiến.

## Resume có chủ đích

Mặc định mỗi lần chạy là một phiên bản mới, không tự resume baseline hoặc run cũ.

Nếu một run thất bại bình thường, đã ngừng hoạt động và có `checkpoints/last.ckpt`, có thể tiếp tục chính phiên bản đó:

```powershell
python -m modal run --detach Model_Improvement/modal_app/modal_dqk_train_test.py --resume-run rfdetr_dqk_enhanced_v1
```

Resume phải giữ nguyên source/config và smoke/full mode. Đây là thao tác ghi tiếp vào run được chỉ định rõ, khác với chạy mới. Một active claim ngăn hai client cùng ghi vào phiên bản đó; hard kill có thể để claim cũ, lúc đó chương trình từ chối resume để tránh ghi đồng thời. Cần xác minh app cũ đã dừng trước khi xử lý claim; không có cơ chế tự phá khóa. Run mới vẫn được cấp phiên bản tiếp theo bình thường.

## Những phần train/test đã được sửa

- Core giữ decoder/matcher/API LWDETR chuẩn. DARM chạy trên RGB sau khi hoàn nguyên normalize, rồi normalize lại; padding được bảo toàn.
- Bốn backbone taps có DAFD rồi UFFR; projector P4 của Medium có một MFA. Không giả định ba mức feature và không repeat/cắt kênh để né lỗi.
- UFFR dùng FFT bin đúng chẵn/lẻ; Gaussian là biến thể theo phân công. Đặt `uffr_mask: binary` để làm ablation theo mô tả mask nhị phân của bài fmars.
- DAFD có prior centers/widths đúng, FiLM gamma=1/beta=0 và gate=0.8 lúc đầu, fusion zero-init, residual 0.15; regularizer trung bình các gate/taps quanh 0.8.
- MFA có nhánh depthwise 5/7/9 và spatial gate một kênh. Thứ tự kết hợp bốn kỹ thuật là thiết kế DQK, không tự nhận là tái hiện nguyên một mô hình trong một bài báo.
- DARM bắt đầu identity; TV weight=1e-4 là lựa chọn thực nghiệm được ghi rõ trong YAML, sparsity weight=1e-5 theo DAFD. Cả hai thực sự được cộng vào loss train và log; không cộng vào metric test.
- Trainer dựng core cải tiến trước optimizer/EMA; không gọi sai `nn.Module.train(**training_kwargs)`, không fallback baseline.
- Optimizer giữ grouping/layer-wise LR native, thêm đầy đủ tham số DQK. Audit kiểm tra không trùng/thiếu tham số và gradient DQK không None/NaN/Inf.
- Best chọn theo validation của trainer, không theo test. Export khôi phục model_config và epoch thật của winner mà `strip_checkpoint` đã bỏ; kiểm tra tensor export đúng winner.
- Adapter enhanced tái tạo kiến trúc từ checkpoint notes, load strict, kiểm tra tensor/mapping, dùng native normalize/postprocess; evaluator COCO giữ ngưỡng/giao thức test đã kiểm chứng.

Seed/workers/AMP/weight decay/loss weights được áp dụng và lưu effective config. Bản này cố định train ở 640 (`multi_scale: false`), khác run legacy có multi-scale; muốn so sánh nhân quả hiệu quả module cần baseline/ablation huấn luyện cùng protocol. Không đảm bảo cứ thêm module là mAP phải tăng.

## Phạm vi kiểm chứng

Các kiểm tra CPU nằm ở `src/testing/test_dqk_pipeline.py`; COCO/mapping ở `test_legacy_evaluation.py`. Smoke chỉ chứng minh luồng ngắn và khả năng nạp checkpoint; không thay thế kiểm tra VRAM ở batch=16 hoặc run đủ 80 epoch. Lượt train chính thức sẽ tự test khi hoàn tất hoặc early stop thành công; nếu train lỗi thì không chạy test và manifest ghi failed.

### Kết quả kiểm chứng thực tế ngày 21/10/2026

- 9/9 kiểm tra CPU của module FFT, prior/gate, gradient, export winner và cấp phiên bản: đạt.
- 10/10 kiểm tra COCO/mapping/protocol: đạt.
- A100 smoke thành công: **rfdetr_dqk_smoke_v4**, app https://modal.com/apps/duongquockiet0511/main/ap-SyXrW1vQTBQ5sioEK7igkk .
- 3 optimizer steps, toàn bộ tham số trainable nằm đúng một lần trong optimizer; 136 tensor tham số DQK có gradient hữu hạn.
- Best có đủ trọng số bốn module; nạp strict và so sánh bằng nhau toàn bộ **648 state entries**; logits inference `[1,300,33]`.
- Native RGB/resize/normalize parity và postprocess parity: đạt.
- Báo cáo tự tải về `Model_Improvement/Results/rfdetr_dqk_smoke_v4/` gồm manifest, effective config, CSV, training audit và `test/smoke_pipeline.json`.
- Đã chạy riêng `--fetch-run rfdetr_dqk_smoke_v4` và tải lại báo cáo thành công, không khởi chạy train/test mới.
- Smoke v1-v3 bắt được lỗi chuẩn bị cache/metadata export, đã sửa; các thư mục đó được giữ nguyên với trạng thái failed. Chưa có lượt train chính thức `rfdetr_dqk_enhanced_vN` được chạy trong công việc này.

**Không dùng điểm smoke làm kết quả cải tiến cuối.** Kết quả legacy đã kiểm chứng vẫn là mAP50:95=88.00%; kết quả enhanced chính thức chỉ có sau lượt train đầy đủ và test toàn bộ do pipeline mới thực hiện.
