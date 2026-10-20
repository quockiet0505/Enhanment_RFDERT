# Kế hoạch kiểm tra và sửa RF-DETR cải tiến của Dương Quốc Kiệt

Ngày rà soát: 21/10/2026. Trạng thái: đã rà soát tĩnh và chạy kiểm tra module độc lập; chưa sửa mã huấn luyện/test, chưa chạy train hoặc đánh giá GPU. File này là kế hoạch trước khi triển khai sửa.

## 1. Kết luận hiện tại

**Code hiện tại chưa đáp ứng yêu cầu huấn luyện và kiểm thử RF-DETR cải tiến một cách đáng tin cậy.** Có lỗi chặn chạy, module khai báo nhưng không tham gia đồ thị tính toán, thiếu loss phụ và khác biệt tiền xử lý test. Không nên huấn luyện dài hoặc dùng kết quả hiện có để kết luận hiệu quả cải tiến trước khi sửa các lỗi P0/P1 bên dưới.

Mục tiêu cuối: trong DQK, train RF-DETR với đúng bốn kỹ thuật được phân công; sau train nạp lại đúng checkpoint cải tiến và chỉ test RF-DETR đó theo quy trình COCO chung. Baseline RF-DETR và ablation chạy riêng phục vụ kiểm chứng khoa học; không tự chạy YOLO/RetinaNet/RT-DETR.

## 2. Tài liệu và phạm vi đã đọc

- `Phân công.docx.md`: Dương Quốc Kiệt phụ trách DARM, UFFR, MFA, DAFD. QAR/DQCD/SCU/LUE/IQCM/HIFRM thuộc người khác, không bắt buộc triển khai vào nhánh này.
- `EL_34_10_13.pdf`: DARM, mục III.B, phương trình (1)-(3), trang 3-4.
- `fmars-13-1885615 (1).pdf`: UFFR/MFA, mục 3.1-3.3 và cấu hình thực nghiệm. Đây là file tương ứng bài fmars được ghi trong phân công.
- `jimaging-12-00401-v2.pdf`: DAFD, mục 3.2, phương trình (1)-(10), trang 5-7.
- Toàn bộ nhóm mã `modules_dqk`, wrapper/train/evaluate/confusion matrix RF-DETR, Modal entrypoint, cấu hình, helper dữ liệu/log/benchmark/report; bộ test DQK và hai file tham chiếu tại `E:/CNTT/NCKH/Train-V2/TEST-COMMON`.
- Đã trích văn bản của cả ba PDF vào `References/extracted` để tra cứu. Công thức/hình có thể mất định dạng khi trích; cần kiểm tra trực quan công thức và sơ đồ tại bước triển khai thuật toán.
- Không tìm thấy `RTK.md` ở DQK hoặc qua tìm kiếm dưới `E:/CNTT`. Do đó chưa thể áp dụng hướng dẫn trong file được tham chiếu đó.

Tài liệu API chính thức đã đối chiếu thêm: https://rfdetr.roboflow.com/latest/reference/rfdetr/ . Tài liệu latest chỉ giúp xác định luồng API, **không thay thế việc đọc source đúng phiên bản 1.10.1/1.11.0** trước khi tích hợp.

## 3. Lỗi P0: phải sửa trước mọi lần train/test nghiêm túc

### P0.1. Nhầm API huấn luyện với API đổi chế độ PyTorch

Vị trí: `Model_Improvement/src/rfdetr/lwdetr_dqk.py`, `train_dqk.py`.

`RFDETRDQKEnhanced` kế thừa `torch.nn.Module` nhưng không định nghĩa bộ huấn luyện. Lệnh `model.train(**train_kwargs)` gọi `nn.Module.train(mode=True)`, không phải trainer RF-DETR. Với đối số `dataset_dir`, lệnh gây `TypeError`. Đã tái hiện đúng loại lỗi trên `nn.Module` độc lập, chưa chạy wrapper vì máy chưa cài `rfdetr`.

Cách sửa dự kiến: giữ `nn.Module.train(mode)` đúng chuẩn PyTorch; tách facade/trainer khỏi mạng lõi, tích hợp mạng cải tiến vào đúng trainer của phiên bản RF-DETR đã chọn. Không chữa bằng cách chuyển sang `base_model.train()` nếu mạng mà trainer dùng vẫn bỏ qua module DQK. Phải kiểm tra optimizer, EMA, checkpoint đều bao gồm mạng cải tiến.

### P0.2. Ba module không được dùng

Wrapper khởi tạo `dafd_modules`, `uffr_modules`, `mfa_modules` nhưng `forward()` chỉ chạy DARM rồi gọi `self.net`. Không có vị trí nào gọi ba ModuleList này. Vì vậy hiện không có tác động DAFD/UFFR/MFA lên dự đoán hoặc detection loss; đếm tham số vẫn có thể tính cả module không chạy.

Cách sửa: xác định cấu trúc feature thật của RFDETRMedium đúng phiên bản, đặt UFFR ở feature backbone trước neck/projector, MFA ở đầu ra neck trước Transformer; đặt DAFD tại feature/tap đã xác minh với bài J.Imaging. Không giả định mọi feature có 256 kênh hay luôn có ba mức. Thứ tự tương tác DAFD với UFFR là thiết kế kết hợp của đề tài, cần mô tả riêng và ablation; ba bài không chứng minh sẵn tổ hợp cả bốn module.

### P0.3. Loss phụ chưa tham gia tối ưu

`loss_tv` chỉ được thêm vào output; chưa có criterion/training step cộng nó vào tổng loss. `loss_sparsity` luôn là tensor 0. Không có hệ số TV/sparsity tương ứng trong YAML. Thêm khóa vào output không có nghĩa trainer tự tối ưu loss đó.

Cách sửa: nối `L_total = L_detection + lambda_tv * L_tv + 1e-5 * L_sparsity`; với DAFD dùng `mean_k((mean(G_k)-0.8)^2)` theo phương trình (10). Xác minh hệ số TV từ bài DARM hoặc ghi rõ là lựa chọn thực nghiệm, không tự nhận là hệ số gốc. Log riêng các loss, gradient và trạng thái bật/tắt.

### P0.4. Test có thể nạp sai checkpoint nhưng vẫn xuất kết quả

Vị trí: `src/testing/common_test.py`, `RFDETRAdapter.__init__`.

Nhánh enhanced dùng `load_state_dict(..., strict=False)` và chỉ in số lượng missing/unexpected keys. Checkpoint baseline hoặc khác cấu trúc có thể không khôi phục đúng backbone/module mới. Chưa có checkpoint local để xác nhận file cloud đang thuộc trường hợp nào.

Cách sửa: thống nhất schema checkpoint gồm architecture, phiên bản, class mapping, cấu hình và state; yêu cầu khôi phục đủ trọng số inference. Chỉ cho phép ngoại lệ đã liệt kê, không bỏ qua thiếu module cải tiến. Không load được thì dừng, không xuất status thành công. Round-trip save/load phải cho prediction giống nhau trong sai số số học.

### P0.5. Đánh giá cũ có thể trở thành đánh giá baseline

`src/rfdetr/evaluate.py` chỉ khởi tạo `RFDETRMedium`, không khởi tạo kiến trúc enhanced. Nếu load lỗi, fallback cuối có thể khởi tạo model không có checkpoint cần đánh giá. `train_dqk.py` cũng fallback về baseline nếu khởi tạo enhanced lỗi.

Cách sửa: bỏ fallback âm thầm sang kiến trúc khác; hợp nhất đường đánh giá vào adapter enhanced theo COCO chung. Trả lỗi rõ ràng khi cấu hình/checkpoint không hợp lệ.

## 4. Đối chiếu bốn thuật toán theo tài liệu

### DARM

`darm.py` đã có residual `I' = I + alpha*R` và TV trên residual: đúng ý tưởng chính. Tuy nhiên TV chưa nối vào objective; vị trí xử lý trước hay sau normalize phải được quy định thống nhất train/test. Cần đối chiếu cấu trúc conv/activation với mô tả/hình trước khi gọi là tái hiện nguyên bản. TV hiện chia theo B*C*H*W là quy ước normalize bổ sung, phải ghi rõ vì ảnh hưởng trọng số loss.

### UFFR

`uffr.py` có FFT2, fftshift, reweight low/high, ifftshift, IFFT; alpha 0.95/1.55 và radius 0.6 phù hợp cấu hình được nêu trong bài fmars. Hiện dùng **mask nhị phân**; phân công ghi mask Gaussian. Không nên tự thay toàn bộ sang Gaussian chỉ để khớp câu phân công: phải chốt khác biệt giữa phần phương trình/mô tả bài fmars và yêu cầu phân công, ghi rõ biến thể được dùng.

Lưới `linspace(-1,1,H/W)` không bám chính xác các bin `fftshift(fftfreq)` với kích thước chẵn; cần kiểm tra tâm DC và đối xứng liên hợp. Việc lấy `.real` có thể che phần ảo do mask không đối xứng. Kiểm tra bằng ảnh hằng, impulse và sóng sin tần số xác định, kích thước chẵn/lẻ.

### MFA

`mfa.py` có nhánh depthwise 5/7/9, lấy trung bình, avgpool tách low/high, gating/dropout/residual. Đây là attention có xấp xỉ tần số không gian, không bắt buộc FFT riêng; bài fmars cũng mô tả cách xấp xỉ này. Cần đối chiếu cụ thể scales/kernel/dilation với cấu hình bài; không mặc định ba kernel hiện tại là tái hiện chính xác. Lỗi lớn nhất vẫn là chưa nối MFA vào forward.

### DAFD

`dafd.py` đã có rFFT, Gaussian ba dải, log amplitude, FiLM, sigmoid gate, irFFT từng dải, concat+1x1 fusion, residual alpha=0.15, fusion khởi tạo 0: phần khung gần phương trình (1)-(9).

Những điểm phải sửa/kiểm chứng:

- Centers thực tế khởi tạo khoảng `[0.1104,0.2612,0.4092]`, widths `[0.0861,0.1168,0.1502]`; không khớp prior `[0.10,0.35,0.60]`, `[0.12,0.18,0.25]` trong bài. Dùng inverse sigmoid của phép biến đổi tương ứng và xác minh convention bán kính.
- Gate conv có bias logit(0.8), nhưng FiLM gamma/beta khởi tạo ngẫu nhiên nên gate cuối không được đảm bảo bằng 0.8. Muốn bảo đảm cần gamma=1, beta=0 ban đầu hoặc phương án tương đương được ghi rõ.
- Chưa tính regularizer (10). Gate stats hiện gộp toàn batch; đủ cho regularizer toàn cục nhưng không đủ nếu sau này cần thống kê từng ảnh cho DQCD (ngoài phạm vi DQK).
- Xóa cách repeat/cắt kênh để che sai shape. Sai số kênh phải báo lỗi hoặc dùng projection học được có chủ đích.
- Nhánh list chỉ xử lý x[0] rồi trả tensor, làm mất cấu trúc còn lại; cần contract tensor/feature collection nhất quán.
- Fusion khởi tạo 0 khiến gradient nhánh trước fusion có thể bằng 0 ở bước đầu, đây là hệ quả hợp lý của identity initialization; kiểm tra gradient nhiều bước, không kết luận chỉ từ một backward.

## 5. Bộ test chung: giữ phần nào, sửa phần nào

Hai file `Train-V2/TEST-COMMON` là mốc giao thức của nhóm; không sửa file ngoài DQK. File config chung hiện bật YOLOv11 và RetinaNet, RFDETR_Medium tắt, nên chạy mặc định trực tiếp sẽ không đáp ứng mục tiêu DQK.

Các thiết lập nên giữ khi port vào DQK: COCOeval bbox, 640, batch=1, conf đầu vào 0.0 cho AP, operating conf=0.55, IoU=0.50 cho P/R/F1, max detections=100, đủ 32 lớp, đúng danh sách 1481 ảnh theo split nhóm. Số lượng 1481 hiện là yêu cầu config, chưa xác minh dataset thật ở cloud. Không đổi split cho khớp con số. P/R/F1 ở operating point là metric bổ sung, khác AP tích phân nhiều ngưỡng.

Bản DQK dùng COCOeval và lấy TP/FP/FN từ evalImgs là hướng hợp lý. Tuy nhiên chưa đủ để gọi là đã chuẩn vì adapter và kiểm soát giao thức còn lỗi:

1. Manual inference chỉ `ToTensor + Resize`, không Normalize theo pipeline backbone chuẩn. Cần tái sử dụng preprocessing/postprocessing chính thức đúng phiên bản, bao gồm RGB, resize, mean/std và scale box về ảnh gốc.
2. Manual top-k lấy một class tốt nhất/query, khác top-k trên query-class pairs của pipeline cần đối chiếu theo source phiên bản. `topk(k)` còn có thể vượt số query ở kiến trúc khác. Phải dùng cùng hậu xử lý với baseline để so sánh công bằng.
3. Adapter nhận imgsz nhưng enhanced khởi tạo không truyền resolution. Hiện default tình cờ là 640; phải truyền rõ và kiểm tra effective config.
4. Nhãn class đang dựa vào danh sách mặc định; cần xác minh label offset/background/dummy category từ dataset và checkpoint, không mặc định 0..31 là đúng. Mọi nhãn không ánh xạ phải báo lỗi thay vì `continue` âm thầm.
5. Train Modal pin rfdetr 1.10.1, test pin 1.11.0, requirements không pin. Chốt một phiên bản, kiểm tra source thực và lock train/test đồng nhất trước khi sửa kiến trúc.
6. `PROJECT_ROOT = Path.cwd()` và `.add_local_dir('./src')` phụ thuộc nơi gọi lệnh. Dùng đường dẫn dựa trên `__file__`, mount đầy đủ src/config và kiểm tra import trong cloud.
7. Train dùng `/vol/datasets/.../coco` trên volume `food-detection-training`; test dùng `/data/dataset/...` trên volume khác. Phải chứng minh cùng split và nhãn bằng manifest/hash, không chỉ cùng tên project.
8. Cloud test dùng thư mục output cố định `Common_Eval_Results`, có thể trả lại file cũ khi lần chạy hiện tại thiếu checkpoint. Dùng run ID riêng, bắt buộc checkpoint tồn tại, kiểm tra đúng một kết quả mới được tạo.
9. Config DQK có hai model nhưng expected_num_models=9; validator đã bị rút gọn, không còn bảo đảm đầy đủ invariant. Cấu hình chế độ DQK một model enhanced, baseline chỉ chạy riêng khi yêu cầu; validator phải kiểm tra kích thước, batch, ngưỡng và số lớp.
10. Hai evaluator có hash source khác nhau nên dù cùng tên protocol vẫn không tự động merge được theo giao thức chung có hash. Cần thống nhất cách mở rộng adapter và cơ chế xác nhận giao thức với nhóm; không giả mạo hash bản gốc.
11. `confusion_matrix.py` dò ngưỡng max-F1 trên test: không dùng con số đó cho bảng benchmark chung. Dùng 0.55 cố định; nếu tune ngưỡng thì chỉ tune validation rồi đóng băng trước test.
12. `evaluate.py` còn ngưỡng 0.01/0.25 và phối trộn nguồn metric; dùng một đường COCO test chính thức duy nhất. Bắt lỗi suy luận phải dừng và ghi ảnh gây lỗi.

## 6. Các vấn đề hỗ trợ

- Resume/checkpoint: chỉ ưu tiên EMA khi file tồn tại chưa chắc là checkpoint tốt nhất toàn bộ; chọn theo metric validation có metadata. Kiểm tra tên file thật của trainer, optimizer/scheduler/scaler/epoch và EMA khi resume. SIGTERM hiện chỉ copy file có sẵn, không lưu trạng thái bước đang chạy; không gọi đó là bảo vệ toàn diện khi timeout.
- YAML: workers/weight_decay/seed/amp_dtype/eval_interval/loss_weights đang không được forward từ hàm train. Config mặc định còn trỏ `rfdetr.yaml` không tồn tại trong thư mục này. Cần validate config, fail rõ ràng, log effective config và tránh âm thầm dùng default.
- Log export dùng validation loss thay train loss khi thiếu train tag, dùng 0 cho metric thiếu và đánh số lại epoch từ 1; có thể gây sai báo cáo/resume. Giữ epoch thật, giá trị thiếu là null/NaN có lý do, lấy đúng callback trainer.
- Benchmark: tham số có thể gồm module không chạy; công cụ FLOPs có thể bỏ FFT/operator không hỗ trợ. Báo unsupported ops/coverage, không trình bày số GFLOPs như phép đếm đầy đủ nếu chưa xác minh. Thống nhất đo latency cả preprocessing/postprocessing hay chỉ forward giữa các model.
- `metrics_reporter.py` có số phần cứng fallback và nhận xét so sánh cố định. Thay bằng dữ liệu đo hoặc N/A; bỏ so sánh RT-DETR trong luồng mặc định DQK.
- `validate_dataset.py`: biểu thức chọn split có thể chọn `valid` cho cả train/test khi key `valid` tồn tại. Sửa alias chỉ cho validation; thêm validator COCO, bbox hữu hạn, ID hợp lệ và kiểm tra trùng ảnh giữa split.
- Downloader kiểm tra tồn tại file train chưa bảo đảm download hoàn chỉnh; tránh tự xóa/chỉnh dataset trong đường train/test, dùng manifest xác thực và thao tác chuẩn bị dữ liệu riêng.

## 7. Kế hoạch sửa theo thứ tự và tiêu chí nghiệm thu

### Bước A: đóng băng môi trường và dữ liệu

- [ ] Chọn/pin RF-DETR và dependencies, đọc source đúng phiên bản; xác minh resolution 640 hợp lệ với patch/window model.
- [ ] Chốt class mapping từ COCO train/valid/test và checkpoint; manifest ảnh, annotation SHA256, bảo đảm không rò rỉ split.
- [ ] Tách run/checkpoint mới khỏi kết quả cũ; lưu metadata đầy đủ.

Đạt khi local/cloud import được, train/test cùng phiên bản và ánh xạ lớp có kiểm tra tự động.

### Bước B: sửa mạng và trainer

- [ ] Tách facade huấn luyện và core nn.Module; không phá `.train(mode)`/`.eval()`.
- [ ] Nối DARM/UFFR/DAFD/MFA đúng feature shape và vị trí; giữ mask/container gốc, không tạo NestedTensor giả thiếu API.
- [ ] Gắn loss TV/sparsity vào objective và optimizer; cấm fallback baseline.
- [ ] Sửa khởi tạo DAFD/lưới tần số theo convention đã chốt; thêm cờ bật/tắt cho ablation.

Đạt khi một train step hữu hạn, optimizer chứa toàn bộ tham số cần học, hook chứng minh mỗi module chạy; sau vài bước các nhánh có gradient phù hợp và tham số thay đổi. Kiểm tra float32 và mixed precision ở kích thước feature chẵn/lẻ, đặc biệt FFT half ở kích thước không phải lũy thừa hai. Nếu cần, FFT chạy float32 rồi cast về dtype đầu vào.

### Bước C: checkpoint và đánh giá

- [ ] Lưu/nạp strict cùng kiến trúc; prediction trước/sau reload khớp.
- [ ] Dùng một adapter enhanced duy nhất, preprocessing và postprocessing đồng nhất train/API chuẩn.
- [ ] Port evaluator chung trong DQK, chỉ bật enhanced; CLI hỗ trợ chọn model/device/output và kiểm tra config.
- [ ] Output từng run riêng, không lấy file cũ, không bỏ qua checkpoint/ảnh lỗi; train xong tự gọi test với best checkpoint được trả về.
- [ ] Bỏ dò ngưỡng trên test và các fallback tạo metric giả/0 khi lỗi.

Đạt khi tiny COCO có dự đoán hoàn hảo cho AP gần 1; trường hợp không prediction, FP trùng, sai lớp, crowd/ignore và category ID không liên tục cho kết quả mong đợi. Trên cùng prediction JSON, evaluator DQK và evaluator chung phải cho metric tương đương.

### Bước D: smoke run rồi mới train đủ

- [ ] Overfit tập train rất nhỏ, validation riêng, 1-2 epoch để kiểm tra loss/checkpoint/resume/EMA/log.
- [ ] Chạy toàn bộ test đúng một model RFDETR_DQK_Enhanced sau reload best; đủ ảnh và đúng protocol.
- [ ] Sau khi các điều kiện trên đạt, mới train đủ cấu hình 80 epoch với early stopping theo validation.
- [ ] Thí nghiệm riêng: baseline RF-DETR, từng module, kết hợp dần và cả bốn; giữ budget/seed/dữ liệu giống nhau. Nếu có điều kiện chạy nhiều seed, báo mean/std. Không tune bằng test.

Đạt khi có log tái lập, metric test thật, latency trên cùng GPU, và bằng chứng ablation. Không cam kết mAP phải tăng chỉ vì thuật toán đã triển khai đúng; domain thực phẩm khác domain haze/underwater/mining của bài gốc.

## 8. Bằng chứng đã chạy và giới hạn kết luận

- Parse AST tất cả file Python trong `Model_Improvement`: đạt cú pháp.
- Chạy forward/backward CPU riêng DARM/UFFR/DAFD/MFA trên tensor batch=2, không gian 15x17: đúng shape, output hữu hạn, tham số có đối tượng gradient. Đây không chứng minh gradient khác 0 hoặc tích hợp trainer đúng.
- Đã in centers/widths DAFD sau sigmoid: sai prior bài như mục 4.
- Đã tái hiện lỗi `nn.Module.train(dataset_dir=...)`: TypeError.
- Máy có torch và pdfplumber, chưa có package rfdetr; trong cây DQK không thấy checkpoint/dataset để chạy end-to-end. Chưa kiểm chứng source dependency cài trong cloud, GPU/AMP, checkpoint thực tế, hay metric tái lập.
- Kết quả lưu `Results/RESULT_20261019_235139/result_RFDETR_DQK_Enhanced.json` có mAP50=0.0024, mAP50_95=0.0018, P/R/F1=0. Đây là dấu hiệu cần điều tra nhưng không đủ xác định nguyên nhân; các lỗi adapter/load/normalize ở trên là khả năng cần kiểm chứng, không phải kết luận về checkpoint đã chạy trước đó.

Ưu tiên thực hiện: **A -> B -> C -> D**. Không chỉ chỉnh tên model/config hoặc bật enabled rồi coi như đã hoàn tất cải tiến.
