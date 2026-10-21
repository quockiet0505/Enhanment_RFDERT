# Kế hoạch sửa, kiểm thử checkpoint cũ và huấn luyện lại RF-DETR DQK

**Cập nhật triển khai pipeline:** đã sửa core/trainer và thêm entrypoint versioned train -> best -> test -> download; smoke A100 `rfdetr_dqk_smoke_v4` đã qua 3 optimizer steps, 136 tensor DQK có gradient hữu hạn, nạp lại strict khớp 648 state entries. Hướng dẫn tại `HUONG_DAN_TRAIN_TEST_DQK_V1.md`. Chưa chạy 80 epoch hoặc test toàn bộ cho enhanced; các mô tả "chưa sửa train" bên dưới là lịch sử trước triển khai.

Ngày: 21/10/2026. Tài liệu bổ sung cho `PLAN_REVIEW_RFDETR_DQK.md`.

**Tiến độ mới nhất:** pha sửa test và đánh giá lại legacy đã hoàn tất. Trên 1.481 ảnh, mAP50=98.74%, mAP50:95=88.00%; strict load đủ 510 state entries, normalize/postprocess parity đạt, 10 semantic tests đạt, evaluator script chung cho metric tổng/per-class/counts trùng khớp. Xem `RESULT_LEGACY_TEST_VERIFIED_20261021.md` và `Model_Improvement/Results/RESULT_20261021_021016`. Chưa sửa train hoặc huấn luyện/fine-tune cải tiến mới. Các phần dưới đây giữ lại kế hoạch/bằng chứng của thời điểm lập ban đầu.

## 1. Kết quả xác minh dữ liệu thật trên Modal

Đã tải trực tiếp ba file `_annotations.coco.json` từ volume **food-detection-training**, thư mục `/datasets/completed-project-5/coco`. Đây là volume/path mà entrypoint train DQK mount vào `/vol` và dùng mặc định. Không dùng bộ `train-80-10-10` cũ trên máy để thay thế.

- Train: **10.353 ảnh, 26.317 bounding box, 32 lớp có annotation**.
- Validation (`valid`): **2.976 ảnh, 7.782 bounding box, 32 lớp có annotation**.
- Test: **1.481 ảnh, 3.911 bounding box, 32 lớp có annotation**.
- Mỗi JSON khai báo 33 category. Category ID 0, tên `19-8-BoSungAnhChoCacLopBiTh-v3zo`, không có annotation; đây là mục nhóm Roboflow, không phải lớp thực phẩm hoặc background của classification head.
- 32 lớp thực có category ID 1..32. Thứ tự tên trùng cấu hình DQK trên cả ba split. Training loader Roboflow của RF-DETR cần remap về label 0..31 sau khi bỏ parent category.
- Không phát hiện duplicate image ID, annotation ID, filename trong từng split; không có annotation tham chiếu sai image/category, bbox không hữu hạn, width/height không dương hoặc vượt ảnh trong dung sai 0.01 pixel của kiểm tra đã chạy.
- Không trùng **filename chính xác** giữa ba split. Đây chưa chứng minh không có ảnh trùng nội dung hoặc ảnh augmentation cùng nguồn giữa split; cần kiểm tra pixel hash/perceptual hash/source group trước thí nghiệm chính thức.
- SHA256 annotation test bằng `d7f69d3b3933acd390fae8b65cb577e9e07d7b9b96031ab39ab74cdbc9901376`, **trùng hash trong kết quả DQK cũ**. Điều này xác nhận cùng nội dung annotation test, chưa xác nhận bytes ảnh hoặc volume test bên kia hoàn toàn giống nhau.

Bằng chứng: `DATASET_CLOUD_AUDIT_20261021.json`, ba JSON trong `cloud_audit`, script tái chạy `python References/audit_downloaded_annotations.py`. `DATASET_AUDIT_20261021.json` là khảo sát bộ local cũ, không phải nguồn kết luận trên.

32 tên lớp: beef, bellpepper, bittergourd, bottlegourd, broccoli, cabbage, carrot, cauliflower, chayote, chicken, chickenegg, chickenleg, chickenwin, corn, cucumber, duckegg, eggplant, garlic, ginger, jicama, okra, onion, pork, potato, pumpkin, radish, scallion, shrimp, spongegourd, sweetpotato, tofu, tomato.

Mức mất cân bằng trong train: chickenleg có 3.461 box, shrimp 241 box, chênh khoảng 14.36 lần. Nên báo AP từng lớp và AP theo kích thước; chưa cần tự thêm loss long-tail thuộc phần công người khác. Không suy diễn số box là số ảnh chứa lớp.

## 2. Kết quả xem cấu hình/log huấn luyện thật

Đã đọc từ volume **rfdetr-runs**, thư mục `/rfdetr_dqk_enhanced/checkpoints`:

- Có best.pth, checkpoint_best_total.pth, checkpoint_best_ema.pth, last.pth, last_ema.pth, last.ckpt, các checkpoint chu kỳ, metrics.csv và training_config.json.
- `training_config.json` ghi **model_name=RFDETRMedium, num_classes=32, resolution=640, hidden_dim=256, projector_scale=['P4']**. Không có khai báo bốn module DQK hoặc loss TV/sparsity trong config đã lưu.
- Class names lưu đủ 32 tên theo thứ tự đúng. Vì vậy nguồn dữ liệu và config cloud đều ủng hộ bài toán 32 lớp, không phải 90.
- Cấu hình thực tế: seed=null, num_workers=2, amp_dtype=auto, run_test=false. YAML DQK ghi seed=42, workers=4, amp_dtype=bf16: các giá trị mong muốn này chưa được áp dụng đầy đủ. Weight decay thực tế 1e-4 trùng YAML nhưng vì cũng là default nên không chứng minh code đã forward trường này.
- Validation sử dụng eval_max_dets=500, còn giao thức test chung là 100; hai metric có thể khác nhau dù model đúng. Chọn checkpoint bằng validation là hợp lệ, nhưng phải ghi rõ giao thức validation, không trộn số validation vào bảng test chung.
- metrics.csv có 493 dòng, epoch từ 0 đến 32: log ghi nhận **33 chỉ số epoch**, không đủ chứng minh mọi chi tiết kết thúc job hoặc nguyên nhân dừng.
- Best `val/ema_mAP_50_95` trong log khoảng **0.874971**, best `val/ema_mAP_50` khoảng **0.987845**. Đây là validation; các cực đại có thể ở epoch khác nhau. Không báo chúng là test accuracy.
- Kết quả test cũ chỉ mAP50_95=0.0018, chênh rất lớn so với validation. Đây là lý do ưu tiên sửa loader/preprocessing/postprocessing rồi đánh giá checkpoint cũ trước; chưa thể quy hết cho model kém hoặc chứng minh nguyên nhân duy nhất.

Tên thư mục `enhanced` không chứng minh mạng đã được cải tiến. Config RFDETRMedium và thiếu loss phụ là dấu hiệu cần kiểm tra state_dict; kết luận chắc chắn cần kết hợp trọng số, source đã chạy và runtime hooks.

### Đã kiểm tra trực tiếp best.pth

Đã tải checkpoint 134.690.432 byte từ đúng path config test đang trỏ tới, đọc CPU bằng `torch.load(..., weights_only=True)`:

- SHA256: `c2bdde40551a54c412656d4c5e0e1710201cd45aaa1d1543ba4d367fed8f1379`.
- `model_name=RFDETRMedium`, `rfdetr_version=1.10.1`, epoch=22 (đánh số từ 0), global_step=14904.
- `model` có 510 key, Lightning `state_dict` cũng có 510 key. **Không có key DARM/UFFR/DAFD/MFA trong cả hai**.
- Final `class_embed.weight=[33,256]`, bias=[33]; encoder classification heads cũng có 33 output. Số lớp checkpoint thật đã được xác nhận, không còn chỉ là suy đoán từ config.
- Checkpoint không có optimizer state để resume training đầy đủ; có thể dùng làm nguồn trọng số warm start, không thay thế `last.ckpt` để resume toàn trạng thái.
- Metadata và trọng số phù hợp RFDETRMedium tiêu chuẩn. Không có bằng chứng bốn module cải tiến được lưu; **không dùng checkpoint này để tuyên bố kết quả DQK enhanced**. Không loại trừ mọi biến đổi lịch sử không được ghi metadata, nhưng mạng bốn module trong repo hiện tại không được biểu diễn bởi state này.

Bằng chứng: `cloud_audit/checkpoint_audit.json`. Cực đại validation AP50 và AP50:95 trong metrics cùng ở epoch 22, phù hợp epoch best.pth; giá trị chênh global_step một đơn vị phản ánh cách ghi log/save, không tự nó là lỗi.

**Lỗi loader cụ thể với checkpoint này:** adapter enhanced hiện lấy `checkpoint['model']` rồi load trực tiếp vào wrapper. Các key checkpoint là tên core như `class_embed.weight`, `backbone...`; wrapper đăng ký core dưới `net.`, các nhánh mới dưới `darm.`, `dafd_modules.`, `uffr_modules.`, `mfa_modules.`. Kiểm đếm cho thấy **0/510 key checkpoint mang prefix mà wrapper hiện yêu cầu**. Không có bước chuyển prefix trong adapter. Với đúng code/path hiện tại, strict=False có thể bỏ qua toàn bộ state được huấn luyện và để wrapper ở trọng số vừa khởi tạo. Đây là lỗi cấu trúc key cụ thể, cần sửa trước khi diễn giải mAP=0.0018; chưa chạy lại đúng bản code lịch sử nên không khẳng định đó là nguyên nhân duy nhất của kết quả ngày 19/10.

## 3. Luồng train hiện tại sai/chưa đủ ở đâu

Luồng dự kiến đúng: xác thực dataset -> xây mạng DQK -> tạo criterion/optimizer có module DQK -> train/validate -> chọn checkpoint tốt nhất theo validation -> lưu đủ kiến trúc/mapping -> nạp lại trong tiến trình mới -> test COCO đúng một model.

Code hiện tại:

1. Entrypoint Modal dựng image/mount volume, gọi train, commit và trả path. **Không có bước tự test sau train**. Trong config cloud `run_test=false` cũng xác nhận không bật test nội bộ.
2. Hàm train tạo RFDETRDQKEnhanced rồi gọi `.train(dataset_dir=...)`; wrapper là nn.Module nên API này sai. Nếu khởi tạo lỗi thì fallback baseline, dễ khiến người chạy tưởng đã train enhanced. Code trên đĩa hiện tại không thể được xem là bằng chứng tái hiện lần train cloud cũ đã thành công.
3. DARM nằm ở forward nhưng ba nhánh còn lại không được gọi; loss phụ chưa gắn objective; hình dạng feature chưa được xác minh đúng.
4. **Không thể chỉ đổi thành `base_model.train(...)` để sửa:** source RF-DETR 1.10.1 `detr.py` dựng mới `RFDETRModelModule(self.model_config, config)`, và module này gọi `build_model_from_config` để xây core. Việc sửa/gắn hook vào model facade đang có có thể bị mất khi trainer dựng core mới. Phải tích hợp tại nơi tạo model của trainer hoặc dùng subclass module/trainer có kiểm soát rõ ràng.
5. Trainer phải sở hữu cùng một mạng cải tiến khi forward, optimizer, EMA, validation, save/load và test. Các tensor loss thêm vào output không tự thành criterion loss; source training_step chỉ cộng key có trong loss_dict và weight_dict.
6. Copy best ưu tiên EMA file không tương đương lựa chọn winner chính thức. Source có `checkpoint_best_total.pth` đã chọn regular/EMA; code hiện tại không ưu tiên file này. Cần dùng metadata winner, không chọn theo tên tồn tại.
7. Tự resume thư mục cũ có thể kéo nhầm baseline vào run mới. Không resume optimizer/EMA của kiến trúc cũ sang kiến trúc mới; việc dùng làm warm start phải là thao tác riêng, có báo cáo key được chuyển.

## 4. Pha I: sửa test và chẩn đoán checkpoint cũ trước

Mục tiêu: trả lời checkpoint cũ thực sự chứa gì và có đánh giá lại đúng được không. Đây là chẩn đoán, chưa phải chứng minh cải tiến đã hoàn thành.

### I.1. Chốt provenance và phân loại checkpoint

- [ ] Lưu bản sao checkpoint/config/metrics kèm hash; không ghi đè file cloud cũ.
- [ ] Kiểm tra state_dict: class_embed và encoder class heads; với custom 32 lớp expected 33 hàng, hidden_dim=256.
- [ ] Liệt kê key DARM/UFFR/DAFD/MFA, shape và metadata kiến trúc. Nếu key chỉ là mạng RFDETRMedium thì đánh giá nó dưới tên baseline/legacy, không gắn nhãn enhanced.
- [ ] Nếu có key DQK, kiểm tra chúng có thực sự được train: log loss, optimizer membership, source phiên bản cũ, hooks và so sánh trạng thái initialization khi có bằng chứng. Chỉ tồn tại key hoặc weight khác zero chưa đủ chứng minh nhánh đã được gọi.
- [ ] Chọn dependency đúng phiên bản training checkpoint cho lượt tái đánh giá đầu tiên. Xác minh package source/API với metadata; không tự nâng train/test khác phiên bản.

### I.2. Sửa adapter/evaluator trong DQK

- [ ] Giữ COCO metric và ngưỡng của common-coco-v7: imgsz=640, batch=1, AP conf đầu vào 0.0, P/R/F1 conf=0.55, IoU=0.5, maxDet=100, 32 lớp, 1481 ảnh.
- [ ] Dùng preprocessing/postprocessing chuẩn đúng phiên bản; RGB, resize/normalize, top-k query-class pairs và rescale box phải khớp. Loại background và map label dựa trên schema checkpoint.
- [ ] Load đúng kiến trúc với kiểm tra key/shape bắt buộc; mọi lỗi checkpoint hoặc ảnh phải dừng. Không fallback pretrained/baseline ngẫu nhiên; không im lặng bỏ nhãn lạ.
- [ ] Một config DQK chỉ enable model đang đánh giá. Nếu checkpoint được xác định baseline thì đặt tên riêng tương ứng; không khởi tạo wrapper enhanced để che mismatch.
- [ ] Chỉ một đường test chính thức, bỏ dò max-F1 trên test. Confusion matrix dùng threshold hợp đồng; trường hợp muốn ngưỡng khác phải báo riêng.
- [ ] Tạo output/run ID mới; export raw predictions, số detection/ảnh/lớp, hash checkpoint/dataset/source, dependency versions, class mapping, effective inference config.
- [ ] Đối chiếu đúng protocol nguồn/hash với nhóm; không đổi nội dung evaluator rồi giữ hash giả của script chung.

### I.3. Kiểm thử trước khi chạy 1481 ảnh

1. Tiny COCO tổng hợp: perfect prediction, empty prediction, duplicate FP, sai lớp, ảnh không GT, sparse category IDs, crowd/ignore, không đủ lớp ở split nhỏ. Kết quả phải khớp pycocotools với convention đã định.
2. Cho cùng tensor logits/boxes, postprocessor DQK và chuẩn phải cho cùng labels/scores/boxes trong sai số float; có trường hợp một query có hai class điểm cao để phát hiện lỗi max-per-query.
3. Save/load round-trip phải khớp prediction; checkpoint sai head/prefix/thiếu DQK phải bị từ chối theo chế độ tương ứng.
4. Dùng một nhóm ảnh validation cố định để kiểm tra trực quan box/class/conf và parity với API chuẩn. Không dùng test để chọn ngưỡng/siêu tham số.
5. Nếu mọi kiểm tra trên đạt, chạy test đầy đủ một lần với checkpoint cũ đã phân loại đúng và lưu kết quả riêng. Không đặt tiêu chí "mAP phải tăng" để tự chỉnh test cho đẹp số.

Điều kiện dừng pha I: kiểm tra metric/parity fail, mapping sai hoặc checkpoint load không đầy đủ -> sửa nguyên nhân; chưa train dài để bù cho evaluator sai.

## 5. Pha II: sửa kiến trúc và smoke train cải tiến

Pha này cần cho mục tiêu bốn kỹ thuật DQK kể cả khi test lại baseline đạt điểm tốt.

- [ ] Chốt integration point dựa trên core/trainer 1.10.1 hoặc phiên bản đã thống nhất. Subclass/module factory phải tạo DQK core trước optimizer và EMA. Hạn chế monkey-patch toàn cục khó kiểm soát.
- [ ] DARM ở vị trí đầu vào thống nhất với normalize; nếu xử lý ảnh [0,1] hay tensor normalized phải mô tả rõ biến thể. Không tùy tiện clamp vì có thể cắt gradient.
- [ ] UFFR tại backbone feature: sửa FFT bin, bảo toàn đối xứng; kiểm tra DC/sinusoid/impulse, kích thước chẵn/lẻ. Chốt binary hay Gaussian theo tài liệu/phân công, ghi rõ sai khác.
- [ ] DAFD ở taps đã chọn, đúng channel; sửa prior centers/widths, FiLM initialization, regularizer target gate 0.8; giữ Gaussian overlapping không normalize across bands nếu bám bài.
- [ ] MFA trước Transformer tại feature neck đã xác định; không giả định Medium có ba mức pyramid. Ghi rõ kernel/dilation/scales so với bài fmars.
- [ ] Criterion cộng TV và sparsity đúng trọng số; detection loss giữ cùng baseline để so sánh module. Không tự dùng một bộ loss của bài khác mà không ghi thiết kế.
- [ ] YAML được validate và forward đầy đủ: seed, workers->num_workers, amp_dtype, weight decay, eval settings, loss weights. Log effective config, code hash và cờ module.
- [ ] FFT float32 trong autocast khi cần, cast về dtype phù hợp; kiểm tra finite loss/gradient ở bf16 trên GPU thật. Không chỉ kiểm thử kích thước power-of-two.
- [ ] Giữ NestedTensor/mask/API đúng; không repeat/cắt channel để né lỗi; forward có targets/aux outputs/grouped queries phải tương thích trainer và inference.

Nghiệm thu smoke: qua ít nhất vài optimizer steps; hook xác nhận cả bốn module chạy đúng feature; parameter IDs đều có trong optimizer; sau các bước đầu phù hợp zero-init, tham số cần học thay đổi; loss tổng khớp các thành phần; EMA có đủ key DQK. Overfit một tập train nhỏ rồi validation riêng 1-2 epoch, reload checkpoint trong tiến trình mới, kiểm tra inference parity và resume.

## 6. Quyết định có cần train lại hay không

### Chỉ cần sửa test và đánh giá lại

Chỉ áp dụng nếu checkpoint đã đúng kiến trúc bốn module, có bằng chứng chúng được optimize cùng loss yêu cầu, và lỗi chỉ ở loader/pre/postprocess/metric. Khi đó không train lại chỉ vì báo cáo test trước bị sai. Giữ kết quả test cũ như lỗi kỹ thuật, không dùng để so sánh.

### Bắt buộc train cải tiến thực sự

Áp dụng nếu checkpoint là RFDETRMedium thuần; thiếu module DQK; nhánh không được gọi; optimizer không chứa tham số DQK; TV/sparsity chưa được tối ưu; hoặc thay đổi thuật toán FFT/FiLM làm thay đổi mạng so với lần train cũ. Với lỗi hiện tại đã tái hiện, **khả năng phải train lại để đáp ứng đủ bốn kỹ thuật là cao**, kể cả checkpoint baseline sau sửa test cho mAP tốt.

**Quyết định sau khi đọc best.pth thật:** checkpoint hiện được config test chọn thiếu cả bốn module, vì vậy cần một run huấn luyện/fine-tune cải tiến mới sau khi sửa kiến trúc và đạt smoke test. Đánh giá lại checkpoint cũ vẫn cần để chẩn đoán bộ test và tận dụng baseline, nhưng dù kết quả tốt cũng không thay thế run cải tiến này.

Không nhất thiết train từ đầu hoàn toàn: có thể warm-start backbone/decoder từ checkpoint baseline đã xác thực, khởi tạo module DQK theo thiết kế rồi fine-tune toàn mạng. Phải gọi là warm start/fine-tune, không resume tiếp optimizer/EMA cũ một cách mù quáng. Để so sánh công bằng, baseline và enhanced cần cùng initialization policy và budget, hoặc báo riêng thí nghiệm warm-start bổ sung.

### Chưa train lại khi còn lỗi hạ tầng/test

Nếu dữ liệu chưa đồng nhất, metric parity fail, save/load fail, gradient NaN, hoặc module chưa được gọi: sửa và chạy smoke trước. Một run 80 epoch không giải quyết các lỗi này.

## 7. Pha III: train đủ và test RF-DETR cải tiến

Sau khi pha II đạt:

- Run directory mới, architecture version mới, không tự resume từ `/rfdetr_dqk_enhanced/checkpoints` cũ.
- Theo kế hoạch cấu hình hiện có: tối đa 80 epoch, patience=10, chọn best validation AP50:95; lưu lý do dừng và checkpoint winner. Đây là cấu hình đề tài, không khẳng định mọi bài gốc dùng y hệt.
- Đặt rõ seed=42, resolution=640, batch/accumulation và effective batch thực tế, AMP/GPU. Nếu thiếu VRAM, thay batch và grad_accum có ghi lại; không tự giảm mà không lưu effective config.
- Checkpoint lưu đầy đủ state inference, schema 32 lớp, module flags, dependency versions, source hash; resume checkpoint riêng có optimizer/scheduler/scaler/EMA/global step/RNG khi cần tái lập.
- Train kết thúc -> chọn checkpoint winner -> commit -> tiến trình test mới nạp đúng path/hash -> assert kiến trúc và số lớp -> chỉ test RF-DETR enhanced -> export kết quả. Nếu train hoặc test lỗi thì trạng thái job là failed, không in hoàn tất thành công.
- Baseline và ablation là các run riêng, cùng data/seed/budget/protocol. Nên có baseline, từng module và tổ hợp đầy đủ để tránh kết luận mọi tăng điểm là do cả bốn module. Với tài nguyên cho phép, thêm nhiều seed và báo độ biến thiên.

Đầu ra nghiệm thu: run manifest, config thực tế, logs detection/TV/sparsity, checkpoint hash, COCO prediction JSON, metric tổng/per-class/AP theo size, confusion matrix, latency/FPS trên cùng GPU và ghi phạm vi FLOPs (FFT/unsupported ops).

## 8. Phạm vi đã làm trong lượt rà soát này

Đã đọc cloud annotation, cấu hình/log huấn luyện, kiểm tra annotation và source trainer đúng phiên bản; đã tạo kế hoạch sửa/test trước rồi mới train lại. Chưa sửa production code train/test, chưa chạy GPU inference hoặc huấn luyện mới. Số ảnh/box nêu ở mục 1 là số bản ghi annotation, chưa kiểm kê từng file ảnh trên volume.
