# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record này nói về cả bức ảnh. Vì :
task là image_classification, tức là mô hình gán nhãn cho toàn bộ ảnh.
Record không có bounding box, mask hay tọa độ nào, chỉ có image_width và image_height của cả ảnh.
Top-5 là năm lớp cạnh tranh cho cùng một ảnh. Cả năm đều là xe (cab, minibus, police_van, recreational_vehicle, streetcar), còn trong ảnh có nhiều xe buýt, xe con, người đi bộ.
Nhãn “cab” cũng không chỉ đúng vào một chiếc xe cụ thể. Trong ảnh, các xe buýt xanh và xe con chiếm phần lớn khung hình, nên “cab” ở đây là phán đoán cho cả cảnh giao thông. Vì vậy nhãn này khá lệch so với nội dung thật của ảnh.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Mô hình không tự nghĩ ra tên lớp. Danh sách này đến từ taxonomy_name mà checkpoint được huấn luyện trên đó, ở đây là ImageNet-1K với 1000 lớp cố định. Checkpoint yolo11n-cls.pt có sẵn bảng ánh xạ chỉ số sang tên. Mô hình chỉ chọn trong tập 1000 lớp này, nên nó không thể nói “cảnh giao thông đường phố” hay “ô nhiễm khói xe” nếu những lớp đó không có trong taxonomy.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Cần ghi đủ 3 lớp vì : 
class_id chỉ có nghĩa trong một taxonomy_name cụ thể. Số 468 ở ImageNet-1K là “cab”, còn ở taxonomy_name khác có thể là lớp hoàn toàn khác.
class_name giúp người đọc hiểu ngay, nhưng tên có thể trùng.
taxonomy_name xác định “từ điển” mà hai trường kia thuộc về, nên bắt buộc để kết quả tái lập và đối chiếu được.
Ghi đủ ba trường thì không bị nhầm lẫn giữa các bộ nhãn và có thể kiểm tra lại sau này.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline phải nêu rõ tiêu chí chọn đúng một lớp để mọi người gán nhãn làm giống nhau, ví dụ:
Lớp ưu tiên là vật/chủ đề nổi bật nhất (chiếm diện tích lớn nhất, ở trung tâm) hay là chủ đề của cả cảnh.
Cách xử lý khi nhiều vật ngang nhau: có quy tắc ưu tiên, hay gắn cờ “mơ hồ” thay vì đoán.
Cách xử lý khi taxonomy_name không có lớp phù hợp: chọn lớp gần nhất, hay dùng nhãn “không thuộc lớp nào”.
Cho phép một nhãn (single-label) hay nhiều nhãn (multi-label).
Với ảnh traffic, nếu không có quy tắc này thì người này chọn “minibus”, người kia chọn “cab”, nhãn sẽ không nhất quán.

6. Vì sao không được lấy `score` của model làm ground truth?
Vì : Score là độ tin cậy của chính mô hình, như dòng chú thích trên biểu đồ đã ghi, chứ không phải bằng chứng rằng nhãn đúng.
Mô hình có thể sai mà vẫn tự tin.Ground truth phải do người gán nhãn theo guideline tạo ra, độc lập với mô hình. Nếu lấy dự đoán của mô hình làm nhãn chuẩn thì sẽ đánh giá mô hình bằng chính câu trả lời của nó (vòng lặp tự xác nhận).Điểm số còn bị ảnh hưởng bởi cách phân bố xác suất, và ở ảnh traffic chỉ 0.51 cho hạng 1, nghĩa là mô hình chưa chắc chắn.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33,69.24,498.92,348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Góc trên-trái nằm cách mép trái ảnh khoảng 385.33 pixel và cách mép trên khoảng 69 pixel, tức là gần đỉnh đầu ông ấy.
Góc dưới-phải nằm ở khoảng (498.92, 348.92), tức là gần gót chân.
Hộp rộng khoảng 113.58 pixel và cao khoảng 279.68 pixel. Nó cao gấp hơn hai lần chiều rộng, đúng với dáng người đứng.
Định dạng xyxy là (x trái, y trên, x phải, y dưới), tính bằng pixel, gốc ở góc trên-trái ảnh. Chiều rộng bằng 498.92 − 385.33 = 113.59, chiều cao bằng 348.92 − 69.24 = 279.68, khớp với hai trường trong file.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Threshold 0.20 có 17 prediction ; threshold 0.60 có 6 prediction

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Ví dụ khi hạ từ ngưỡng 0.60 xuống ngưỡng 0.20, reviewer phải xem thêm 11 hộp so với ngưỡng 0.60, và nhiều hộp thuộc lớp mới , có hộp nhiễu/score thấp như : spoon, bottle, potted plant, dining table. Vì vậy nên khối lượng kiểm tra tăng

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Vẽ hộp chữ nhật phải ôm sát khít quanh phần nhìn thấy của vật thể : cạnh hộp chạm sát mép ngoài cùng của vật (không chừa khoảng trống, không cắt mất phần vật), và nếu vật bị cắt bởi mép ảnh thì cạnh hộp đặt đúng tại mép ảnh. Không vẽ một box chung cho hai vật thể tách rời

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

Chọn person 0.61 ở mép trái, hộp [0.12, 263.18, 61.55, 310.78]. Trong ảnh chỉ thấy một cánh tay và bàn tay nhô vào từ mép ảnh, phần còn lại của người bị cắt khỏi khung hình và bị bàn che. Hộp oven bên trái cũng bắt đầu từ x ≈ 0.11, tức là bị cắt ở mép ảnh.

Viết thành quy tắc guideline: cách vẽ hộp cho vật bị cắt hoặc bị che (chỉ ôm phần nhìn thấy, chạm mép ảnh). 
Đưa lên hỏi Lab Coach (escalation): câu hỏi “chỉ thấy một bàn tay thì có tính là một person không?”. Đây là quyết định về chính sách gán nhãn: nếu cho tính thì dữ liệu có thêm nhiều hộp “người” rất nhỏ, nếu không thì mô hình bị dạy bỏ qua người bị che. Ảnh này không đủ để tự quyết, và nếu mỗi người tự đoán thì nhãn sẽ không nhất quán. Khi Lab Coach trả lời, kết quả mới được ghi thành quy tắc.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348; 3–5 điểm đầu của `polygon_xy`=[446, 70], [445, 71], [444, 71], [443, 72], [442, 72]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Box của người này là [385.45, 66.44, 498.02, 348.58], diện tích khoảng 31.760 pixel². Polygon chỉ phủ khoảng 18.170 pixel², tức khoảng 57% diện tích box. Gần một nửa box là nền (tường, nồi, bếp) chứ không phải người.
Polygon cho biết hình dáng thật: đầu tròn, hai tay chống hông tạo khoảng trống hình tam giác, hai chân tách ra, nền lọt giữa hai chân. Box không cho biết gì về những chi tiết này.
Polygon cho biết vật thể nằm ở pixel nào, nên tính được diện tích thật, xác định biên chính xác và biết chỗ vật này chạm hoặc chồng lên vật khác.
Những vật có hình dạng không vuông vức, như dụng cụ treo (spoon) hoặc mặt bàn (dining table, polygon khoảng 55% box), được hưởng lợi nhiều nhất so với box.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id (ví dụ kitchen-001) là tên riêng của từng vật thể cụ thể trong một ảnh. Nó dùng để phân biệt vật này với vật khác cùng loại. Ví dụ ảnh kitchen có 4 cái bowl, mỗi cái có một instance_id khác nhau, và cũng dùng để tham chiếu khi ghi chú, sửa hoặc duyệt đúng mask đó.
Hai vế của câu “không phải class ID và không phải tracking ID” hiểu như sau:
Không phải class ID: class ID (ở đây là 0 cho person trong COCO-80) chỉ loại vật. Mọi person đều có cùng class ID, còn instance_id thì mỗi vật một giá trị. Không dùng nó để biết vật là “cái gì”.
Không phải tracking ID: tracking ID dùng trong video để nói “người này ở khung 1 và người này ở khung 50 là cùng một người”. instance_id chỉ có nghĩa trong một ảnh tĩnh, đánh số theo thứ tự mô hình xuất ra, và không liên kết vật thể giữa các ảnh hay các khung hình.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Vẽ biên mask sát theo mép thật của vật, nhìn thấy của instance, không tràn sang object bên cạnh; vùng bị che khuất thì chỉ vẽ phần còn nhìn thấy (trừ khi guideline nói khác),chỉ bao gồm các pixel của chính vật đó: không để lọt nền vào trong, không cắt mất phần vật, và với biên mờ thì đặt biên tại nơi vật hết rõ ràng. Mỗi vật riêng biệt là một mask riêng, không gộp các vật chạm nhau thành một.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

Chọn kitchen-005 (dining table) và kitchen-007 (oven) ở góc trái dưới. Trong ảnh, mặt nạ bàn (xanh) chạy dưới mặt nạ lò (tím) và các bowl (hồng) nằm chồng lên mặt bàn, nên đâu là biên của bàn, đâu là biên của bowl rất khó xác định. Phần bàn cũng bị che bởi bowl, chai mứt và chiếc cốc nhỏ.
Dùng guideline: cách xử lý khi hai vật chồng nhau (mỗi pixel chỉ thuộc một vật, vật nằm phía trước được ưu tiên, biên giữa hai vật đặt tại đường tiếp giáp nhìn thấy được) và cách vẽ biên mờ. Đây là kỹ thuật chung, áp dụng cho mọi ảnh, nên phải có trong guideline.
Dùng escalation : trong trường hợp mơ hồ mà quy tắc không giải quyết được. Ví dụ vùng tối ở góc trái dưới nơi oven 0.49 và person 0.43 cùng chạm nhau: không rõ vùng đó là lò, là người hay là bóng đổ. Khi ảnh không đủ thông tin để quyết định và nhiều người đọc sẽ đoán khác nhau, đó không còn là vấn đề kỹ thuật vẽ mà là quyết định về nghĩa của nội dung ảnh.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |  |  |  |  |
| Phát hiện vật thể |  |  |  |  |
| Instance segmentation |  |  |  |  |

| Tác vụ | Ground truth (đơn vị nhãn) | Lỗi / chỗ khó quyết định (dẫn chứng JSON/PNG) | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn cho cả ảnh: `class_id` + `class_name` + `taxonomy_name`, do người gán theo guideline | `traffic`: top-1 "cab" (0.51) dù ảnh chủ yếu là xe buýt. `kitchen`: top-1 "gong" (0.42), vì ImageNet-1K không có lớp nhà bếp | Chọn 1 lớp theo quy tắc ưu tiên. Ảnh mơ hồ hoặc taxonomy thiếu lớp thì gắn cờ, không đoán | Nhãn đúng quy tắc chọn lớp, ba trường id/tên/taxonomy khớp nhau, ảnh top-1 thấp hoặc mơ hồ |
| Phát hiện vật thể | Mỗi vật 1 hộp: `class_name` + `bbox_xyxy` (pixel) | `kitchen`: 17 hộp ở ngưỡng 0.20, 11 ở 0.35, 6 ở 0.60. "person 0.61" ở mép trái chỉ thấy tay (bị cắt mép). Các hộp bowl chồng nhau ở góc dưới trái | Vẽ hộp khít phần nhìn thấy, vật bị cắt thì hộp chạm mép ảnh. Ca mơ hồ ("chỉ thấy tay có tính là person không") hỏi Lab Coach | Vật có thật không, đúng lớp không, hộp khít không, hộp trùng, vật bị sót, ảnh hưởng của ngưỡng điểm |
| Instance segmentation | Mỗi vật 1 mask/polygon (`polygon_xy`) + `class_name`, `instance_id` riêng trong ảnh | `kitchen-001` (person): polygon chỉ phủ khoảng 57% box. `kitchen-005` (dining table) chồng lên `kitchen-007` (oven) ở góc trái dưới. `kitchen-004` "potted plant" thực ra là bó lá treo. `kitchen-010` (bowl) chỉ 12 điểm nên biên thô | Vẽ biên sát mép thật, mỗi vật 1 mask, mỗi pixel thuộc 1 vật (vật phía trước ưu tiên). Biên mờ hoặc nội dung mơ hồ thì escalate | Biên lọt nền hoặc cắt mất vật, các vật chạm nhau bị gộp, pixel gán cho 2 vật, mask quá thô, nhãn lớp sai |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

Quy tắc: chỉ dùng các file ảnh và JSON được giao cho bài lab (mẫu COCO), chỉ làm trong môi trường lab, không tải lên, sao chép hay chia sẻ ra ngoài, và không đưa vào bất kỳ dữ liệu cá nhân hay dữ liệu thật của khách hàng.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi: dừng lại, không xử lý hay chia sẻ tiếp, và báo cho Lab Coach. Nếu tài liệu lab chỉ định một đầu mối bảo mật khác thì báo người đó.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
