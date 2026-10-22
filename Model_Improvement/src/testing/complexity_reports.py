"""Attach an audited complexity estimate without recomputing accuracy metrics."""
import csv
import io
import json
from pathlib import Path


def attach(test_dir):
    folder = Path(test_dir)
    profile = json.loads((folder/"complexity_profile.json").read_text(encoding="utf-8"))
    if not profile["complete_operator_coverage"] or profile["unsupported_ops"] or profile["GFLOPs"] is None:
        raise ValueError("Cannot publish a partial FLOP count")
    result_path = folder/"result_RFDETR_DQK_Enhanced.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["model_provenance"]["checkpoint_sha256"] != profile["checkpoint_sha256"]:
        raise ValueError("Profile and accuracy report refer to different checkpoints")
    value = round(profile["GFLOPs"], 2)
    result["GFLOPs"] = value
    result["complexity_profile"] = profile
    result["hardware_note"] = "GFLOPs is an analytic fvcore estimate (1 FMA = 1 FLOP); see complexity_profile.json. Original accuracy/latency unchanged."
    csv_path = folder/"summary_coco_comparison.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    matched = 0
    for row in rows:
        if row[fields[0]] == result["model_name"]:
            row["GFLOPs"] = f"{value:.2f}"
            matched += 1
    if matched != 1:
        raise ValueError("Expected exactly one matching summary row")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    md_path = folder/"summary_coco_comparison.md"
    lines = md_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.startswith("| "+result["model_name"]+" |"):
            cells = line.split("|")
            cells[fields.index("GFLOPs")+1] = f" {value:.2f} "
            lines[index] = "|".join(cells)
    marker = "<!-- complexity-estimate -->"
    text = "\n".join(lines)
    if marker in text:
        text = text.split(marker)[0].rstrip()
    text += ("\n\n"+marker+"\nGFLOPs: **ước lượng**, input 1×3×640×640, eval FP32; "
             "1 FMA = 1 FLOP theo fvcore. Bao gồm DARM/DAFD/UFFR/MFA và FFT "
             "ước lượng; không còn operator chưa có bộ đếm trong trace này. "
             "Không gồm resize ảnh, hậu xử lý detection và đọc đĩa. "
             "Xem `complexity_profile.json` để biết công thức và phạm vi; "
             "không so trực tiếp với bảng dùng 2 FLOP/MAC.\n")
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path.write_text(output.getvalue(), encoding="utf-8-sig", newline="")
    md_path.write_text(text, encoding="utf-8")
    return {p.name: p.read_bytes() for p in (result_path, csv_path, md_path, folder/"complexity_profile.json")}
