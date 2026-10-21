"""Read-only audit of the three downloaded DQK Modal COCO annotation files."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    config = json.loads((ROOT.parent / 'Model_Improvement/configs/models_config.json').read_text(encoding='utf-8'))
    report = {
        'source_volume': 'food-detection-training',
        'source_directory': '/datasets/completed-project-5/coco',
        'scope': 'Annotation contents only; image bytes and cloud checkpoint not validated here',
        'splits': {},
    }
    filenames = {}
    for split in ('train', 'valid', 'test'):
        path = ROOT / 'cloud_audit' / f'{split}_annotations.coco.json'
        raw = path.read_bytes()
        data = json.loads(raw)
        images = {int(image['id']): image for image in data['images']}
        categories = sorted(data['categories'], key=lambda category: int(category['id']))
        counts = Counter(int(ann['category_id']) for ann in data['annotations'])
        cat_ids = {int(category['id']) for category in categories}
        errors = Counter()
        for ann in data['annotations']:
            if int(ann['image_id']) not in images:
                errors['unknown_image_id'] += 1
                continue
            if int(ann['category_id']) not in cat_ids:
                errors['unknown_category_id'] += 1
            bbox = ann.get('bbox', [])
            if len(bbox) != 4 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in bbox):
                errors['invalid_bbox'] += 1
                continue
            x, y, width, height = bbox
            if width <= 0 or height <= 0:
                errors['nonpositive_bbox'] += 1
            image = images[int(ann['image_id'])]
            if x < -0.01 or y < -0.01 or x + width > image['width'] + 0.01 or y + height > image['height'] + 0.01:
                errors['bbox_outside_image'] += 1
        ann_ids = [ann['id'] for ann in data['annotations']]
        image_names = [image['file_name'] for image in data['images']]
        filenames[split] = set(image_names)
        foreground = [category for category in categories if category['name'] in config['classes']]
        report['splits'][split] = {
            'annotation_sha256': hashlib.sha256(raw).hexdigest(),
            'images': len(data['images']),
            'annotations': len(data['annotations']),
            'declared_categories': len(categories),
            'annotated_categories': len(counts),
            'configured_foreground_categories': len(foreground),
            'class_order_matches_config': [category['name'] for category in foreground] == config['classes'],
            'categories': [dict(category, annotation_count=counts[int(category['id'])]) for category in categories],
            'duplicate_image_ids': len(data['images']) - len(images),
            'duplicate_annotation_ids': len(ann_ids) - len(set(ann_ids)),
            'duplicate_filenames': len(image_names) - len(set(image_names)),
            'annotation_errors': dict(errors),
        }
    report['exact_filename_overlap_between_splits'] = {
        f'{left}/{right}': len(filenames[left] & filenames[right])
        for left, right in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]
    }
    old_result = ROOT.parent / 'Model_Improvement/Results/RESULT_20261019_235139/result_RFDETR_DQK_Enhanced.json'
    result = json.loads(old_result.read_text(encoding='utf-8'))
    report['test_annotation_hash_matches_saved_result'] = (
        report['splits']['test']['annotation_sha256'] == result['protocol']['annotation_sha256']
    )
    output = ROOT / 'DATASET_CLOUD_AUDIT_20261021.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for split, summary in report['splits'].items():
        print(split, {key: value for key, value in summary.items() if key != 'categories'})
    print('filename_overlap', report['exact_filename_overlap_between_splits'])
    print('saved_result_test_hash_match', report['test_annotation_hash_matches_saved_result'])


if __name__ == '__main__':
    main()
