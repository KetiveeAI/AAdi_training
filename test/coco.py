import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone.utils.data.importers import COCODetectionDatasetImporter as load_coco_annotations

# Paths
dataset_dir = "data/coco/images"  # folder with images
json_path = "/data/coco/annotations.json"  # COCO annotation file

# Create a new FiftyOne dataset
dataset = fo.Dataset(name="coco_custom")

# Load COCO annotations
load_coco_annotations(
    dataset,
    json_path=json_path,
    images_dir=dataset_dir,
    label_field="ground_truth",
    include_id=True
)

# Launch FiftyOne app
session = fo.launch_app(dataset)
