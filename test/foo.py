import fiftyone as fo
dataset = fo.load_dataset("quickstart")
session = fo.launch_app(dataset)
