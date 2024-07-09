DEBUG_MODE = False
RESCALE = True
IMAGE_SIZE = [1024, 1024]
STAFF_SIZE = [1408, 192]
SYMBOL_SIZE = [224, 224]
OUTPUT_FOLDERS = {
    'humdrum_output': 'humdrum_output',
    'mei_output': 'mei_output',
    'pdf_images': 'pdf_images',
    'pdf_temp': 'pdf_temp',
    'preprocessed_images': 'preprocessed_images'
}
MODEL_PATHES = {
    'staffs': 'models/object_detection/best_staff.onnx',
    'symbols': 'models/object_detection/best_symbols.onnx',
    'all_symbols': 'models/classification/best_all_symbols.onnx',
    'clef': 'models/classification/best_clef.onnx',
    'mensuration': 'models/classification/best_mensuration.onnx'
}
LABEL_PATHES = {
    'staffs': 'models/object_detection/best_staff.yaml',
    'symbols': 'models/object_detection/best_symbols.yaml',
    'all_symbols': 'models/classification/best_all_symbols.yaml',
    'clef': 'models/classification/best_clef.yaml',
    'mensuration': 'models/classification/best_mensuration.yaml'
}
