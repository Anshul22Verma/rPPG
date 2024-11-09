config = {
    # CACHE DIRECTORY --> to save pre-processing directory
    "CACHE_DIR": ".//_cache_",
    # DATA SELECTION --> dataset + configuration for split
    "DATA_SELECTION": {
        
        "NAME": "V4V",
        "FS": 30,  # Used for metric computation
        "DF_LOC": "<path to dataframe after creation>",
        "SPLIT_BY": "sex",
        "TRAIN_SPLIT": "M", 
        "CONDITION": "=="
    },
    # PRE-PROCESSING --> pre-processing configuration
    # PRE-PROCESSING face detection configuration
    "CROP_FACE": {
        "DO_CROP_FACE": True,
        "BACKEND": "",
        "USE_LARGE_FACE_BOX": True,
        "LARGE_BOX_COEF": 0.1,
        "DO_DYNAMIC_DETECTION": True,
        "DYNAMIC_DETECTION_FREQUENCY": 1,
        "USE_MEDIAN_FACE_BOX": False,
    },
    # PRE-PROCESSING resizing frame with face dimension
    "RESIZE": {
        "H": 128,
        "W": 128
    },
    # 
    "DATA_TYPE": ["DiffNormalized", "Standardized"],
    "LABEL_TYPE": ["DiffNormalized"],
    "DO_CHUNK": True,
    "CHUNK_LENGTH": 100,


}