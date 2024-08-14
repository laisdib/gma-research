OPENPOSE = {
    "Nose": 0,
    "Neck": 1,
    "RShoulder": 2,
    "RElbow": 3,
    "RWrist": 4,
    "LShoulder": 5,
    "LElbow": 6,
    "LWrist": 7,
    "RHip": 8,
    "RKnee": 9,
    "RAnkle": 10,
    "LHip": 11,
    "LKnee": 12,
    "LAnkle": 13,
    "REye": 14,
    "LEye": 15,
    "REar": 16,
    "LEar": 17,
    "Background": 18
}

OPENPOSE_PARENT_JOINT_PAIR = [
    (1, 2),  # Neck -> RShoulder
    (2, 3),  # RShoulder -> RElbow
    (3, 4),  # RElbow -> RWrist
    (1, 5),  # Neck -> LShoulder
    (5, 6),  # LShoulder -> LElbow
    (6, 7),  # LElbow -> LWrist
    (8, 9),  # RHip -> RKnee
    (9, 10),  # RKnee -> RAnkle
    (11, 12),  # LHip -> LKnee
    (12, 13)  # LKnee -> LAnkle
]
