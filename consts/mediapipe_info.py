MEDIAPIPE = {
    "nose": 0,
    "left eye (inner)": 1,
    "left eye": 2,
    "left eye (outer)": 3,
    "right eye (inner)": 4,
    "right eye": 5,
    "right eye (outer)": 6,
    "left ear": 7,
    "right ear": 8,
    "mouth (left)": 9,
    "mouth (right)": 10,
    "left shoulder": 11,
    "right shoulder": 12,
    "left elbow": 13,
    "right elbow": 14,
    "left wrist": 15,
    "right wrist": 16,
    "left pinky": 17,
    "right pinky": 18,
    "left index": 19,
    "right index": 20,
    "left thumb": 21,
    "right thumb": 22,
    "left hip": 23,
    "right hip": 24,
    "left knee": 25,
    "right knee": 26,
    "left ankle": 27,
    "right ankle": 28,
    "left heel": 29,
    "right heel": 30,
    "left foot index": 31,
    "right foot index": 32
}

MEDIAPIPE_PARENT_JOINT_PAIR = [
    (11, 13),  # Left shoulder -> Left elbow
    (12, 14),  # Right shoulder -> Right elbow
    (13, 15),  # Left elbow -> Left wrist
    (14, 16),  # Right elbow -> Right wrist
    (15, 17),  # Left wrist -> Left pinky
    (15, 19),  # Left wrist -> Left index
    (15, 21),  # Left wrist -> Left thumb
    (16, 18),  # Right wrist -> Right pinky
    (16, 20),  # Right wrist -> Right index
    (16, 22),  # Right wrist -> Right thumb
    (23, 25),  # Left hip -> Left knee
    (24, 26),  # Right hip -> Right knee
    (25, 27),  # Left knee -> Left ankle
    (26, 28),  # Right knee -> Right ankle
    (27, 29),  # Left ankle -> Left heel
    (28, 30),  # Right ankle -> Right heel
    (29, 31),  # Left heel -> Left foot index
    (30, 32),  # Right heel -> Right foot index
    (0, 1),  # Nose -> Left eye (inner)
    (0, 4),  # Nose -> Right eye (inner)
    (1, 2),  # Left eye (inner) -> Left eye
    (4, 5),  # Right eye (inner) -> Right eye
    (2, 3),  # Left eye -> Left eye (outer)
    (5, 6),  # Right eye -> Right eye (outer)
    (3, 7),  # Left eye (outer) -> Left ear
    (6, 8),  # Right eye (outer) -> Right ear
    (9, 11),  # Mouth (left) -> Left shoulder
    (10, 12),  # Mouth (right) -> Right shoulder
    (11, 23),  # Left shoulder -> Left hip
    (12, 24)  # Right shoulder -> Right hip
]
