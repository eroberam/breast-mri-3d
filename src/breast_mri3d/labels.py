"""Independent binary labels. Anatomy may overlap a proposed lesion."""

LABELS = ("left_breast", "right_breast", "fibroglandular", "vessels", "lesion")
ENVELOPE = "breast_envelope"
REFERENCE_LABELS = (*LABELS, ENVELOPE)
DISPLAY = {
    ENVELOPE: ("Breast envelope", (239, 178, 160, 115)),
    "left_breast": ("Left breast", (238, 175, 161, 120)),
    "right_breast": ("Right breast", (231, 164, 153, 120)),
    "fibroglandular": ("Fibroglandular tissue", (98, 171, 216, 240)),
    "vessels": ("Vessels (auxiliary)", (176, 92, 200, 255)),
    "lesion": ("Proposed lesion", (214, 54, 80, 255)),
}
