"""Video prototype pipeline (Phases V1-V9).

Motion-aware keyframe selection -> fast screening -> Cellpose analysis
(reusing the frozen static modules) -> monolayer features and prototype
score -> temporal smoothing -> annotated video + results CSVs.
"""
