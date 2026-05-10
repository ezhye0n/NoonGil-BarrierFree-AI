from roboflow import Roboflow
rf = Roboflow(api_key="vgvnQD3ktPGMXOXsiCG7")

# 검색어별 대표 데이터셋 (Universe에서 직접 확인 후 교체)
datasets = [
    ("vg2024", "pothole-detect-yolo", 1),
    ("vspace", "curb-xtrmz", 1),
    ("class-obuqr", "stair-x1rgs", 1),
]

for workspace, project_name, version in datasets:
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(version).download("yolov8")