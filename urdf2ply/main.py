from pathlib import Path
from yourdfpy import URDF
import trimesh
import numpy as np
import csv

# --- Constants --- #
POINT_SAMPLES = 50000
USE_COLLISION = False

COLOR_MODE = "custom"  # 또는 "random"
CSV_PATH = "parol6/urdf/PAROL6.csv"

CUSTOM_COLORS = {
    "base_link": [100, 100, 100],
    "L1": [255, 100, 100],
    "L2": [100, 255, 100],
    "L3": [100, 100, 255],
    "L4": [255, 255, 100],
    "L5": [255, 100, 255],
    "L6": [100, 255, 255],
}


def load_colors_from_csv(csv_path):
    colors = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                link_name = row['Link Name']
                r = float(row['Color Red']) * 255
                g = float(row['Color Green']) * 255
                b = float(row['Color Blue']) * 255
                colors[link_name] = np.array([r, g, b], dtype=np.uint8)
                print(f"  [DEBUG] {link_name}: RGB({r:.1f}, {g:.1f}, {b:.1f})")
    except Exception as e:
        print(f"[WARN] Failed to load CSV colors: {e}")
        import traceback
        traceback.print_exc()
    return colors


def get_color_for_link(link_name, csv_colors=None):
    if COLOR_MODE == "csv":
        if csv_colors and link_name in csv_colors:
            return csv_colors[link_name]
        else:
            return np.array([150, 150, 150], dtype=np.uint8)
    
    elif COLOR_MODE == "custom":
        if link_name in CUSTOM_COLORS:
            return np.array(CUSTOM_COLORS[link_name], dtype=np.uint8)
        else:
            return np.array([150, 150, 150], dtype=np.uint8)
    
    elif COLOR_MODE == "random":
        np.random.seed(hash(link_name) % (2**32))
        color = np.random.randint(50, 256, size=3, dtype=np.uint8)
        return color
    
    return np.array([192, 192, 192], dtype=np.uint8)


def main():
    root_dir = Path(__file__).resolve().parent
    urdf_path = root_dir / "parol6" / "urdf" / "PAROL6.urdf"
    mesh_root = root_dir / "parol6"
    output_dir = root_dir / "output"
    output_dir.mkdir(exist_ok=True)

    print(f"[INFO] Loading URDF: {urdf_path}")
    
    try:
        robot = URDF.load(str(urdf_path))
    except Exception as e:
        print(f"[ERROR] Failed to load URDF: {e}")
        return

    print(f"[INFO] Robot loaded successfully")
    print(f"[INFO] Color mode: {COLOR_MODE}")

    csv_colors = None
    if COLOR_MODE == "csv":
        csv_path_full = root_dir / CSV_PATH
        print(f"[INFO] Loading colors from CSV: {csv_path_full}")
        csv_colors = load_colors_from_csv(csv_path_full)
        print(f"[INFO] Loaded colors for: {list(csv_colors.keys())}")

    # 관절 각도 설정
    if len(robot.actuated_joint_names) > 0:
        cfg = np.array([0.0, 0.5, -0.5, 0.0, 0.0, 0.0])
        robot.update_cfg(cfg)
        print(f"[INFO] Joint configuration: {cfg}")

    all_points = []
    all_colors = []

    # ViserUrdf 방식: link_map을 순회
    for link_name, link in robot.link_map.items():
        print(f"\n[INFO] Processing link: {link_name}")
        
        visuals = link.visuals if not USE_COLLISION else link.collisions
        
        if not visuals:
            print(f"  → No visual geometry")
            continue

        for idx, vis in enumerate(visuals):
            geom = vis.geometry
            if geom.mesh is None:
                print(f"  → No mesh geometry")
                continue

            # 메쉬 파일 경로
            mesh_filename = geom.mesh.filename
            if mesh_filename.startswith("../"):
                mesh_filename = mesh_filename[3:]
            mesh_path = (mesh_root / mesh_filename).resolve()

            print(f"  → Loading mesh: {mesh_path.name}")

            if not mesh_path.exists():
                print(f"  [WARN] Missing: {mesh_path}")
                continue

            try:
                # STL 로드
                mesh = trimesh.load_mesh(str(mesh_path), force='mesh')
                
                if not isinstance(mesh, trimesh.Trimesh):
                    print(f"  [WARN] Not a Trimesh")
                    continue

                print(f"  → Loaded: {len(mesh.vertices)} vertices")

                # 링크의 world transform 가져오기 (cfg 반영됨)
                link_transform = robot.get_transform(link_name)
                
                # visual origin 적용
                if vis.origin is not None:
                    visual_transform = vis.origin
                    combined_transform = link_transform @ visual_transform
                else:
                    combined_transform = link_transform

                # mesh scale 적용
                if geom.mesh.scale is not None:
                    scale_matrix = np.eye(4)
                    scale_matrix[:3, :3] *= geom.mesh.scale
                    combined_transform = combined_transform @ scale_matrix

                print(f"  → Transform applied")

                # 메쉬 변환
                mesh_transformed = mesh.copy()
                mesh_transformed.apply_transform(combined_transform)
                
                print(f"  → Center: {mesh_transformed.centroid}")

                # 색상
                link_color = get_color_for_link(link_name, csv_colors)
                print(f"  → Color: RGB{tuple(link_color)}")

                # 샘플링
                print(f"  → Sampling {POINT_SAMPLES} points...")
                points = mesh_transformed.sample(POINT_SAMPLES)
                colors = np.tile(link_color, (len(points), 1))

                # 개별 저장
                output_filename = f"{link_name}_{idx}.ply" if len(visuals) > 1 else f"{link_name}.ply"
                output_path = output_dir / output_filename
                
                point_cloud = trimesh.PointCloud(points, colors=colors)
                point_cloud.export(str(output_path))
                
                print(f"  ✓ Saved: {output_path}")

                all_points.append(points)
                all_colors.append(colors)

            except Exception as e:
                print(f"  [ERROR] {e}")
                import traceback
                traceback.print_exc()
                continue

    # 전체 로봇
    if all_points:
        print(f"\n[INFO] Creating combined robot...")
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors)
        
        print(f"  → Bounds:")
        print(f"    X: [{combined_points[:,0].min():.3f}, {combined_points[:,0].max():.3f}]")
        print(f"    Y: [{combined_points[:,1].min():.3f}, {combined_points[:,1].max():.3f}]")
        print(f"    Z: [{combined_points[:,2].min():.3f}, {combined_points[:,2].max():.3f}]")
        
        combined_cloud = trimesh.PointCloud(combined_points, colors=combined_colors)
        combined_path = output_dir / "robot_complete.ply"
        combined_cloud.export(str(combined_path))
        
        print(f"✓ Saved: {combined_path}")
        print(f"  Total points: {len(combined_points)}")
    else:
        print(f"\n[WARN] No points generated")

    print(f"\n[DONE] Files in: {output_dir}")


if __name__ == "__main__":
    main()