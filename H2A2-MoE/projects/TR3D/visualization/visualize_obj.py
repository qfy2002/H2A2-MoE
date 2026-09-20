#!/usr/bin/env python
"""Open a TR3D-exported OBJ scene on a machine with Open3D and a display."""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize TR3D OBJ output.')
    parser.add_argument('scene_dir', type=Path)
    parser.add_argument('--point-size', type=float, default=3.5)
    parser.add_argument(
        '--box-radius', type=float, default=0.012,
        help='Detection-box edge half-width/radius in scene units.')
    parser.add_argument(
        '--box-style', choices=['square', 'cylinder', 'line'],
        default='square')
    parser.add_argument('--width', type=int, default=1600)
    parser.add_argument('--height', type=int, default=1000)
    parser.add_argument(
        '--remove-top', type=float, default=0.0,
        help='Remove this fraction of the scene height from the point cloud.')
    parser.add_argument(
        '--remove-wall', choices=['x-min', 'x-max', 'y-min', 'y-max'],
        help='Remove points near one room boundary while retaining the floor.')
    parser.add_argument(
        '--wall-thickness', type=float, default=0.15,
        help='Boundary thickness removed by --remove-wall, in scene units.')
    parser.add_argument(
        '--crop-to-gt', action='store_true',
        help='Crop points to the GT-box extent plus --crop-margin.')
    parser.add_argument(
        '--crop-margin', type=float, default=0.8,
        help='Context retained around GT boxes when --crop-to-gt is used.')
    parser.add_argument(
        '--denoise', action='store_true',
        help='Remove sparse statistical outliers from the point cloud.')
    parser.add_argument(
        '--denoise-neighbors', type=int, default=20,
        help='Neighbor count used by statistical outlier removal.')
    parser.add_argument(
        '--denoise-std-ratio', type=float, default=1.5,
        help='Outlier distance threshold in neighborhood standard deviations.')
    parser.add_argument('--hide-gt', action='store_true')
    parser.add_argument('--hide-pred', action='store_true')
    parser.add_argument(
        '--pred-file', type=Path,
        help='Prediction OBJ to use instead of the default *_pred.obj.')
    parser.add_argument(
        '--overlay', action='store_true',
        help='Overlay GT and predictions instead of showing them side by side.')
    parser.add_argument(
        '--paper-output', type=Path,
        help='Interactively choose a predicted-result view and save it.')
    parser.add_argument(
        '--paper-mode', choices=['points', 'pred', 'gt', 'overlay'],
        default='pred',
        help='Boxes shown when --paper-output is enabled.')
    parser.add_argument(
        '--camera-json', type=Path,
        help='Load this camera if it exists and save the current camera when '
             'the paper image is saved.')
    return parser.parse_args()


def find_scene_file(scene_dir: Path, suffix: str) -> Path:
    files = list(scene_dir.glob(f'*_{suffix}.obj'))
    if len(files) != 1:
        raise FileNotFoundError(
            f'Expected one *_{suffix}.obj in {scene_dir}, found {len(files)}')
    return files[0]


def resolve_scene_file(scene_dir: Path, path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = scene_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def contains_faces(path: Path) -> bool:
    """Detection box OBJ files contain faces after their first vertices."""
    with path.open() as file:
        for index, line in enumerate(file):
            if line.startswith('f '):
                return True
            if index >= 63:
                return False
    return False


def load_points(path: Path):
    # Segmentation OBJ files can contain millions of points. NumPy is much
    # faster than parsing those files line by line in Python.
    values = np.loadtxt(
        path, dtype=np.float32, usecols=(1, 2, 3, 4, 5, 6), ndmin=2)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(values[:, :3])
    cloud.colors = o3d.utility.Vector3dVector(
        np.clip(values[:, 3:6] / 255.0, 0, 1))
    return cloud


def remove_top_points(cloud, fraction):
    if fraction == 0:
        return cloud
    if not 0 < fraction < 1:
        raise ValueError('--remove-top must be in [0, 1).')
    points = np.asarray(cloud.points)
    z_min, z_max = float(points[:, 2].min()), float(points[:, 2].max())
    z_limit = z_max - (z_max - z_min) * fraction
    keep = np.flatnonzero(points[:, 2] <= z_limit)
    cropped = cloud.select_by_index(keep.tolist())
    print(f'Removed {len(points) - len(keep)} top points above z={z_limit:.3f}')
    return cropped


def remove_wall_points(cloud, side, thickness):
    if side is None:
        return cloud
    if thickness <= 0:
        raise ValueError('--wall-thickness must be positive.')
    points = np.asarray(cloud.points)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    axis = 0 if side.startswith('x-') else 1
    if side.endswith('min'):
        wall = points[:, axis] <= minimum[axis] + thickness
    else:
        wall = points[:, axis] >= maximum[axis] - thickness
    # Preserve the floor strip so the room footprint remains readable.
    wall &= points[:, 2] > minimum[2] + 0.06
    keep = np.flatnonzero(~wall)
    cropped = cloud.select_by_index(keep.tolist())
    print(f'Removed {int(wall.sum())} points from wall {side} '
          f'(thickness={thickness:.3f})')
    return cropped


def crop_points_to_boxes(cloud, boxes_path, margin):
    if margin < 0:
        raise ValueError('--crop-margin must be non-negative.')
    vertices = []
    with boxes_path.open() as input_file:
        for line in input_file:
            if line.startswith('v '):
                vertices.append([float(value) for value in line.split()[1:4]])
    if not vertices:
        raise ValueError(f'No box vertices found in {boxes_path}.')
    vertices = np.asarray(vertices, dtype=np.float64)
    lower = vertices.min(axis=0) - margin
    upper = vertices.max(axis=0) + margin
    points = np.asarray(cloud.points)
    keep_mask = np.all((points >= lower) & (points <= upper), axis=1)
    keep = np.flatnonzero(keep_mask)
    cropped = cloud.select_by_index(keep.tolist())
    print(f'Cropped {len(points) - len(keep)} points outside GT extent '
          f'(margin={margin:.3f}); retained {len(keep)} points')
    return cropped


def remove_statistical_outliers(cloud, neighbors, std_ratio):
    if neighbors < 2:
        raise ValueError('--denoise-neighbors must be at least 2.')
    if std_ratio <= 0:
        raise ValueError('--denoise-std-ratio must be positive.')
    _, keep = cloud.remove_statistical_outlier(
        nb_neighbors=neighbors, std_ratio=std_ratio)
    filtered = cloud.select_by_index(keep)
    print(f'Removed {len(cloud.points) - len(filtered.points)} statistical '
          f'outliers; retained {len(filtered.points)} points '
          f'(neighbors={neighbors}, std_ratio={std_ratio:.2f})')
    return filtered


def filter_points(cloud, args):
    cloud = remove_top_points(cloud, args.remove_top)
    if args.denoise:
        cloud = remove_statistical_outliers(
            cloud, args.denoise_neighbors, args.denoise_std_ratio)
    return remove_wall_points(cloud, args.remove_wall, args.wall_thickness)


def _edge_mesh(start, end, radius, color, style):
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 1e-8:
        return None
    if style == 'cylinder':
        edge_mesh = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=length, resolution=10, split=1)
    else:
        width = radius * 2
        edge_mesh = o3d.geometry.TriangleMesh.create_box(
            width=width, height=width, depth=length)
        edge_mesh.translate((-radius, -radius, -length * 0.5))
    edge_mesh.compute_vertex_normals()
    unit_direction = direction / length
    z_axis = np.asarray([0.0, 0.0, 1.0])
    cosine = float(np.clip(np.dot(z_axis, unit_direction), -1.0, 1.0))
    if cosine < 1.0 - 1e-8:
        if cosine <= -1.0 + 1e-8:
            rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(
                np.asarray([np.pi, 0.0, 0.0]))
        else:
            axis = np.cross(z_axis, unit_direction)
            axis /= np.linalg.norm(axis)
            angle = np.arccos(cosine)
            rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(
                axis * angle)
        edge_mesh.rotate(rotation, center=(0.0, 0.0, 0.0))
    edge_mesh.translate((start + end) * 0.5)
    edge_mesh.paint_uniform_color(color)
    return edge_mesh


def load_boxes(path: Path, fallback_color, radius=0.0, style='line'):
    vertices, vertex_colors, edges = [], [], set()
    with path.open() as file:
        for line in file:
            if line.startswith('v '):
                values = [float(value) for value in line.split()[1:]]
                vertices.append(values[:3])
                if len(values) >= 6:
                    vertex_colors.append(
                        np.clip(np.asarray(values[3:6]) / 255.0, 0, 1))
                else:
                    vertex_colors.append(np.asarray(fallback_color))
            elif line.startswith('f '):
                face = [int(value.split('/')[0]) - 1
                        for value in line.split()[1:]]
                for start, end in zip(face, face[1:] + face[:1]):
                    edges.add(tuple(sorted((start, end))))
    if not vertices or not edges:
        return None
    vertices = np.asarray(vertices)
    sorted_edges = np.asarray(sorted(edges))
    vertex_colors = np.asarray(vertex_colors)
    edge_colors = (vertex_colors[sorted_edges[:, 0]] +
                   vertex_colors[sorted_edges[:, 1]]) * 0.5
    if style != 'line' and radius > 0:
        boxes = o3d.geometry.TriangleMesh()
        for edge, color in zip(sorted_edges, edge_colors):
            edge_mesh = _edge_mesh(
                vertices[edge[0]], vertices[edge[1]], radius, color, style)
            if edge_mesh is not None:
                boxes += edge_mesh
        return boxes

    boxes = o3d.geometry.LineSet()
    boxes.points = o3d.utility.Vector3dVector(vertices)
    boxes.lines = o3d.utility.Vector2iVector(sorted_edges)
    boxes.colors = o3d.utility.Vector3dVector(edge_colors)
    return boxes


def _capture_rgb(visualizer):
    image = np.asarray(
        visualizer.capture_screen_float_buffer(do_render=True))
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def _crop_render(image, background=245, margin=16):
    foreground = np.any(np.abs(image.astype(np.int16) - background) > 8,
                        axis=2)
    rows, columns = np.where(foreground)
    if not len(rows):
        return image
    top = max(int(rows.min()) - margin, 0)
    bottom = min(int(rows.max()) + margin + 1, image.shape[0])
    left = max(int(columns.min()) - margin, 0)
    right = min(int(columns.max()) + margin + 1, image.shape[1])
    return image[top:bottom, left:right]


def _save_prediction_render(image, output_path):
    from PIL import Image

    image = _crop_render(image)
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path, dpi=(300, 300))
    print(f'Predicted-result figure saved: {output_path}')


def show_detection_paper_view(points, gt, pred, args, scene_name):
    if args.paper_mode in ('gt', 'overlay') and gt is None:
        raise ValueError(f'Paper mode {args.paper_mode} requires GT boxes.')
    if args.paper_mode in ('pred', 'overlay') and pred is None:
        raise ValueError(
            f'Paper mode {args.paper_mode} requires prediction boxes.')

    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window(
        f'{scene_name} {args.paper_mode} - press S, P, or Space to save',
        width=args.width, height=args.height)
    visualizer.add_geometry(points)
    if args.paper_mode in ('gt', 'overlay'):
        visualizer.add_geometry(gt)
    if args.paper_mode in ('pred', 'overlay'):
        visualizer.add_geometry(pred)
    options = visualizer.get_render_option()
    options.background_color = np.asarray([0.96, 0.96, 0.96])
    options.point_size = args.point_size
    options.line_width = 2.0
    view_control = visualizer.get_view_control()
    camera_path = None
    if args.camera_json is not None:
        camera_path = args.camera_json.expanduser().resolve()
        if camera_path.is_file():
            camera = o3d.io.read_pinhole_camera_parameters(str(camera_path))
            view_control.convert_from_pinhole_camera_parameters(
                camera, allow_arbitrary=True)
            print(f'Loaded camera: {camera_path}')

    def save_prediction(vis):
        if camera_path is not None:
            camera_path.parent.mkdir(parents=True, exist_ok=True)
            camera = vis.get_view_control().convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters(str(camera_path), camera)
            print(f'Camera saved: {camera_path}')
        _save_prediction_render(_capture_rgb(vis), args.paper_output)
        return False

    for key in (ord('S'), ord('P'), ord(' ')):
        visualizer.register_key_callback(key, save_prediction)
    print(f'Paper {args.paper_mode} mode: click the window, adjust the view, '
          'then press S, P, or Space to save.')
    visualizer.run()
    visualizer.destroy_window()


def show_segmentation_paper_view(cloud, args, scene_name):
    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window(
        f'{scene_name} {args.paper_mode} - press S, P, or Space to save',
        width=args.width, height=args.height)
    visualizer.add_geometry(cloud)
    options = visualizer.get_render_option()
    options.background_color = np.asarray([0.96, 0.96, 0.96])
    options.point_size = args.point_size
    view_control = visualizer.get_view_control()
    camera_path = None
    if args.camera_json is not None:
        camera_path = args.camera_json.expanduser().resolve()
        if camera_path.is_file():
            camera = o3d.io.read_pinhole_camera_parameters(str(camera_path))
            view_control.convert_from_pinhole_camera_parameters(
                camera, allow_arbitrary=True)
            print(f'Loaded camera: {camera_path}')

    def save_render(vis):
        if camera_path is not None:
            camera_path.parent.mkdir(parents=True, exist_ok=True)
            camera = vis.get_view_control().convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters(str(camera_path), camera)
            print(f'Camera saved: {camera_path}')
        _save_prediction_render(_capture_rgb(vis), args.paper_output)
        return False

    for key in (ord('S'), ord('P'), ord(' ')):
        visualizer.register_key_callback(key, save_render)
    print(f'Segmentation {args.paper_mode} mode: click the window, adjust the '
          'view, then press S, P, or Space to save.')
    visualizer.run()
    visualizer.destroy_window()


def main():
    args = parse_args()
    scene_dir = args.scene_dir.expanduser().resolve()
    points_path = find_scene_file(scene_dir, 'points')
    gt_path = find_scene_file(scene_dir, 'gt')
    pred_path = (resolve_scene_file(scene_dir, args.pred_file)
                 if args.pred_file is not None
                 else find_scene_file(scene_dir, 'pred'))
    is_detection = contains_faces(gt_path)
    points = load_points(points_path)
    if is_detection:
        points = remove_top_points(points, args.remove_top)
        if args.crop_to_gt:
            points = crop_points_to_boxes(points, gt_path, args.crop_margin)
        if args.denoise:
            points = remove_statistical_outliers(
                points, args.denoise_neighbors, args.denoise_std_ratio)
        points = remove_wall_points(
            points, args.remove_wall, args.wall_thickness)
    else:
        points = filter_points(points, args)
    geometries = [points]

    if is_detection:
        gt = (None if args.hide_gt else
              load_boxes(gt_path, (0.0, 0.2, 1.0), args.box_radius,
                         args.box_style))
        pred = (None if args.hide_pred else
                load_boxes(pred_path, (1.0, 0.1, 0.0), args.box_radius,
                           args.box_style))
        if args.paper_output is not None:
            show_detection_paper_view(
                points, gt, pred, args, scene_dir.name)
            return
        if args.overlay:
            print('Detection overlay: RGB points + GT + predictions')
            if gt is not None:
                geometries.append(gt)
            if pred is not None:
                geometries.append(pred)
        else:
            print('Detection side by side (left to right): GT | prediction')
            bounds = points.get_axis_aligned_bounding_box()
            offset = max(float(bounds.get_extent()[0]) * 1.15, 1.0)
            pred_points = o3d.geometry.PointCloud(points)
            pred_points.translate((offset, 0, 0))
            geometries.append(pred_points)
            if gt is not None:
                geometries.append(gt)
            if pred is not None:
                pred.translate((offset, 0, 0))
                geometries.append(pred)
    else:
        if args.crop_to_gt:
            raise ValueError('--crop-to-gt is only available for detection.')
        if args.paper_output is not None:
            if args.paper_mode == 'overlay':
                raise ValueError(
                    'Segmentation paper mode does not support overlay.')
            paper_paths = dict(
                points=points_path, gt=gt_path, pred=pred_path)
            cloud = load_points(paper_paths[args.paper_mode])
            cloud = filter_points(cloud, args)
            show_segmentation_paper_view(
                cloud, args, scene_dir.name)
            return
        print('Segmentation mode (left to right): RGB points | GT | prediction')
        bounds = points.get_axis_aligned_bounding_box()
        offset = max(float(bounds.get_extent()[0]) * 1.1, 1.0)
        if not args.hide_gt:
            gt = filter_points(load_points(gt_path), args)
            gt.translate((offset, 0, 0))
            geometries.append(gt)
        if not args.hide_pred:
            pred = filter_points(load_points(pred_path), args)
            pred.translate((offset * 2, 0, 0))
            geometries.append(pred)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(
        scene_dir.name, width=args.width, height=args.height)
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    options = visualizer.get_render_option()
    options.background_color = np.asarray([0.96, 0.96, 0.96])
    options.point_size = args.point_size
    options.line_width = 2.0
    visualizer.run()
    visualizer.destroy_window()


if __name__ == '__main__':
    main()
