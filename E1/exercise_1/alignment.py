""" Procrustes Aligment for point clouds """
import numpy as np
from pathlib import Path


def procrustes_align(pc_x, pc_y):
    """
    calculate the rigid transform to go from point cloud pc_x to point cloud pc_y, assuming points are corresponding
    :param pc_x: Nx3 input point cloud
    :param pc_y: Nx3 target point cloud, corresponding to pc_x locations
    :return: rotation (3, 3) and translation (3,) needed to go from pc_x to pc_y
    """
    R = np.zeros((3, 3), dtype=np.float32)
    t = np.zeros((3,), dtype=np.float32)

    # DONE: Your implementation starts here ###############

    # 1. get centered pc_x and centered pc_y
    mean_x = pc_x.mean(axis=0)
    mean_y = pc_y.mean(axis=0)
    centered_pc_x = pc_x - mean_x
    centered_pc_y = pc_y - mean_y

    # 2. create X and Y both of shape 3XN by reshaping centered pc_x, centered pc_y
    X = centered_pc_x.T
    Y = centered_pc_y.T

    # 3. estimate rotation
    u, x, vt = np.linalg.svd(np.matmul(X, Y.T))
    if np.linalg.det(u) * np.linalg.det(vt.T) < 0:
        vt[-1, :] *= -1
    R = np.matmul(vt.T, u.T)

    # 4. estimate translation
    t = mean_y - np.matmul(R, mean_x)

    # R and t should now contain the rotation (shape 3x3) and translation (shape 3,)
    # Your implementation ends here ###############

    t_broadcast = np.broadcast_to(t[:, np.newaxis], (3, pc_x.shape[0]))
    print('Procrustes Aligment Loss: ', np.abs((np.matmul(R, pc_x.T) + t_broadcast) - pc_y.T).mean())

    return R, t


def load_correspondences():
    """
    loads correspondences between meshes from disk
    """

    load_obj_as_np = lambda path: np.array(list(map(lambda x: list(map(float, x.split(' ')[1:4])), path.read_text().splitlines())))
    path_x = (Path(__file__).parent / "resources" / "points_input.obj").absolute()
    path_y = (Path(__file__).parent / "resources" / "points_target.obj").absolute()
    return load_obj_as_np(path_x), load_obj_as_np(path_y)
