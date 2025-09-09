
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def obstacle_pose_orientation_from_pointcloud(
    env,
    cam_names=("RGB_D1_camera",),
    max_obstacles=1,
    z_threshold=0.1,  # ✅ 테이블 제거
) -> torch.Tensor:
    cam1 = env.scene[cam_names[0]]
    num_envs = env.num_envs

    if "obstacle_pose" not in env.extras:
        env.extras["obstacle_pose"] = torch.zeros((num_envs, 3), device=env.device)
    if "obstacle_orientation" not in env.extras:
        env.extras["obstacle_orientation"] = torch.zeros((num_envs, 3), device=env.device)

    results = []

    for i in range(num_envs):
        depth1 = cam1.data.output["distance_to_image_plane"][i, ..., 0]
        pc1 = create_pointcloud_from_depth(
            cam1.data.intrinsic_matrices[i],
            depth1,
            cam1.data.pos_w[i],
            cam1.data.quat_w_ros[i],
            device=env.device,
        )

        if pc1 is None or pc1.shape[0] == 0:
            results.append(torch.zeros(6, device=env.device))
            continue

        # ✅ 테이블 제거
        pc1 = pc1[pc1[:, 2] > z_threshold]

        pc_np = pc1.detach().cpu().numpy()

        try:
            labels = DBSCAN(eps=0.03, min_samples=10).fit(pc_np).labels_
            centers = []
            directions = []

            for label in set(labels):
                if label == -1:
                    continue
                cluster_np = pc_np[labels == label]
                if cluster_np.shape[0] < 10:
                    continue

                center = cluster_np.mean(axis=0)
                pca = PCA(n_components=3)
                pca.fit(cluster_np)
                direction = pca.components_[0]

                centers.append(torch.tensor(center, device=env.device))
                directions.append(torch.tensor(direction, device=env.device))

            if centers:
                env.extras["obstacle_pose"][i] = centers[0]
                env.extras["obstacle_orientation"][i] = directions[0]
                results.append(torch.cat([centers[0], directions[0]]))
            else:
                results.append(torch.zeros(6, device=env.device))

        except Exception:
            results.append(torch.zeros(6, device=env.device))

    return torch.stack(results)




def obstacle_pose_dir_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    pose = env.extras.get("obstacle_pose", torch.zeros((env.num_envs, 3), device=env.device))
    direction = env.extras.get("obstacle_orientation", torch.zeros((env.num_envs, 3), device=env.device))
    size = env.extras.get("obstacle_size", torch.ones((env.num_envs, 2), device=env.device))  # (r, h) 예시
    return torch.cat([pose, direction, size], dim=-1)





