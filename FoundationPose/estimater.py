# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# FoundationPose算法
# 位姿优化
# 位姿跟踪



from Utils import *
from datareader import *
import itertools
from learning.training.predict_score import *
from learning.training.predict_pose_refine import *
import yaml


class FoundationPose:
  def __init__(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer:ScorePredictor=None, refiner:PoseRefinePredictor=None, glctx=None, debug=0, debug_dir='/home/bowen/debug/novel_pose_debug/', registration_views=40, registration_inplane_step=60, registration_topk=0):
    self.gt_pose = None
    self.ignore_normal_flip = True
    self.debug = debug
    self.debug_dir = debug_dir
    # A value of zero preserves the original all-candidates path.  The demo
    # applications pass a small positive value to use coarse-to-fine
    # registration and avoid refining obviously poor initial hypotheses.
    self.registration_topk = max(0, int(registration_topk))
    self.registration_views = max(1, int(registration_views))
    self.registration_inplane_step = int(registration_inplane_step)
    os.makedirs(debug_dir, exist_ok=True)

    self.reset_object(model_pts, model_normals, symmetry_tfs=symmetry_tfs, mesh=mesh)
    self.make_rotation_grid(
      min_n_views=self.registration_views,
      inplane_step=self.registration_inplane_step,
    )
    self.fast_rot_grid = self.rot_grid
    # Build the conservative fallback grid lazily.  Normal startup and the
    # common fast registration path pay no extra initialization cost; a hard
    # view can still request the original-density search on demand.
    self.fallback_registration_views = max(40, self.registration_views)
    self.fallback_registration_inplane_step = min(60, self.registration_inplane_step)
    self.full_rot_grid = (
      self.fast_rot_grid
      if (
        self.fallback_registration_views == self.registration_views
        and self.fallback_registration_inplane_step == self.registration_inplane_step
      )
      else None
    )

    self.glctx = glctx

    if scorer is not None:
      self.scorer = scorer
    else:
      self.scorer = ScorePredictor()

    if refiner is not None:
      self.refiner = refiner
    else:
      self.refiner = PoseRefinePredictor()

    self.pose_last = None   # Used for tracking; per the centered mesh


  def reset_object(self, model_pts, model_normals, symmetry_tfs=None, mesh=None):
    max_xyz = mesh.vertices.max(axis=0)
    min_xyz = mesh.vertices.min(axis=0)
    self.model_center = (min_xyz+max_xyz)/2
    if mesh is not None:
      self.mesh_ori = mesh.copy()
      mesh = mesh.copy()
      mesh.vertices = mesh.vertices - self.model_center.reshape(1,3)

    model_pts = mesh.vertices
    self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
    self.vox_size = max(self.diameter/20.0, 0.003)
    logging.info(f'self.diameter:{self.diameter}, vox_size:{self.vox_size}')
    self.dist_bin = self.vox_size/2
    self.angle_bin = 20  # Deg
    pcd = toOpen3dCloud(model_pts, normals=model_normals)
    pcd = pcd.voxel_down_sample(self.vox_size)
    self.max_xyz = np.asarray(pcd.points).max(axis=0)
    self.min_xyz = np.asarray(pcd.points).min(axis=0)
    self.pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
    self.normals = F.normalize(torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device='cuda'), dim=-1)
    logging.info(f'self.pts:{self.pts.shape}')
    self.mesh_path = None
    self.mesh = mesh
    if self.mesh is not None:
      self.mesh_path = f'/tmp/{uuid.uuid4()}.obj'
      self.mesh.export(self.mesh_path)
    self.mesh_tensors = make_mesh_tensors(self.mesh)

    if symmetry_tfs is None:
      self.symmetry_tfs = torch.eye(4).float().cuda()[None]
    else:
      self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device='cuda', dtype=torch.float)

    logging.info("reset done")



  def get_tf_to_centered_mesh(self):
    tf_to_center = torch.eye(4, dtype=torch.float, device='cuda')
    tf_to_center[:3,3] = -torch.as_tensor(self.model_center, device='cuda', dtype=torch.float)
    return tf_to_center


  def to_device(self, s='cuda:0'):
    for k in self.__dict__:
      self.__dict__[k] = self.__dict__[k]
      if torch.is_tensor(self.__dict__[k]) or isinstance(self.__dict__[k], nn.Module):
        logging.info(f"Moving {k} to device {s}")
        self.__dict__[k] = self.__dict__[k].to(s)
    for k in self.mesh_tensors:
      logging.info(f"Moving {k} to device {s}")
      self.mesh_tensors[k] = self.mesh_tensors[k].to(s)
    if self.refiner is not None:
      self.refiner.model.to(s)
    if self.scorer is not None:
      self.scorer.model.to(s)
    if self.glctx is not None:
      self.glctx = dr.RasterizeCudaContext(s)



  def make_rotation_grid(self, min_n_views=40, inplane_step=60):
    cam_in_obs = sample_views_icosphere(n_views=min_n_views)
    # ``sample_views_icosphere`` intentionally starts at an icosphere with
    # 42 vertices.  For real-time registration we often need fewer viewpoints
    # than that.  Select a deterministic, evenly distributed subset instead of
    # silently keeping all 42 viewpoints.
    if min_n_views < len(cam_in_obs):
      view_ids = np.linspace(0, len(cam_in_obs) - 1, int(min_n_views), dtype=int)
      cam_in_obs = cam_in_obs[view_ids]
    logging.info(f'cam_in_obs:{cam_in_obs.shape}')
    rot_grid = []
    for i in range(len(cam_in_obs)):
      for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
        cam_in_ob = cam_in_obs[i]
        R_inplane = euler_matrix(0,0,inplane_rot)
        cam_in_ob = cam_in_ob@R_inplane
        ob_in_cam = np.linalg.inv(cam_in_ob)
        rot_grid.append(ob_in_cam)

    rot_grid = np.asarray(rot_grid)
    logging.info(f"rot_grid:{rot_grid.shape}")
    rot_grid = mycpp.cluster_poses(30, 99999, rot_grid, self.symmetry_tfs.data.cpu().numpy())
    rot_grid = np.asarray(rot_grid)
    logging.info(f"after cluster, rot_grid:{rot_grid.shape}")
    self.rot_grid = torch.as_tensor(rot_grid, device='cuda', dtype=torch.float)
    logging.info(f"self.rot_grid: {self.rot_grid.shape}")


  def generate_random_pose_hypo(self, K, rgb, depth, mask, scene_pts=None, rot_grid=None):
    '''
    @scene_pts: torch tensor (N,3)
    '''
    ob_in_cams = (self.rot_grid if rot_grid is None else rot_grid).clone()
    center = self.guess_translation(depth=depth, mask=mask, K=K)
    ob_in_cams[:,:3,3] = torch.tensor(center, device='cuda', dtype=torch.float).reshape(1,3)
    return ob_in_cams


  def guess_translation(self, depth, mask, K):
    vs,us = np.where(mask>0)
    if len(us)==0:
      logging.info(f'mask is all zero')
      return np.zeros((3))
    uc = (us.min()+us.max())/2.0
    vc = (vs.min()+vs.max())/2.0
    if torch.is_tensor(depth):
      valid = torch.as_tensor(mask, device=depth.device, dtype=torch.bool) & (depth >= 0.001)
      if not bool(valid.any().item()):
        logging.info(f"valid is empty")
        return np.zeros((3))
      # ``np.median`` averages the two middle samples for an even-sized set;
      # use the matching interpolation mode rather than torch.median (which
      # selects one of them) so the initial translation remains unchanged.
      valid_depths = depth[valid].reshape(-1)
      lower = torch.median(valid_depths)
      if valid_depths.numel() % 2:
        median = lower
      else:
        upper = torch.kthvalue(valid_depths, valid_depths.numel() // 2 + 1).values
        median = (lower + upper) * 0.5
      zc = float(median.item())
    else:
      valid = mask.astype(bool) & (depth>=0.001)
      if not valid.any():
        logging.info(f"valid is empty")
        return np.zeros((3))
      zc = float(np.median(depth[valid]))
    center = (np.linalg.inv(K)@np.asarray([uc,vc,1]).reshape(3,1))*zc

    if self.debug>=2:
      pcd = toOpen3dCloud(center.reshape(1,3))
      o3d.io.write_point_cloud(f'{self.debug_dir}/init_center.ply', pcd)

    return center.reshape(3)


  def register(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=5, force_full=False):
    '''Copmute pose from given pts to self.pcd
    @pts: (N,3) np array, downsampled scene points
    @force_full: use the conservative full-density rotation search
    '''
    # Registration uses a deterministic rotation grid; reseeding every call
    # needlessly touches CUDA's RNG state and adds synchronization overhead.
    logging.info('Welcome')
    self.last_registration_metrics = dict(force_full_requested=bool(force_full),
                                         search_executed=False, full_search=False)

    if self.glctx is None:
      if glctx is None:
        self.glctx = dr.RasterizeCudaContext()
        # self.glctx = dr.RasterizeGLContext()
      else:
        self.glctx = glctx

    # Keep the depth tensor on CUDA throughout filtering and XYZ conversion.
    # The previous NumPy path copied the full 848x480 frame to CUDA for each
    # filter and copied it back before the predictors copied it to CUDA again.
    # This preserves the kernels and values while removing two synchronizing
    # round trips from every registration.
    depth = torch.as_tensor(depth, device='cuda', dtype=torch.float32).contiguous()
    depth = erode_depth(depth, radius=2, device='cuda')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')
    ob_mask_t = torch.as_tensor(ob_mask, device='cuda', dtype=torch.bool)
    valid_object = (depth >= 0.001) & ob_mask_t
    if int(valid_object.sum().item()) < 4:
      self.last_registration_metrics['early_return'] = 'insufficient_valid_depth'
      logging.info(f'valid too small, return')
      pose = np.eye(4)
      pose[:3,3] = self.guess_translation(
        depth=depth.detach().cpu().numpy(), mask=ob_mask, K=K
      )
      return pose

    xyz_map = depth2xyzmap_batch(
      depth[None],
      torch.as_tensor(K, device='cuda', dtype=torch.float32)[None],
      zfar=np.inf,
    )[0]
    normal_map = None

    if self.debug>=2:
      xyz_map_cpu = xyz_map.detach().cpu().numpy()
      valid_cpu = xyz_map_cpu[...,2]>=0.001
      pcd = toOpen3dCloud(xyz_map_cpu[valid_cpu], rgb[valid_cpu])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_raw.ply',pcd)
      cv2.imwrite(f'{self.debug_dir}/ob_mask.png', (ob_mask*255.0).clip(0,255))

    if self.debug>=2:
      imageio.imwrite(f'{self.debug_dir}/color.png', rgb)
      depth_cpu = depth.detach().cpu().numpy()
      cv2.imwrite(f'{self.debug_dir}/depth.png', (depth_cpu*1000).astype(np.uint16))
      valid = xyz_map_cpu[...,2]>=0.001
      pcd = toOpen3dCloud(xyz_map_cpu[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_complete.ply',pcd)

    self.H, self.W = depth.shape[:2]
    self.K = K
    self.ob_id = ob_id
    self.ob_mask = ob_mask

    fast_rot_grid = getattr(self, 'fast_rot_grid', self.rot_grid)
    full_rot_grid = getattr(self, 'full_rot_grid', fast_rot_grid)
    if force_full and full_rot_grid is None:
      # This worker is single-threaded in the autonomous entry point, so the
      # one-time cache fill cannot race with another registration call.
      self.make_rotation_grid(
        min_n_views=self.fallback_registration_views,
        inplane_step=self.fallback_registration_inplane_step,
      )
      full_rot_grid = self.rot_grid
      self.full_rot_grid = full_rot_grid
      self.rot_grid = fast_rot_grid
    active_rot_grid = full_rot_grid if force_full else fast_rot_grid
    active_topk = 0 if force_full else self.registration_topk
    full_search = bool((force_full or (
      self.registration_views == self.fallback_registration_views and
      self.registration_inplane_step == self.fallback_registration_inplane_step
    )) and not (0 < active_topk < len(active_rot_grid)))
    self.last_registration_metrics.update(
      search_executed=True, full_search=full_search,
      search_scope='full_density_all_candidates' if full_search else 'configured_grid',
      rotation_candidates=int(len(active_rot_grid)), requested_topk=int(active_topk),
    )
    poses = self.generate_random_pose_hypo(
      K=K,
      rgb=rgb,
      depth=depth,
      mask=ob_mask,
      scene_pts=None,
      rot_grid=active_rot_grid,
    )
    logging.info(f'poses:{poses.shape}')

    if self.debug >= 2:
      add_errs = self.compute_add_err_to_gt_pose(poses)
      logging.info(f"after viewpoint, add_errs min:{add_errs.min()}")

    # Score the inexpensive initial hypotheses once, then spend the expensive
    # refinement passes only on the best candidates.  The final score below
    # still decides among the refined poses, so this is a latency optimization
    # without changing the final selection model.
    candidate_ids = torch.arange(len(poses), device=poses.device, dtype=torch.long)
    if 0 < active_topk < len(poses):
      coarse_scores, _ = self.scorer.predict(
        mesh=self.mesh,
        rgb=rgb,
        depth=depth,
        K=K,
        ob_in_cams=poses,
        normal_map=normal_map,
        mesh_tensors=self.mesh_tensors,
        glctx=self.glctx,
        mesh_diameter=self.diameter,
        get_vis=False,
      )
      coarse_scores = torch.as_tensor(coarse_scores, device=poses.device, dtype=torch.float).reshape(-1)
      keep = min(active_topk, len(coarse_scores))
      candidate_ids = torch.topk(coarse_scores, k=keep, largest=True, sorted=False).indices
      poses = poses.index_select(0, candidate_ids)
      logging.info(f"coarse registration kept {len(poses)}/{len(coarse_scores)} hypotheses")

    self.last_registration_metrics['refined_candidates'] = int(len(poses))
    poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=poses, normal_map=normal_map, xyz_map=xyz_map, glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration, get_vis=self.debug>=2)
    if vis is not None:
      imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)

    scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K, ob_in_cams=poses, normal_map=normal_map, mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter, get_vis=self.debug>=2)
    if vis is not None:
      imageio.imwrite(f'{self.debug_dir}/vis_score.png', vis)

    if self.debug >= 2:
      add_errs = self.compute_add_err_to_gt_pose(poses)
      logging.info(f"final, add_errs min:{add_errs.min()}")

    scores = torch.as_tensor(scores, device=poses.device, dtype=torch.float).reshape(-1)
    ids = scores.argsort(descending=True)
    logging.info(f'sort ids:{ids}')
    scores = scores[ids]
    poses = poses[ids]

    logging.info(f'sorted scores:{scores}')

    best_pose = poses[0]@self.get_tf_to_centered_mesh()
    self.pose_last = poses[0]
    self.best_id = int(candidate_ids[ids[0]].item())

    self.poses = poses
    self.scores = scores

    return best_pose.detach().cpu().numpy()


  def compute_add_err_to_gt_pose(self, poses):
    '''
    @poses: wrt. the centered mesh
    '''
    return -torch.ones(len(poses), device='cuda', dtype=torch.float)


  def track_one(self, rgb, depth, K, iteration, extra={}):
    if self.pose_last is None:
      logging.info("Please init pose by register first")
      raise RuntimeError
    logging.info("Welcome")

    depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    depth = erode_depth(depth, radius=2, device='cuda')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')
    logging.info("depth processing done")

    xyz_map = depth2xyzmap_batch(depth[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[0]

    pose, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=self.pose_last.reshape(1,4,4).data.cpu().numpy(), normal_map=None, xyz_map=xyz_map, mesh_diameter=self.diameter, glctx=self.glctx, iteration=iteration, get_vis=self.debug>=2)
    logging.info("pose done")
    if self.debug>=2:
      extra['vis'] = vis
    self.pose_last = pose
    return (pose@self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4,4)
