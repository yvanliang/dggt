import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import Adam
import os
from IPython import embed
from torch.utils.data import Dataset, DataLoader
import random
import open3d as o3d
from PIL import Image
from torchvision import transforms as TF
import numpy as np
import json


WAYMO_OPENCV2DATASET = np.array(
    [
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)


def _normalize_waymo_caption_base(name):
    base = os.path.basename(str(name).lstrip("\ufeff").rstrip("/"))
    if base.endswith(".tfrecord"):
        base = base[: -len(".tfrecord")]
    if base.startswith("segment-"):
        base = base[len("segment-"):]
    suffix = "_with_camera_labels"
    if base.endswith(suffix):
        base = base[:-len(suffix)]
    return base


def _load_waymo_matrix4(path, name):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Missing Waymo {name}: {path}")
    values = np.loadtxt(path, dtype=np.float32)
    if values.size != 16:
        raise ValueError(f"Waymo {name} must contain 16 values, got shape {values.shape}: {path}")
    return values.reshape(4, 4).astype(np.float32)


def _load_waymo_intrinsics_matrix(path):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Missing Waymo intrinsics: {path}")
    values = np.loadtxt(path, dtype=np.float32)
    if values.shape == (3, 3):
        return values.astype(np.float32)
    flat = values.reshape(-1)
    if flat.size < 4:
        raise ValueError(f"Waymo intrinsics must contain at least fx, fy, cx, cy: {path}")
    fx, fy, cx, cy = [float(v) for v in flat[:4]]
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def resize_flow(flow, target_size):
    height, width = flow.shape[-3:-1]
    if (height, width) == target_size:
        return flow
    if len(flow.shape) == 3:
        flow = flow[None, ...]
    target_height, target_width = target_size
    kernel_size_h = height // target_height
    kernel_size_w = width // target_width
    flow[torch.norm(flow, p=2, dim=-1) < 0.5] = -100000
    if kernel_size_h > 0 and kernel_size_w > 0:
        flow = F.max_pool2d(
            flow.permute(0, 3, 1, 2),
            kernel_size=(kernel_size_h, kernel_size_w),
        )
        flow = F.interpolate(flow, size=target_size, mode="nearest")
    else:
        flow = F.interpolate(flow.permute(0, 3, 1, 2), size=target_size, mode="nearest")
    flow = flow.permute(0, 2, 3, 1)
    flow[torch.norm(flow, p=2, dim=-1) > 1000] = 0
    return flow.squeeze()

    
def load_and_preprocess_flow(flow_path_list, extrinsic_paths, intrinsic_path, height, width):
    if len(flow_path_list) == 0:
        raise ValueError("At least 1 image is required")

    flows = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # print(f"[DEBUG] flow_path_list: {flow_path_list}")
    for i, flow_path in enumerate(flow_path_list):
        # print(f"[DEBUG] Processing flow_path[{i}]: {flow_path}")
        # print(f"[DEBUG] Is file: {os.path.isfile(flow_path) if flow_path else 'Empty/None'}")
        # print(f"[DEBUG] Is dir: {os.path.isdir(flow_path) if flow_path else 'Empty/None'}")
        depth_and_flow = np.load(flow_path)
        flow = depth_and_flow
        flow = torch.tensor(flow).float()
        flow = resize_flow(flow, (height, width))
        flows.append(flow)
    
    return torch.stack(flows)


def load_and_preprocess_images(image_path_list, mode="crop"):
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    
    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:

        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size
        
        # Original behavior: set width to 518px
        new_width = target_size
        # Calculate height maintaining aspect ratio, divisible by 14
        new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images
    images = torch.stack(images)  # concatenate images
    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images


def load_and_preprocess_binary_masks(mask_path_list, mode="crop", threshold=0.5):
    # Check for empty list
    if len(mask_path_list) == 0:
        raise ValueError("At least 1 mask is required")

    masks = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    for mask_path in mask_path_list:
        mask = Image.open(mask_path).convert("L")

        width, height = mask.size
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14

        mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)
        mask = to_tensor(mask)

        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            mask = mask[:, start_y : start_y + target_size, :]

        mask = mask.gt(float(threshold)).to(torch.float32)
        mask = mask.expand(3, -1, -1).contiguous()

        shapes.add((mask.shape[1], mask.shape[2]))
        masks.append(mask)

    if len(shapes) > 1:
        print(f"Warning: Found masks with different shapes: {shapes}")
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        padded_masks = []
        for mask in masks:
            h_padding = max_height - mask.shape[1]
            w_padding = max_width - mask.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                mask = torch.nn.functional.pad(
                    mask, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0
                )
            padded_masks.append(mask)
        masks = padded_masks

    masks = torch.stack(masks)
    if len(mask_path_list) == 1 and masks.dim() == 3:
        masks = masks.unsqueeze(0)

    return masks


class WaymoOpenDataset(Dataset):
    def __init__(
        self,
        image_dir,
        scene_names=None,
        sequence_length=None,
        start_idx=-1,
        mode=1,
        views=1,
        intervals=2,
        caption_root=None,
        scene_name_to_index_path="/data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/metadata/training/scene_name_to_index.json",
        waymo_train_list_path=None,
        waymo_val_list_path=None,
    ):
        #mode 1 : train
        #mode 2 : pure reconstruction
        #mode 3 : interplation
        
        self.image_dir = image_dir
        self.sequence_length = sequence_length
        if mode == 1:
            interval = 1
        elif mode == 2:
            interval = 1
        elif mode == 3:
            interval = intervals
        else:
            interval = 1
        self.interval =  interval
        self.mode = mode
        if mode == 1:
            test_mode = False
            load_flow = False
        elif mode == 2:
            test_mode = True
            load_flow = False
        elif mode == 3:
            test_mode = True
            load_flow = True
        else:
            pass
        self.test_mode = test_mode
        self.load_flow = load_flow
        self.views = views
        self.caption_root = caption_root
        self.scene_name_to_index_path = scene_name_to_index_path
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.waymo_train_list_path = waymo_train_list_path or os.path.join(repo_root, "data", "waymo_train_list_full.txt")
        self.waymo_val_list_path = waymo_val_list_path or os.path.join(repo_root, "data", "waymo_val_list_full.txt")
        self.scene_name_to_base = self._build_scene_name_to_base(
            scene_name_to_index_path=scene_name_to_index_path,
            image_dir=image_dir,
        )

        # Scan all scene folders and collect image paths
        if scene_names is None:
            scene_names = [] 
            scene_names_ = [f"{i:03d}" for i in range(0, 99)]
            scene_names = scene_names + scene_names_
        self.scenes = scene_names
        self.image_paths = []
        self.sky_mask_paths = []
        self.dynamic_mask_path = []
        self.extrinsic_paths = []
        self.intrinsic_paths = []
        self.semantic_mask_path = []
        self.depth_flow_paths = []
        self.ego_paths = []

        self.start_idx = start_idx

        for scene_name in scene_names:
            scene_path = os.path.join(image_dir, scene_name, "images")
            if os.path.isdir(scene_path):
                # image
                if self.views == 1:
                    image_paths = sorted(
                        [
                            os.path.join(scene_path, f)
                            for f in os.listdir(scene_path)
                            if f.endswith(("_0.jpg", "_0.png"))
                        ]
                    )
                    self.image_paths.append(image_paths)
                elif self.views == 3:
                    views_image_lists = []
                    for v in range(3):
                        suffixes = (f"_{v}.jpg", f"_{v}.png")
                        files_v = sorted(
                            [os.path.join(scene_path, f) for f in os.listdir(scene_path) if f.endswith(suffixes)]
                        )
                        views_image_lists.append(files_v)
                    lengths = [len(l) for l in views_image_lists]
                    if len(set(lengths)) != 1:
                        raise RuntimeError(f"Inconsistent number of images across views in scene {scene_name}, lengths: {lengths}")
                    self.image_paths.append(views_image_lists)

                # sky_mask
                sky_mask_path = os.path.join(image_dir, scene_name, "sky_masks")
                if os.path.isdir(sky_mask_path):
                    if self.views == 1:
                        sky_mask_paths = sorted(
                            [os.path.join(sky_mask_path, f) for f in os.listdir(sky_mask_path) if f.endswith(("_0.jpg", "_0.png"))]
                        )
                        self.sky_mask_paths.append(sky_mask_paths)
                    elif self.views == 3:
                        views_sky_lists = []
                        for v in range(3):
                            suffixes = (f"_{v}.jpg", f"_{v}.png")
                            files_v = sorted([os.path.join(sky_mask_path, f) for f in os.listdir(sky_mask_path) if f.endswith(suffixes)])
                            views_sky_lists.append(files_v)
                        self.sky_mask_paths.append(views_sky_lists)
                else:
                    self.sky_mask_paths.append([] if self.views == 1 else [[] for _ in range(3)])

                # extrinsic
                extrinsic_path = os.path.join(image_dir, scene_name, "ego_pose")
                if os.path.isdir(extrinsic_path):
                    extrinsic_paths = sorted([
                        os.path.join(extrinsic_path, f)
                        for f in os.listdir(extrinsic_path)
                        if f.endswith(".txt")
                    ])
                    self.extrinsic_paths.append(extrinsic_paths)
                else:
                    self.extrinsic_paths.append([])

                # extrinsic
                ego_path = os.path.join(image_dir, scene_name, "extrinsics")
                # ego_path = os.path.join(image_dir, scene_name, "extrinsics")
                if os.path.isdir(ego_path):
                    ego_path =  os.path.join(ego_path, "0.txt")
                    self.ego_paths.append(ego_path)
                else:
                    self.ego_paths.append("")

                # intrinsic
                intrinsic_path = os.path.join(image_dir, scene_name, "intrinsics")
                if os.path.isdir(intrinsic_path):
                    if self.views == 1:
                        intrinsic_paths = os.path.join(intrinsic_path, "0.txt")
                        self.intrinsic_paths.append(intrinsic_paths)
                    elif self.views == 3:
                        intrinsics_views = []
                        for v in range(3):
                            p = os.path.join(intrinsic_path, f"{v}.txt")
                            intrinsics_views.append(p if os.path.exists(p) else "")
                        self.intrinsic_paths.append(intrinsics_views)
                else:
                    self.intrinsic_paths.append("" if self.views == 1 else ["" for _ in range(3)])

                # dynamic mask
                dynamic_mask_path = os.path.join(image_dir, scene_name, "fine_dynamic_masks/all")
                if os.path.isdir(dynamic_mask_path):
                    if self.views == 1:
                        dynamic_mask_paths = sorted(
                            [os.path.join(dynamic_mask_path, f) for f in os.listdir(dynamic_mask_path) if f.endswith(("_0.jpg", "_0.png"))]
                        )
                        self.dynamic_mask_path.append(dynamic_mask_paths)
                    elif self.views == 3:
                        views_dyn_lists = []
                        for v in range(3):
                            suffixes = (f"_{v}.jpg", f"_{v}.png")
                            files_v = sorted([os.path.join(dynamic_mask_path, f) for f in os.listdir(dynamic_mask_path) if f.endswith(suffixes)])
                            views_dyn_lists.append(files_v)
                        self.dynamic_mask_path.append(views_dyn_lists)
                else:
                    self.dynamic_mask_path.append([] if self.views == 1 else [[] for _ in range(3)])
                # depth
                depth_path = os.path.join(image_dir, scene_name, "depth_flows_4")
                if os.path.isdir(depth_path):
                    if self.views == 1:
                        depth_paths = sorted(
                            [os.path.join(depth_path, f) for f in os.listdir(depth_path) if f.endswith("_0.npy")]
                        )
                        self.depth_flow_paths.append(depth_paths)
                    elif self.views == 3:
                        views_depth_lists = []
                        for v in range(3):
                            suffix = f"_{v}.npy"
                            files_v = sorted(
                                [os.path.join(depth_path, f) for f in os.listdir(depth_path) if f.endswith(suffix)]
                            )
                            views_depth_lists.append(files_v)
                        self.depth_flow_paths.append(views_depth_lists)
                else:
                    self.depth_flow_paths.append([] if self.views == 1 else [[] for _ in range(3)])
                # semantic mask
                semantic_mask_path = os.path.join(image_dir, scene_name, "custom_masks")
                if os.path.isdir(semantic_mask_path):
                    if self.views == 1:
                        semantic_mask_paths = sorted(
                            [os.path.join(semantic_mask_path, f) for f in os.listdir(semantic_mask_path) if f.endswith(("_0.jpg", "_0.png"))]
                        )
                        self.semantic_mask_path.append(semantic_mask_paths)
                    elif self.views == 3:
                        views_sem_lists = []
                        for v in range(3):
                            suffixes = (f"_{v}.jpg", f"_{v}.png")
                            files_v = sorted([os.path.join(semantic_mask_path, f) for f in os.listdir(semantic_mask_path) if f.endswith(suffixes)])
                            views_sem_lists.append(files_v)
                        self.semantic_mask_path.append(views_sem_lists)
                else:
                    self.semantic_mask_path.append([] if self.views == 1 else [[] for _ in range(3)])


    def _build_scene_name_to_base(self, scene_name_to_index_path, image_dir):
        mapping = self._load_scene_name_to_base(scene_name_to_index_path)
        list_paths = self._get_preferred_waymo_list_paths(image_dir)
        list_mapping = self._load_scene_name_to_base_from_lists(list_paths)
        # Prefer explicit train/val list indexing for numeric scene ids (000, 001, ...).
        mapping.update(list_mapping)
        return mapping

    def _get_preferred_waymo_list_paths(self, image_dir):
        image_dir_l = str(image_dir).lower()
        has_train = os.path.exists(self.waymo_train_list_path)
        has_val = os.path.exists(self.waymo_val_list_path)
        if "validation" in image_dir_l or "/val" in image_dir_l:
            return [p for p in (self.waymo_val_list_path, self.waymo_train_list_path) if os.path.exists(p)]
        if "training" in image_dir_l or "/train" in image_dir_l:
            return [p for p in (self.waymo_train_list_path, self.waymo_val_list_path) if os.path.exists(p)]
        if has_train and has_val:
            return [self.waymo_train_list_path, self.waymo_val_list_path]
        if has_train:
            return [self.waymo_train_list_path]
        if has_val:
            return [self.waymo_val_list_path]
        return []

    def _load_scene_name_to_base_from_lists(self, paths):
        mapping = {}
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8-sig") as f:
                    lines = [line.strip().lstrip("\ufeff") for line in f if line.strip()]
            except Exception as exc:
                if self.caption_root:
                    raise RuntimeError(f"Failed to read Waymo list file: {path}") from exc
                continue
            for idx, item in enumerate(lines):
                key = f"{idx:03d}"
                if key not in mapping:
                    mapping[key] = _normalize_waymo_caption_base(item)
        return mapping


    def _load_scene_name_to_base(self, path):
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as exc:
            if self.caption_root:
                raise RuntimeError(f"Failed to read scene_name_to_index mapping: {path}") from exc
            return {}
        if not isinstance(data, dict):
            if self.caption_root:
                raise ValueError(f"scene_name_to_index mapping must be a JSON object: {path}")
            return {}
        mapping = {}
        for key, value in data.items():
            key_s = str(key)
            value_s = str(value)
            if value_s.isdigit():
                mapping[value_s.zfill(3)] = _normalize_waymo_caption_base(key_s)
            if key_s.isdigit():
                mapping[key_s.zfill(3)] = _normalize_waymo_caption_base(value_s)
        return mapping

    def _load_caption(self, scene_name, start_idx):
        if not self.caption_root:
            return None
        scene_key = str(scene_name).zfill(3) if str(scene_name).isdigit() else str(scene_name)
        scene_base = self.scene_name_to_base.get(scene_key, str(scene_name))
        scene_base = _normalize_waymo_caption_base(scene_base)
        clip_index = int(start_idx) // 29
        path = os.path.join(str(self.caption_root), "pinhole_front", f"{scene_base}_{clip_index}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Caption file not found for scene={scene_name!r}, clip_index={clip_index}: {path}"
            )
        try:
            with open(path, "r") as f:
                payload = json.load(f)
        except Exception as exc:
            raise RuntimeError(f"Failed to read caption JSON: {path}") from exc
        if isinstance(payload, dict):
            caption = payload.get("caption")
            if caption is not None:
                return str(caption)
            raise KeyError(f"Caption JSON lacks 'caption': {path}")
        raise ValueError(f"Caption JSON must be an object with key 'caption': {path}")


    def __len__(self):
        return len(self.scenes)

    def _sample_start_within_caption_trunk(self, total_frames, trunk_frames=29):
        """Sample a contiguous raw-pretrain window without crossing caption trunks."""
        sequence_length = int(self.sequence_length)
        total_frames = int(total_frames)
        trunk_frames = int(trunk_frames)
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {self.sequence_length}")
        if total_frames <= sequence_length:
            return 0
        valid_trunks = []
        for base in range(0, total_frames - trunk_frames + 1, trunk_frames):
            end = base + trunk_frames
            if end - base >= sequence_length:
                valid_trunks.append((base, end))
        if not valid_trunks:
            return random.randint(0, max(0, total_frames - sequence_length))
        base, end = random.choice(valid_trunks)
        return random.randint(base, end - sequence_length)

    def _fixed_start_within_caption_trunk(self, total_frames, trunk_frames=29):
        """Return a deterministic start index without crossing caption trunks."""
        sequence_length = int(self.sequence_length)
        total_frames = int(total_frames)
        trunk_frames = int(trunk_frames)
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {self.sequence_length}")
        max_start = max(0, total_frames - sequence_length)
        start = min(max(0, int(self.start_idx)), max_start)
        if total_frames <= sequence_length:
            return 0

        trunk_base = (start // trunk_frames) * trunk_frames
        trunk_end = min(trunk_base + trunk_frames, total_frames)
        if trunk_end - trunk_base < sequence_length:
            return min(trunk_base, max_start)
        if start + sequence_length > trunk_end:
            start = trunk_end - sequence_length
        return min(max(trunk_base, start), max_start)

    def _load_front_waymo_camera_gt(self, idx, indices, image_seq):
        if self.views != 1:
            raise ValueError("Raw Waymo camera GT loading currently expects front-camera views=1 clips.")
        ego_pose_paths = self.extrinsic_paths[idx]
        camera_extrinsic_path = self.ego_paths[idx] if idx < len(self.ego_paths) else ""
        intrinsic_path = self.intrinsic_paths[idx]
        if isinstance(intrinsic_path, (list, tuple)):
            intrinsic_path = intrinsic_path[0] if len(intrinsic_path) > 0 else ""
        if len(ego_pose_paths) == 0:
            raise RuntimeError(f"Scene {self.scenes[idx]} is missing ego_pose/*.txt files.")
        if max(indices) >= len(ego_pose_paths):
            raise RuntimeError(
                f"Scene {self.scenes[idx]} has {len(ego_pose_paths)} ego poses, "
                f"but clip needs frame {max(indices)}."
            )

        cam_to_ego = _load_waymo_matrix4(camera_extrinsic_path, "front camera extrinsics")
        cam_to_ego = (cam_to_ego @ WAYMO_OPENCV2DATASET).astype(np.float32)
        ego_to_world_start = _load_waymo_matrix4(ego_pose_paths[indices[0]], "ego pose")
        ego_start_inv = np.linalg.inv(ego_to_world_start).astype(np.float32)
        camera_to_world = []
        for frame_idx in indices:
            ego_to_world = _load_waymo_matrix4(ego_pose_paths[frame_idx], "ego pose")
            camera_to_world.append((ego_start_inv @ ego_to_world @ cam_to_ego).astype(np.float32))
        camera_to_world = np.stack(camera_to_world, axis=0)[:, None]

        intrinsics = _load_waymo_intrinsics_matrix(intrinsic_path)[None]
        with Image.open(image_seq[0]) as img:
            raw_width, raw_height = img.size
        raw_image_size_hw = torch.tensor([int(raw_height), int(raw_width)], dtype=torch.long)
        return (
            torch.from_numpy(camera_to_world).float(),
            torch.from_numpy(intrinsics).float(),
            raw_image_size_hw,
        )

    def __getitem__(self, idx):
        image_paths = self.image_paths[idx]
        sky_mask_paths = self.sky_mask_paths[idx]
        dynamic_mask_paths = self.dynamic_mask_path[idx]
        semantic_mask_paths = self.semantic_mask_path[idx]

        total_frames = len(image_paths[0] if self.views == 3 else image_paths)
        start_idx = (
            self._fixed_start_within_caption_trunk(total_frames)
            if self.mode == 1 and int(self.start_idx) >= 0
            else self._sample_start_within_caption_trunk(total_frames)
            if self.mode == 1
            else 0
        )

        if self.mode == 1:
            indices = list(range(start_idx, start_idx + self.sequence_length))

            #images
            if self.views == 1:
                seq = [image_paths[i] for i in indices]
                images = load_and_preprocess_images(seq)  # [S, C, H, W]
            elif self.views == 3:
                seq = []
                for i in indices:
                    for v in range(3):
                        seq.append(image_paths[v][i])
                images = load_and_preprocess_images(seq)  # [S*3, C, H, W]

            #sky masks
            if self.views == 1:
                mask_seq = [sky_mask_paths[i] for i in indices]
                masks = load_and_preprocess_binary_masks(mask_seq)  # [S, C, H, W]
            elif self.views == 3:
                mask_seq = []
                for i in indices:
                    for v in range(3):
                        mask_seq.append(sky_mask_paths[v][i])
                masks = load_and_preprocess_binary_masks(mask_seq)  # [S*3, C, H, W]

            timestamps = np.array(indices) - start_idx
            timestamps = timestamps / timestamps[-1] * (self.sequence_length / 4)
            frame_ids = np.array(indices, dtype=np.int64) - int(start_idx // 29) * 29
            if self.views == 3:
                timestamps = np.repeat(timestamps, 3)
                frame_ids = np.repeat(frame_ids, 3)

            input_dict = {
                "images": images,
                "masks": masks,
                "image_paths": seq,
                "timestamps": timestamps,
                "frame_ids": frame_ids,
                "interval": [1] * (self.sequence_length - 1),
                "scene_name": self.scenes[idx],
                "start_idx": start_idx,
                "clip_index": start_idx // 29,
            }
            caption = self._load_caption(self.scenes[idx], start_idx)
            if caption is not None:
                if self.views == 1:
                    caption = caption.replace("multi-camera", "front-camera")
                input_dict["caption"] = caption

            if self.views == 1:
                camera_to_world, intrinsics, raw_image_size_hw = self._load_front_waymo_camera_gt(idx, indices, seq)
                input_dict["camera_to_world_corrected"] = camera_to_world
                input_dict["intrinsics"] = intrinsics
                input_dict["raw_image_size_hw"] = raw_image_size_hw
        
            if len(dynamic_mask_paths) > 0:
                if self.views == 1:
                    dy_mask_seq = [dynamic_mask_paths[i] for i in indices]
                    dynamic_mask = load_and_preprocess_binary_masks(dy_mask_seq)  # [S, C, H, W]
                elif self.views == 3:
                    dy_mask_seq = []
                    for i in indices:
                        for v in range(3):
                            dy_mask_seq.append(dynamic_mask_paths[v][i])
                    dynamic_mask = load_and_preprocess_binary_masks(dy_mask_seq)  # [S*3, C, H, W]
                input_dict["dynamic_mask"] = dynamic_mask

            
            # if len(semantic_mask_paths) > 0:
            #     if self.views == 1:
            #         sem_mask_seq = [semantic_mask_paths[i] for i in indices]
            #         semantic_mask = load_and_preprocess_images(sem_mask_seq)  # [S, C, H, W]
            #     elif self.views == 3:
            #         sem_mask_seq = []
            #         for i in indices:
            #             for v in range(3):
            #                 sem_mask_seq.append(semantic_mask_paths[v][i])
            #         semantic_mask = load_and_preprocess_images(sem_mask_seq)  # [S*3, C, H, W]
            #     semantic_mask = semantic_mask * 255 / 10
            #     semantic_mask = semantic_mask.int()
            #     semantic_mask[semantic_mask > 9] = 255
            #     input_dict["semantic_mask"] = semantic_mask

            return input_dict

        elif self.mode == 2: 
            start_idx = 0
            indices = [start_idx + i * self.interval for i in range(self.sequence_length)]
            intervals = [self.interval for _ in range(self.sequence_length - 1)]
            
            timestamps = np.array(indices) - start_idx
            timestamps = timestamps / timestamps[-1] * (self.sequence_length / 4)
            frame_ids = np.array(indices, dtype=np.int64) - int(start_idx // 29) * 29
            if self.views == 3:
                timestamps = np.repeat(timestamps, 3)
                frame_ids = np.repeat(frame_ids, 3)

            #images
            if self.views == 1:
                seq = [image_paths[i] for i in indices]
                images = load_and_preprocess_images(seq)  # [S, C, H, W]
            elif self.views == 3:
                seq = []
                for i in indices:
                    for v in range(3):
                        seq.append(image_paths[v][i])
                images = load_and_preprocess_images(seq)  # [S*3, C, H, W]

            #sky masks
            if self.views == 1:
                mask_seq = [sky_mask_paths[i] for i in indices]
                masks = load_and_preprocess_binary_masks(mask_seq)  # [S, C, H, W]
            elif self.views == 3:
                mask_seq = []
                for i in indices:
                    for v in range(3):
                        mask_seq.append(sky_mask_paths[v][i])
                masks = load_and_preprocess_binary_masks(mask_seq)  # [S*3, C, H, W]
                


            input_dict = {
                "images": images,
                "image_paths": seq,
                "masks": masks,
                "timestamps": timestamps,
                "frame_ids": frame_ids,
                "interval": intervals,
                "scene_name": self.scenes[idx],
                "start_idx": start_idx,
                "clip_index": start_idx // 29,
            }
            caption = self._load_caption(self.scenes[idx], start_idx)
            if caption is not None:
                input_dict["caption"] = caption
            if len(dynamic_mask_paths) > 0:
                if self.views == 1:
                    dy_mask_seq = [dynamic_mask_paths[i] for i in indices]
                    dynamic_mask = load_and_preprocess_binary_masks(dy_mask_seq)  # [S, C, H, W]
                elif self.views == 3:
                    dy_mask_seq = []
                    for i in indices:
                        for v in range(3):
                            dy_mask_seq.append(dynamic_mask_paths[v][i])
                    dynamic_mask = load_and_preprocess_binary_masks(dy_mask_seq)  # [S*3, C, H, W]
                input_dict["dynamic_mask"] = dynamic_mask

            if len(self.depth_flow_paths) > 0 and len(self.depth_flow_paths[idx]) > 0:
                if self.views == 1:
                    if len(self.depth_flow_paths[idx]) > 0:
                        depth_seq = [self.depth_flow_paths[idx][i] for i in indices]
                        depth_data = load_and_preprocess_flow(depth_seq, None, None, images.shape[2], images.shape[3])
                    else:
                        depth_data = torch.zeros(len(indices), images.shape[2], images.shape[3])
                    input_dict["gt_depth"] = depth_data
                elif self.views == 3:
                    # Check if all views have depth paths
                    if all(len(self.depth_flow_paths[idx][v]) > 0 for v in range(3)):
                        depth_seq = []
                        for i in indices:
                            for v in range(3):
                                depth_seq.append(self.depth_flow_paths[idx][v][i])
                        depth_data = load_and_preprocess_flow(depth_seq, None, None, images.shape[2], images.shape[3])
                    else:
                        depth_data = torch.zeros(len(indices) * 3, images.shape[2], images.shape[3])
                    input_dict["gt_depth"] = depth_data
            else:
                # No depth data available, create zero tensor with same shape as images
                if self.views == 1:
                    depth_data = torch.zeros(len(indices), images.shape[2], images.shape[3])
                else:
                    depth_data = torch.zeros(len(indices) * 3, images.shape[2], images.shape[3])
                input_dict["gt_depth"] = depth_data

            return input_dict

        else:  # self.mode == 3
            start_idx = 0
            indices = [start_idx + i * self.interval for i in range(self.sequence_length)]
            intervals = [self.interval for _ in range(self.sequence_length - 1)]
            target_indices = [start_idx + i for i in range(self.sequence_length * self.interval - (self.interval - 1))]

            timestamps = np.array(indices) - start_idx
            timestamps = timestamps / timestamps[-1] * (self.sequence_length / 4)
            if self.views == 3:
                timestamps = np.repeat(timestamps, 3)
            
            # images
            if self.views == 1:
                seq = [image_paths[i] for i in indices]
                images = load_and_preprocess_images(seq)  # [S, C, H, W]
                target_seq = [image_paths[i] for i in target_indices]
                target_images = load_and_preprocess_images(target_seq)  # [T, C, H, W]
            elif self.views == 3:
                seq = []
                for i in indices:
                    for v in range(3):
                        seq.append(image_paths[v][i])
                images = load_and_preprocess_images(seq)  # [S*3, C, H, W]

                target_seq = []
                for i in target_indices:
                    for v in range(3):
                        target_seq.append(image_paths[v][i])
                target_images = load_and_preprocess_images(target_seq)  # [T*3, C, H, W]

            # sky masks
            if self.views == 1:
                mask_seq = [sky_mask_paths[i] for i in indices]
                masks = load_and_preprocess_binary_masks(mask_seq)  # [S, C, H, W]
                target_mask_seq = [sky_mask_paths[i] for i in target_indices]
                target_masks = load_and_preprocess_binary_masks(target_mask_seq)  # [T, C, H, W]
            elif self.views == 3:
                mask_seq = []
                for i in indices:
                    for v in range(3):
                        mask_seq.append(sky_mask_paths[v][i])
                masks = load_and_preprocess_binary_masks(mask_seq)  # [S*3, C, H, W]

                target_mask_seq = []
                for i in target_indices:
                    for v in range(3):
                        target_mask_seq.append(sky_mask_paths[v][i])
                target_masks = load_and_preprocess_binary_masks(target_mask_seq)  # [T*3, C, H, W]

            input_dict = {
                "images": images,
                "targets": target_images,
                "masks": masks,
                "image_paths": seq,
                "timestamps": timestamps,
                # "target_timestamps": target_timestamps,
                "interval": intervals,
                "target_masks": target_masks,
                "scene_name": self.scenes[idx],
                "start_idx": start_idx,
                "clip_index": start_idx // 29,
            }
            caption = self._load_caption(self.scenes[idx], start_idx)
            if caption is not None:
                input_dict["caption"] = caption

            if len(dynamic_mask_paths) > 0:
                if self.views == 1:
                    dy_mask_seq = [dynamic_mask_paths[i] for i in indices]
                    dynamic_mask = load_and_preprocess_binary_masks(dy_mask_seq)  # [S, C, H, W]
                    target_dy_mask_seq = [dynamic_mask_paths[i] for i in target_indices]
                    target_dynamic_mask = load_and_preprocess_binary_masks(target_dy_mask_seq)  # [T, C, H, W]
                elif self.views == 3:
                    dy_mask_seq = []
                    target_dy_mask_seq = []
                    for i in indices:
                        for v in range(3):
                            dy_mask_seq.append(dynamic_mask_paths[v][i])
                    for i in target_indices:
                        for v in range(3):
                            target_dy_mask_seq.append(dynamic_mask_paths[v][i])
                    dynamic_mask = load_and_preprocess_binary_masks(dy_mask_seq)         # [S*3, C, H, W]
                    target_dynamic_mask = load_and_preprocess_binary_masks(target_dy_mask_seq)  # [T*3, C, H, W]
                input_dict["dynamic_mask"] = target_dynamic_mask

            if len(self.depth_flow_paths) > 0 and len(self.depth_flow_paths[idx]) > 0:
                if self.views == 1:
                    if len(self.depth_flow_paths[idx]) > 0:
                        depth_seq = [self.depth_flow_paths[idx][i] for i in indices]
                        depth_data = load_and_preprocess_flow(depth_seq, None, None, images.shape[2], images.shape[3])
                        target_depth_seq = [self.depth_flow_paths[idx][i] for i in target_indices]
                        target_depth_data = load_and_preprocess_flow(target_depth_seq, None, None, images.shape[2], images.shape[3])
                    else:
                        target_depth_data = torch.zeros(len(target_indices), images.shape[2], images.shape[3])
                    input_dict["gt_depth"] = target_depth_data
                elif self.views == 3:
                    # Check if all views have depth paths
                    if all(len(self.depth_flow_paths[idx][v]) > 0 for v in range(3)):
                        depth_seq = []
                        target_depth_seq = []
                        for i in indices:
                            for v in range(3):
                                depth_seq.append(self.depth_flow_paths[idx][v][i])
                        for i in target_indices:
                            for v in range(3):
                                target_depth_seq.append(self.depth_flow_paths[idx][v][i])
                        depth_data = load_and_preprocess_flow(depth_seq, None, None, images.shape[2], images.shape[3])
                        target_depth_data = load_and_preprocess_flow(target_depth_seq, None, None, images.shape[2], images.shape[3])
                    else:
                        target_depth_data = torch.zeros(len(target_indices) * 3, images.shape[2], images.shape[3])
                    input_dict["gt_depth"] = target_depth_data
            else:
                # No depth data available, create zero tensor with same shape as images
                if self.views == 1:
                    target_depth_data = torch.zeros(len(target_indices), images.shape[2], images.shape[3])
                else:
                    target_depth_data = torch.zeros(len(target_indices) * 3, images.shape[2], images.shape[3])
                input_dict["gt_depth"] = target_depth_data
            return input_dict
