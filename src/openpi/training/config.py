"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.wuji_policy as wuji_policy
import openpi.shared.download as _download
import openpi.shared.nnx_utils as nnx_utils
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

# RoboDojo normalization assets bundled with this adapter (openpi/assets/RoboDojo_assets).
_ROBODOJO_ASSETS_DIR = pathlib.Path(__file__).resolve().parents[3] / "assets" / "RoboDojo_assets"
# Norm stats for Tianji Marvin + Wuji Hand (written by compute_norm_stats).
_WUJI_ASSETS_DIR = pathlib.Path(__file__).resolve().parents[3] / "assets" / "Wuji_assets"

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Optional source episode identities to load from a LeRobot dataset. Keeping
    # this in the frozen config prevents a local partial mirror from silently
    # changing which episodes enter training.
    episode_indices: Sequence[int] = ()
    # Video decoder backend for LeRobot datasets. Forced to pyav by default because
    # torchcodec is present in some environments but not fully functional at runtime.
    video_backend: Literal["pyav", "torchcodec", "video_reader"] = "pyav"
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Optional contiguous task frame counts for deterministic weighted sampling.
    # When set, each task is sampled with p_i proportional to n_i**task_sampling_exponent.
    task_frame_counts: Sequence[int] = ()
    task_sampling_exponent: float | None = None

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = False
    # Physical camera streams exposed to the policy. The source dataset may
    # retain all streams so paired camera conditions share identical records.
    enabled_cameras: tuple[str, ...] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[
                aloha_policy.AlohaInputs(
                    adapt_to_pi=self.adapt_to_pi,
                    enabled_cameras=self.enabled_cameras,
                )
            ],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotWujiDataConfig(DataConfigFactory):
    """
    Config for Tianji Marvin + dual Wuji Hand Gen1 (adapted from wuji-openpi).

    Dataset features:
    - observation.state: 54 dims (7 left arm + 20 left hand + 7 right arm + 20 right hand)
    - action: 54 dims
    - observation.images.cam_left_wrist / cam_right_wrist
    - base camera: observation.images.stereo_right (default) or cam_high via base_image_key

    Requires model.action_dim=54. Load pi05_base with PartialCheckpointWeightLoader.
    """

    extra_delta_transform: bool = False
    # LeRobot key for the head / stereo base camera.
    base_image_key: str = "observation.images.stereo_right"
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": self.base_image_key,
                        "observation/left_wrist_image": "observation.images.cam_left_wrist",
                        "observation/right_wrist_image": "observation.images.cam_right_wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[wuji_policy.WujiInputs(model_type=model_config.model_type)],
            outputs=[wuji_policy.WujiOutputs(action_dim=model_config.action_dim)],
        )

        # Arms: delta; hands: absolute (same as wuji-openpi).
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(7, -20, 7, -20)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"
    # Optional exact checkpoint directory. XPolicyLab train.sh uses this to
    # keep policy checkpoints under policy/<name>/checkpoints/<6-tuple>.
    checkpoint_dir_override: str | None = None

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 8
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        if self.checkpoint_dir_override:
            return pathlib.Path(self.checkpoint_dir_override).resolve()
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    TrainConfig(
        name="pi05_base_aloha_full_sim_arx-x5_seed_0",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="RoboDojo_sim_arx-x5_v30",
            assets=AssetsConfig(
                assets_dir=str(_ROBODOJO_ASSETS_DIR),
                asset_id="arx_x5_sim",
            ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                prompt_from_task=True,  # Set to True for prompt by task_name
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        seed=0,
        batch_size=256,
        fsdp_devices=2,
        num_train_steps=60000,
    ),
    TrainConfig(
        name="pi05_base_aloha_full_sim_arx-x5_seed_1",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="RoboDojo_sim_arx-x5_v30",
            assets=AssetsConfig(
                assets_dir=str(_ROBODOJO_ASSETS_DIR),
                asset_id="arx_x5_sim",
            ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                prompt_from_task=True,  # Set to True for prompt by task_name
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        seed=1,
        batch_size=256,
        fsdp_devices=2,
        num_train_steps=60000,
    ),
    TrainConfig(
        name="pi05_base_aloha_full_sim_arx-x5_seed_2",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="RoboDojo_sim_arx-x5_v30",
            assets=AssetsConfig(
                assets_dir=str(_ROBODOJO_ASSETS_DIR),
                asset_id="arx_x5_sim",
            ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                prompt_from_task=True,  # Set to True for prompt by task_name
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        seed=2,
        batch_size=256,
        fsdp_devices=2,
        num_train_steps=60000,
    ),
    # Tianji Marvin + dual Wuji Hand Gen1 (54D). Set data.repo_id to your LeRobot dataset
    # before compute_norm_stats / train. Skip RoboDojo process_data.py; use wuji mcap→LeRobot.
    TrainConfig(
        name="pi05_wuji_marvin_54d",
        model=pi0_config.Pi0Config(pi05=True, action_dim=54, action_horizon=50, max_token_len=256),
        data=LeRobotWujiDataConfig(
            # Replace with your LeRobot repo id or local dataset id under HF_LEROBOT_HOME.
            repo_id="tianji_marvin_wuji",
            assets=AssetsConfig(
                assets_dir=str(_WUJI_ASSETS_DIR),
                asset_id="tianji_marvin_wuji",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
            # If your dataset uses cam_high instead of stereo_right, set:
            # base_image_key="observation.images.cam_high",
        ),
        weight_loader=weight_loaders.PartialCheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=30_000,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
    ),
]


def _g31c_train_only_filter(*patterns: str) -> Filter:
    """Freeze every parameter except paths matched by one of the recipe patterns."""
    return nnx.Not(nnx.Any(*(nnx_utils.PathRegex(pattern) for pattern in patterns)))


def _g31c_policy_metadata(recipe: str) -> dict[str, Any]:
    return {
        "task_info": {
            "benchmark": "RoboTwin-2.0",
            "task": "adjust_bottle",
            "task_config": "demo_clean",
            "scene": "Easy",
        },
        "robot_config": {
            "embodiment": "aloha-agilex",
            "action_type": "joint",
            "state_action_dim": 14,
        },
        "input_config": {
            "cameras": ["head", "left_wrist", "right_wrist"],
            "language_instruction": True,
        },
        "training_purpose": {
            "stage": "G3.1c",
            "type": "trainable-set-recipe",
            "selection_axis": "trainable-set",
        },
        "recipe": recipe,
    }


def _g31c_recipe_configs(base: TrainConfig) -> list[TrainConfig]:
    """Register the four π0.5 trainable-set recipes used by manip G3.1c."""
    interface = r"(?:action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)/.*"
    lora = r".*lora.*"
    action_expert = r".*llm.*_1.*"

    strict_model = dataclasses.replace(
        base.model,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    expert_model = dataclasses.replace(
        base.model,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
    )
    dual_lora_model = strict_model
    hybrid_model = dataclasses.replace(
        base.model,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m",
    )
    common = {"ema_decay": None, "fsdp_devices": 1}
    return [
        dataclasses.replace(
            base,
            name="pi05_g31c_strict_dual_lora_interface",
            model=strict_model,
            freeze_filter=_g31c_train_only_filter(lora, interface),
            policy_metadata=_g31c_policy_metadata("strict-dual-lora-interface"),
            **common,
        ),
        dataclasses.replace(
            base,
            name="pi05_g31c_action_expert_interface",
            model=expert_model,
            freeze_filter=_g31c_train_only_filter(action_expert, interface),
            policy_metadata=_g31c_policy_metadata("action-expert-interface"),
            **common,
        ),
        dataclasses.replace(
            base,
            name="pi05_g31c_builtin_dual_lora",
            model=dual_lora_model,
            freeze_filter=dual_lora_model.get_freeze_filter(),
            policy_metadata=_g31c_policy_metadata("builtin-dual-lora"),
            **common,
        ),
        dataclasses.replace(
            base,
            name="pi05_g31c_vlm_lora_expert_full",
            model=hybrid_model,
            freeze_filter=hybrid_model.get_freeze_filter(),
            policy_metadata=_g31c_policy_metadata("vlm-lora-expert-full"),
            **common,
        ),
    ]


_CONFIGS.extend(_g31c_recipe_configs(_CONFIGS[0]))


def _g33_policy_metadata(task: str, recipe: str) -> dict[str, Any]:
    return {
        "task_info": {
            "benchmark": "RoboTwin-2.0",
            "task": task,
            "task_config": "demo_clean",
            "scene": "Easy",
        },
        "robot_config": {
            "embodiment": "aloha-agilex",
            "action_type": "joint",
            "state_action_dim": 14,
        },
        "input_config": {
            "cameras": ["head", "left_wrist", "right_wrist"],
            "language_instruction": True,
        },
        "training_purpose": {
            "stage": "G3.3",
            "type": "cross-task-start-factorial",
            "comparison_axes": ["recipe", "start"],
            "training_seed": 0,
        },
        "recipe": recipe,
    }


def _g33_recipe_configs(base: TrainConfig) -> list[TrainConfig]:
    """Register matching built-in/hybrid configs for both G3.3 tasks."""
    templates = {
        config.policy_metadata["recipe"]: config
        for config in _g31c_recipe_configs(base)
        if config.policy_metadata is not None
        and config.policy_metadata["recipe"] in {"builtin-dual-lora", "vlm-lora-expert-full"}
    }
    configs = []
    for task in ("lift_pot", "handover_block"):
        for recipe, template in templates.items():
            recipe_name = recipe.replace("-", "_")
            configs.append(
                dataclasses.replace(
                    template,
                    name=f"pi05_g33_{task}_{recipe_name}",
                    policy_metadata=_g33_policy_metadata(task, recipe),
                )
            )
    return configs


_CONFIGS.extend(_g33_recipe_configs(_CONFIGS[0]))


def _g34_policy_metadata() -> dict[str, Any]:
    return {
        "task_info": {
            "benchmark": "RoboTwin-2.0",
            "tasks": ["adjust_bottle", "lift_pot", "pick_dual_bottles", "handover_block"],
            "task_config": "demo_clean",
            "scene": "Easy",
        },
        "robot_config": {
            "embodiment": "aloha-agilex",
            "action_type": "joint",
            "state_action_dim": 14,
        },
        "input_config": {
            "cameras": ["head", "left_wrist", "right_wrist"],
            "language_instruction": True,
            "canonical_task_id_model_input": False,
        },
        "training_purpose": {
            "stage": "G3.4",
            "type": "four-task-joint-training",
            "training_seed": 0,
            "sampling_rule": "p_i_proportional_to_n_i_power_0.43",
        },
        "recipe": "builtin-dual-lora",
    }


def _g34_recipe_config(base: TrainConfig) -> TrainConfig:
    """Register the frozen four-task G3.4 joint recipe."""
    template = next(
        config
        for config in _g31c_recipe_configs(base)
        if config.policy_metadata is not None and config.policy_metadata["recipe"] == "builtin-dual-lora"
    )
    assert isinstance(template.data, LeRobotAlohaDataConfig)
    joint_repo_id = "RoboTwin-g34-joint4-aloha_agilex-joint"
    return dataclasses.replace(
        template,
        name="pi05_g34_joint4_builtin_dual_lora",
        data=dataclasses.replace(
            template.data,
            repo_id=joint_repo_id,
            assets=AssetsConfig(asset_id=joint_repo_id),
            base_config=DataConfig(
                prompt_from_task=True,
                task_frame_counts=(7188, 5554, 6129, 14084),
                task_sampling_exponent=0.43,
            ),
        ),
        policy_metadata=_g34_policy_metadata(),
        batch_size=32,
        seed=0,
        ema_decay=None,
        fsdp_devices=1,
    )


def _g34_pick_dual_bottles_config(base: TrainConfig) -> TrainConfig:
    """Register the matched single-task comparator for G3.4."""
    template = next(
        config
        for config in _g31c_recipe_configs(base)
        if config.policy_metadata is not None and config.policy_metadata["recipe"] == "builtin-dual-lora"
    )
    assert isinstance(template.data, LeRobotAlohaDataConfig)
    repo_id = "RoboTwin-pick_dual_bottles-aloha_agilex-joint"
    metadata = _g33_policy_metadata("pick_dual_bottles", "builtin-dual-lora")
    metadata["training_purpose"] = {
        "stage": "G3.4",
        "type": "matched-single-task-comparator",
        "training_seed": 0,
    }
    return dataclasses.replace(
        template,
        name="pi05_g34_pick_dual_bottles_builtin_dual_lora",
        data=dataclasses.replace(
            template.data,
            repo_id=repo_id,
            assets=AssetsConfig(asset_id=repo_id),
        ),
        policy_metadata=metadata,
        batch_size=32,
        seed=0,
        ema_decay=None,
        fsdp_devices=1,
    )


_CONFIGS.extend([_g34_recipe_config(_CONFIGS[0]), _g34_pick_dual_bottles_config(_CONFIGS[0])])


_G4_J10_TASKS = (
    "grab_roller",
    "adjust_bottle",
    "lift_pot",
    "dump_bin_bigbin",
    "click_alarmclock",
    "pick_dual_bottles",
    "handover_block",
    "beat_block_hammer",
    "place_a2b_left",
    "place_a2b_right",
)


def _g4_j10_policy_metadata(distribution: str) -> dict[str, Any]:
    return {
        "task_info": {
            "benchmark": "RoboTwin-2.0",
            "tasks": list(_G4_J10_TASKS),
            "environment_distribution": distribution,
        },
        "robot_config": {
            "embodiment": "aloha-agilex",
            "action_type": "joint",
            "state_action_dim": 14,
        },
        "input_config": {
            "cameras": ["head", "left_wrist", "right_wrist"],
            "language_instruction": True,
            "canonical_task_id_model_input": False,
        },
        "training_purpose": {
            "stage": "G4.2",
            "type": "matched-joint10-training",
            "controlled_axis": "environment_distribution",
            "training_seed": 0,
            "sampling_rule": "p_i_proportional_to_n_i_power_0.43",
        },
        "recipe": "builtin-dual-lora",
    }


def _g4_j10_recipe_configs(base: TrainConfig) -> list[TrainConfig]:
    """Register the matched clean/mixed Joint10 G4.2 recipes."""
    template = next(
        config
        for config in _g31c_recipe_configs(base)
        if config.policy_metadata is not None and config.policy_metadata["recipe"] == "builtin-dual-lora"
    )
    assert isinstance(template.data, LeRobotAlohaDataConfig)
    variants = {
        "clean": (
            "RoboTwin-g4-j10-clean-aloha_agilex-joint",
            (4728, 7188, 5554, 12122, 4252, 6129, 14084, 5682, 7451, 7349),
        ),
        "mixed": (
            "RoboTwin-g4-j10-mixed-aloha_agilex-joint",
            (4820, 7412, 5709, 13462, 4258, 6320, 14571, 6038, 7712, 7683),
        ),
    }
    configs = []
    for distribution, (repo_id, frame_counts) in variants.items():
        configs.append(
            dataclasses.replace(
                template,
                name=f"pi05_g4_j10_{distribution}_builtin_dual_lora",
                data=dataclasses.replace(
                    template.data,
                    repo_id=repo_id,
                    assets=AssetsConfig(asset_id=repo_id),
                    base_config=DataConfig(
                        prompt_from_task=True,
                        task_frame_counts=frame_counts,
                        task_sampling_exponent=0.43,
                    ),
                ),
                policy_metadata=_g4_j10_policy_metadata(distribution),
                batch_size=32,
                seed=0,
                ema_decay=None,
                fsdp_devices=1,
                num_train_steps=35_748,
                lr_schedule=_optimizer.CosineDecaySchedule(
                    warmup_steps=1_000,
                    peak_lr=2.5e-5,
                    decay_steps=35_748,
                    decay_lr=2.5e-6,
                ),
            )
        )
    return configs


_CONFIGS.extend(_g4_j10_recipe_configs(_CONFIGS[0]))


_G51_TASKS = (
    "place_empty_cup",
    "place_container_plate",
    "press_stapler",
    "turn_switch",
    "adjust_bottle",
)
_G51_FRAME_COUNTS = (9133, 8398, 5990, 4892, 7188)


def _g51_policy_metadata(camera_condition: str, seed: int) -> dict[str, Any]:
    cameras = ["head"] if camera_condition == "head_only" else ["head", "left_wrist", "right_wrist"]
    return {
        "task_info": {
            "benchmark": "RoboTwin-2.0",
            "tasks": list(_G51_TASKS),
            "task_config": "demo_clean",
            "scene": "Easy",
        },
        "robot_config": {
            "embodiment": "aloha-agilex",
            "action_type": "joint",
            "state_action_dim": 14,
        },
        "input_config": {
            "camera_condition": camera_condition,
            "cameras": cameras,
            "language_instruction": True,
            "canonical_task_id_model_input": False,
        },
        "training_purpose": {
            "stage": "G5.1",
            "type": "matched-camera-adaptation",
            "controlled_axis": "camera_inputs",
            "training_seed": seed,
            "sampling_rule": "uniform-task-p-0.2",
            "reference_passes": 10,
        },
        "recipe": "builtin-dual-lora",
    }


def _g51_recipe_configs(base: TrainConfig) -> list[TrainConfig]:
    """Register the four matched G5.1 camera-condition/seed recipes."""
    template = next(
        config
        for config in _g31c_recipe_configs(base)
        if config.policy_metadata is not None and config.policy_metadata["recipe"] == "builtin-dual-lora"
    )
    assert isinstance(template.data, LeRobotAlohaDataConfig)
    repo_id = "RoboTwin-g51-easy5-clean-aloha_agilex-joint"
    cameras = {
        "head_only": ("cam_high",),
        "three_view": ("cam_high", "cam_left_wrist", "cam_right_wrist"),
    }
    configs = []
    for camera_condition, enabled_cameras in cameras.items():
        configs.extend(
            (
                dataclasses.replace(
                    template,
                    name=f"pi05_g51_easy5_{camera_condition}_s{seed}_builtin_dual_lora",
                    data=dataclasses.replace(
                        template.data,
                        repo_id=repo_id,
                        assets=AssetsConfig(asset_id=repo_id),
                        base_config=DataConfig(
                            prompt_from_task=True,
                            task_frame_counts=_G51_FRAME_COUNTS,
                            task_sampling_exponent=0.0,
                        ),
                        enabled_cameras=enabled_cameras,
                    ),
                    policy_metadata=_g51_policy_metadata(camera_condition, seed),
                    batch_size=32,
                    seed=seed,
                    ema_decay=None,
                    fsdp_devices=1,
                    num_train_steps=11_126,
                    lr_schedule=_optimizer.CosineDecaySchedule(
                        warmup_steps=311,
                        peak_lr=2.5e-5,
                        decay_steps=11_126,
                        decay_lr=2.5e-6,
                    ),
                )
            )
            for seed in (0, 1)
        )
    return configs


_CONFIGS.extend(_g51_recipe_configs(_CONFIGS[0]))


_G6_RBDJ_TASKS = (
    "stack_bowls",
    "cover_blocks",
    "insert_tubes",
    "fill_pen_holder",
)
_G6_RBDJ_FRAME_COUNTS = (18_558, 27_219, 16_144, 36_100)
_G6_RBDJ_EPISODE_INDICES = (
    *range(3_000, 3_050),  # stack_bowls
    *range(300, 350),  # cover_blocks
    *range(1_400, 1_450),  # insert_tubes
    *range(800, 850),  # fill_pen_holder
)
_G6_RBDJ_REPO_ID = "RoboDojo-g6-joint4-arx_x5-joint"


def _g6_rbdj_policy_metadata() -> dict[str, Any]:
    return {
        "task_info": {
            "benchmark": "RoboDojo",
            "tasks": list(_G6_RBDJ_TASKS),
            "held_out_eval_task": "stack_blocks_by_language",
            "source_revision": "cfb06d1dadcdf03ccc923102b0e3f3b0a68dfc43",
        },
        "robot_config": {
            "embodiment": "dual-arx-x5",
            "action_type": "absolute-joint-gripper",
            "state_action_dim": 14,
            "prediction_horizon": 50,
        },
        "input_config": {
            "cameras": ["head", "left_wrist", "right_wrist"],
            "language_instruction": True,
            "canonical_task_id_model_input": False,
        },
        "training_purpose": {
            "stage": "G6",
            "type": "robodojo-joint4-baseline",
            "training_seed": 0,
            "sampling_rule": "uniform-task-p-0.25",
            "reference_passes": 10,
            "candidate_frames": sum(_G6_RBDJ_FRAME_COUNTS),
        },
        "recipe": "builtin-dual-lora",
    }


def _g6_rbdj_recipe_config(base: TrainConfig) -> TrainConfig:
    """Register the authorized G6 RoboDojo four-task joint recipe."""
    template = next(
        config
        for config in _g31c_recipe_configs(base)
        if config.policy_metadata is not None and config.policy_metadata["recipe"] == "builtin-dual-lora"
    )
    assert isinstance(template.data, LeRobotAlohaDataConfig)
    return dataclasses.replace(
        template,
        name="pi05_g6_rbdj_joint4_s0_builtin_dual_lora",
        project_name="manip-pi05-rbdj",
        data=dataclasses.replace(
            template.data,
            repo_id=_G6_RBDJ_REPO_ID,
            assets=AssetsConfig(asset_id=_G6_RBDJ_REPO_ID),
            base_config=DataConfig(
                prompt_from_task=True,
                episode_indices=_G6_RBDJ_EPISODE_INDICES,
                task_frame_counts=_G6_RBDJ_FRAME_COUNTS,
                task_sampling_exponent=0.0,
            ),
            enabled_cameras=("cam_high", "cam_left_wrist", "cam_right_wrist"),
        ),
        policy_metadata=_g6_rbdj_policy_metadata(),
        batch_size=32,
        seed=0,
        ema_decay=None,
        fsdp_devices=1,
        num_train_steps=30_632,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=857,
            peak_lr=2.5e-5,
            decay_steps=30_632,
            decay_lr=2.5e-6,
        ),
    )


_CONFIGS.append(_g6_rbdj_recipe_config(_CONFIGS[0]))

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
