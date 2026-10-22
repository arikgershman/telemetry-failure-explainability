"""Version 1 contracts shared by data generation, parsing, and model consumers.

Times are seconds, positions meters, forces newtons, joint angles radians,
joint torques N m, and quaternions ordered (w, x, y, z).
"""

from enum import Enum
from typing import Annotated, Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    """Reject unknown fields and non-finite numbers at every nested level."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Nonnegative = Annotated[Number, Field(ge=0)]
Name = Annotated[str, Field(min_length=1, pattern=r"\S")]
Vector3 = Annotated[list[Number], Field(min_length=3, max_length=3)]
Quaternion = Annotated[list[Number], Field(min_length=4, max_length=4)]
JointVector = Annotated[list[Number], Field(min_length=1)]
LOGGING_DT_S = 0.02


class Task(str, Enum):
    PICK_UP_OBJECT = "pick_up_object"
    TRANSPORT_OBJECT = "transport_object"


class Subtask(str, Enum):
    APPROACH = "approach"
    ALIGN = "align"
    GRASP = "grasp"
    LIFT = "lift"
    MAINTAIN_GRASP = "maintain_grasp"
    MOVE_ALONG_PATH = "move_along_path"
    REACH_TARGET = "reach_target"


TASK_SUBTASKS: dict[Task, tuple[Subtask, ...]] = {
    Task.PICK_UP_OBJECT: (Subtask.APPROACH, Subtask.ALIGN, Subtask.GRASP, Subtask.LIFT),
    Task.TRANSPORT_OBJECT: (
        Subtask.MAINTAIN_GRASP, Subtask.MOVE_ALONG_PATH, Subtask.REACH_TARGET,
    ),
}


class FailureMode(str, Enum):
    POSITION_ERROR = "position_error"
    ORIENTATION_ERROR = "orientation_error"
    INADEQUATE_GRIP_STRENGTH = "inadequate_grip_strength"
    EARLY_GRIPPER_RELEASE = "early_gripper_release"
    EXCESSIVE_SPEED = "excessive_speed"


class TelemetryFeature(str, Enum):
    JOINT_POSITIONS = "joint_positions"
    JOINT_VELOCITIES = "joint_velocities"
    JOINT_TORQUES = "joint_torques"
    GRIPPER_STATE = "gripper_state"
    GRIPPER_FORCE = "gripper_force"
    EE_POS = "ee_pos"
    EE_QUAT = "ee_quat"
    OBJ_POS = "obj_pos"
    OBJ_QUAT = "obj_quat"
    OBJ_VEL = "obj_vel"
    CONTACT_FORCE = "contact_force"


class RawTelemetryRow(ContractModel):
    timestamp: Nonnegative
    subtask: Subtask
    joint_positions: JointVector
    joint_velocities: JointVector
    joint_torques: JointVector
    gripper_state: Annotated[Number, Field(ge=0, le=1)]
    gripper_force: Nonnegative
    ee_pos: Vector3
    ee_quat: Quaternion
    obj_pos: Vector3
    obj_quat: Quaternion
    obj_vel: Vector3
    contact_force: Nonnegative

    @model_validator(mode="after")
    def matching_joint_dimensions(self) -> "RawTelemetryRow":
        if len({len(self.joint_positions), len(self.joint_velocities),
                len(self.joint_torques)}) != 1:
            raise ValueError("joint positions, velocities, and torques must have equal lengths")
        return self


class MetadataRow(ContractModel):
    pair_id: Name  # String preserves zero-padded identifiers, e.g. "000".
    task: Task
    success_file: Name
    failure_file: Name
    failed_subtask: Subtask
    failure_mode: FailureMode
    perturbation_param: Name
    perturbation_value: Number
    seed: Annotated[int, Field(strict=True, ge=0)]

    @model_validator(mode="after")
    def valid_pair(self) -> "MetadataRow":
        if self.failed_subtask not in TASK_SUBTASKS[self.task]:
            raise ValueError("failed_subtask must belong to task")
        if self.success_file == self.failure_file:
            raise ValueError("success_file and failure_file must differ")
        return self


class SubtaskMetadata(ContractModel):
    subgoal_name: Name
    action_description: Name


TASK_GOALS: dict[Task, str] = {
    Task.PICK_UP_OBJECT: "Pick up object",
    Task.TRANSPORT_OBJECT: "Transport object",
}
SUBTASK_METADATA: dict[Subtask, SubtaskMetadata] = {
    Subtask.APPROACH: SubtaskMetadata(subgoal_name="Approach", action_description="Move arm toward object"),
    Subtask.ALIGN: SubtaskMetadata(subgoal_name="Align", action_description="Position and orient gripper"),
    Subtask.GRASP: SubtaskMetadata(subgoal_name="Grasp", action_description="Close gripper around object"),
    Subtask.LIFT: SubtaskMetadata(subgoal_name="Lift", action_description="Lift object off support"),
    Subtask.MAINTAIN_GRASP: SubtaskMetadata(subgoal_name="Maintain grasp", action_description="Hold object securely"),
    Subtask.MOVE_ALONG_PATH: SubtaskMetadata(subgoal_name="Move along path", action_description="Move held object along intended path"),
    Subtask.REACH_TARGET: SubtaskMetadata(subgoal_name="Reach target", action_description="Bring held object to target position"),
}


class DeterministicDivergence(ContractModel):
    """Reserved metrics contract; this scaffold does not compute divergence.

    A future implementation must document alignment and thresholds. A missing
    subtask-level object (null) means not computed, never zero divergence.
    """

    first_divergence_timestamp: Nonnegative | None = None
    max_abs_deltas: dict[TelemetryFeature, Nonnegative] = Field(default_factory=dict)


def validate_observation_sequence(rows: Sequence[RawTelemetryRow]) -> None:
    """Check order and fixed joint dimensions for raw or compressed samples."""
    if not rows:
        raise ValueError("telemetry must contain at least one observation")
    joint_count = len(rows[0].joint_positions)
    for index, row in enumerate(rows):
        if len(row.joint_positions) != joint_count:
            raise ValueError("joint count must remain constant throughout a run")
        if index and row.timestamp <= rows[index - 1].timestamp:
            raise ValueError("timestamps must be strictly increasing with no duplicates")


class SubtaskBT(ContractModel):
    subtask: Subtask
    metadata: SubtaskMetadata
    success_samples: Annotated[list[RawTelemetryRow], Field(min_length=1)]
    failure_samples: Annotated[list[RawTelemetryRow], Field(min_length=1)]
    deterministic_divergence: DeterministicDivergence | None = None

    @model_validator(mode="after")
    def valid_samples(self) -> "SubtaskBT":
        for samples in (self.success_samples, self.failure_samples):
            validate_observation_sequence(samples)
            if any(row.subtask != self.subtask for row in samples):
                raise ValueError("every sample must match its BT subtask")
        return self


class PairedBehaviorTree(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    metadata: MetadataRow
    overall_goal: Name
    subtasks: Annotated[list[SubtaskBT], Field(min_length=1)]

    @model_validator(mode="after")
    def valid_structure(self) -> "PairedBehaviorTree":
        if tuple(node.subtask for node in self.subtasks) != TASK_SUBTASKS[self.metadata.task]:
            raise ValueError("BT subtasks must match the complete ordered task template")
        for field in ("success_samples", "failure_samples"):
            validate_observation_sequence([
                row for node in self.subtasks for row in getattr(node, field)
            ])
        if len(self.subtasks[0].success_samples[0].joint_positions) != len(
            self.subtasks[0].failure_samples[0].joint_positions
        ):
            raise ValueError("paired runs must use the same joint dimensions")
        return self

    def _view(self, *, include_success: bool, include_divergence: bool) -> dict[str, Any]:
        # Ground-truth failure labels, perturbations, filenames, and seeds must
        # never enter model prompts. Keep them only in the storage representation.
        nodes = []
        for node in self.subtasks:
            excluded = set()
            if not include_success:
                excluded.add("success_samples")
            if not include_divergence:
                excluded.add("deterministic_divergence")
            nodes.append(node.model_dump(mode="json", exclude=excluded))
        return {
            "schema_version": self.schema_version,
            "pair_id": self.metadata.pair_id,
            "task": self.metadata.task.value,
            "overall_goal": self.overall_goal,
            "subtasks": nodes,
        }

    def to_failure_only(self) -> dict[str, Any]:
        """Baselines 1 and 2: failure telemetry and semantic task structure."""
        return self._view(include_success=False, include_divergence=False)

    def to_paired_without_divergence(self) -> dict[str, Any]:
        """Baseline 3: paired telemetry and semantics, without divergence."""
        return self._view(include_success=True, include_divergence=False)

    def to_full_paired(self) -> dict[str, Any]:
        """Main model: paired telemetry, semantics, and divergence placeholders."""
        return self._view(include_success=True, include_divergence=True)
