from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathoScreenPaths:
    output_root: Path
    pathway_id: int

    @property
    def p_dir(self):
        return self.output_root / f"P{self.pathway_id}"

    @property
    def checkpoints_dir(self):
        return self.p_dir / "checkpoints"

    @property
    def calibration_dir(self):
        return self.p_dir / "calibration"

    @property
    def predictions_dir(self):
        return self.p_dir / "predictions"

    @property
    def logs_dir(self):
        return self.p_dir / "logs"

    @property
    def metrics_dir(self):
        return self.p_dir / "metrics"

    def ensure(self) -> None:
        for d in [
            self.checkpoints_dir,
            self.calibration_dir,
            self.predictions_dir,
            self.logs_dir,
            self.metrics_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def checkpoint_best(self):
        return self.checkpoints_dir / "best.pt"

    def checkpoint_last(self):
        return self.checkpoints_dir / "last.pt"

    def calibrator_path(self):
        return self.calibration_dir / "isotonic.pkl"

    def brier_path(self) :
        return self.calibration_dir / "brier.json"
