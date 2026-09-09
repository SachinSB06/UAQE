"""Tests for UAQE Autonomous Optimization Cockpit Live Pipeline Indicator.

Verifies the 14 test cases specified in Phase 17:
1. INITIALIZING event -> stage 1 ACTIVE
2. INGESTING event -> stage 1 COMPLETED -> stage 2 ACTIVE
3. INSPECTING event -> stage 2 COMPLETED -> stage 3 ACTIVE
4. CALIBRATING -> stage 4 ACTIVE
5. PROFILING -> stage 5 ACTIVE
6. SEARCHING -> stage 6 ACTIVE
7. EVALUATING -> stage 7 ACTIVE
8. SELECTING -> stage 8 ACTIVE
9. VALIDATING -> stage 9 ACTIVE
10. PACKAGING -> stage 10 ACTIVE
11. COMPLETED -> final state COMPLETED
12. FAILED -> current stage FAILED
13. Job isolation -> job A stage state never appears in job B
14. SSE reconnect -> current stage remains correct without resetting to INITIALIZING
"""

import unittest
from typing import List, Dict, Any, Optional, Tuple

PIPELINE_STAGES = [
    "INITIALIZING",
    "INGESTING",
    "INSPECTING",
    "CALIBRATING",
    "PROFILING",
    "SEARCHING",
    "EVALUATING",
    "SELECTING",
    "VALIDATING",
    "PACKAGING",
    "COMPLETED",
]


def map_to_pipeline_stage(raw: Optional[str]) -> Optional[str]:
    """Python mirror of frontend mapToPipelineStage."""
    if not raw:
        return None
    s = raw.strip().upper()

    if s in PIPELINE_STAGES:
        return s

    if s in ("INITIALIZATION", "INITIALIZE", "INIT", "OPTIMIZATION_STARTED"):
        return "INITIALIZING"
    if s in ("INGEST", "INGESTION", "DATASET_INGESTING", "MODEL_INGESTING"):
        return "INGESTING"
    if s in ("INSPECT", "INSPECTION", "AUDITING", "COMPATIBILITY_CHECK"):
        return "INSPECTING"
    if s in ("CALIBRATE", "CALIBRATION", "PREPROCESSING", "ADAPTING"):
        return "CALIBRATING"
    if s in ("PROFILE", "FP32_BASELINE", "BASELINE_MEASUREMENT", "BENCHMARKING"):
        return "PROFILING"
    if s in ("SEARCH", "CANDIDATE_EXPLORATION", "GENERATING_CANDIDATES"):
        return "SEARCHING"
    if s in ("EVALUATE", "EVALUATION", "CANDIDATE_START", "CANDIDATE_DONE", "EVALUATING_CANDIDATE"):
        return "EVALUATING"
    if s in ("SELECT", "SELECTION", "BEST_CANDIDATE_SELECTED"):
        return "SELECTING"
    if s in ("VALIDATE", "VALIDATION", "VALIDATING_CONSTRAINTS"):
        return "VALIDATING"
    if s in ("PACKAGE", "PACKAGING_ARTIFACTS", "EXPORTING"):
        return "PACKAGING"
    if s in ("COMPLETE", "COMPLETED", "OPTIMIZATION_COMPLETED", "FINAL_RESULT", "DONE"):
        return "COMPLETED"

    return None


def compute_stage_states(
    events: List[Dict[str, Any]],
    is_complete: bool = False,
    is_failed: bool = False,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    """Compute currentStage, states of all 11 stages, and operationDetail."""
    filtered = [e for e in events if not job_id or not e.get("job_id") or e.get("job_id") == job_id]

    detected_stage = None
    operation_detail = None
    has_failure = is_failed
    failed_at = None

    if is_complete:
        detected_stage = "COMPLETED"
        operation_detail = "Autonomous optimization complete"
    else:
        for ev in reversed(filtered):
            ev_type = (ev.get("type") or "").upper()
            if ev_type in ("ERROR", "FAILED", "BASELINE_INVALID"):
                has_failure = True
                operation_detail = ev.get("error") or ev.get("message") or "Optimization halted"
                if ev.get("stage") and ev.get("stage").upper() != "FAILED":
                    failed_at = map_to_pipeline_stage(ev.get("stage"))
                continue

            stage_from_field = map_to_pipeline_stage(ev.get("stage"))
            stage_from_type = map_to_pipeline_stage(ev.get("type")) if ev_type != "TELEMETRY" else None
            stage_from_phase = map_to_pipeline_stage(ev.get("phase")) if ev.get("phase", "").upper() != "EXECUTING" else None

            stage_found = stage_from_field or stage_from_type or stage_from_phase
            if not detected_stage and stage_found:
                detected_stage = stage_found
                if has_failure and not failed_at:
                    failed_at = stage_found

            if not operation_detail:
                if ev.get("candidate_id") and stage_found in ("EVALUATING", "SELECTING"):
                    operation_detail = f"Candidate {ev.get('candidate_id')}"
                elif ev.get("message") and "telemetry" not in ev.get("message", "").lower():
                    operation_detail = ev.get("message")

            if detected_stage and operation_detail:
                break

    if has_failure:
        detected_stage = "FAILED"
        if not failed_at:
            failed_at = "INITIALIZING"

    if not detected_stage:
        detected_stage = "INITIALIZING"

    stage_idx = PIPELINE_STAGES.index(failed_at) if is_failed and failed_at in PIPELINE_STAGES else (
        PIPELINE_STAGES.index(detected_stage) if detected_stage in PIPELINE_STAGES else 0
    )

    stage_states = {}
    for idx, stage in enumerate(PIPELINE_STAGES):
        if is_complete:
            stage_states[stage] = "COMPLETED"
        elif is_failed:
            if idx < stage_idx:
                stage_states[stage] = "COMPLETED"
            elif idx == stage_idx:
                stage_states[stage] = "FAILED"
            else:
                stage_states[stage] = "PENDING"
        else:
            if idx < stage_idx:
                stage_states[stage] = "COMPLETED"
            elif idx == stage_idx:
                stage_states[stage] = "ACTIVE"
            else:
                stage_states[stage] = "PENDING"

    return {
        "current_stage": detected_stage,
        "operation_detail": operation_detail,
        "stage_states": stage_states,
        "is_failed": is_failed,
        "is_complete": is_complete
    }


class TestLivePipelineIndicator(unittest.TestCase):

    def test_01_initializing_event_stage_1_active(self):
        events = [{"type": "stage_start", "stage": "INITIALIZATION", "message": "Initializing..."}]
        res = compute_stage_states(events)
        self.assertEqual(res["current_stage"], "INITIALIZING")
        self.assertEqual(res["stage_states"]["INITIALIZING"], "ACTIVE")
        self.assertEqual(res["stage_states"]["INGESTING"], "PENDING")

    def test_02_ingesting_event_stage_1_completed_stage_2_active(self):
        events = [
            {"type": "stage_start", "stage": "INITIALIZING"},
            {"type": "stage_start", "stage": "INGESTING", "message": "Ingesting model and dataset"}
        ]
        res = compute_stage_states(events)
        self.assertEqual(res["current_stage"], "INGESTING")
        self.assertEqual(res["stage_states"]["INITIALIZING"], "COMPLETED")
        self.assertEqual(res["stage_states"]["INGESTING"], "ACTIVE")
        self.assertEqual(res["stage_states"]["INSPECTING"], "PENDING")

    def test_03_inspecting_event_stage_2_completed_stage_3_active(self):
        events = [
            {"type": "stage_start", "stage": "INITIALIZING"},
            {"type": "stage_start", "stage": "INGESTING"},
            {"type": "stage_start", "stage": "INSPECTING", "message": "Auditing model capabilities"}
        ]
        res = compute_stage_states(events)
        self.assertEqual(res["current_stage"], "INSPECTING")
        self.assertEqual(res["stage_states"]["INITIALIZING"], "COMPLETED")
        self.assertEqual(res["stage_states"]["INGESTING"], "COMPLETED")
        self.assertEqual(res["stage_states"]["INSPECTING"], "ACTIVE")
        self.assertEqual(res["stage_states"]["CALIBRATING"], "PENDING")

    def test_04_calibrating_stage_4_active(self):
        events = [
            {"type": "stage_start", "stage": "INITIALIZING"},
            {"type": "stage_start", "stage": "INGESTING"},
            {"type": "stage_start", "stage": "INSPECTING"},
            {"type": "stage_start", "stage": "CALIBRATING"}
        ]
        res = compute_stage_states(events)
        self.assertEqual(res["current_stage"], "CALIBRATING")
        self.assertEqual(res["stage_states"]["CALIBRATING"], "ACTIVE")
        self.assertEqual(res["stage_states"]["PROFILING"], "PENDING")

    def test_05_profiling_stage_5_active(self):
        events = [
            {"type": "stage_start", "stage": "INITIALIZING"},
            {"type": "stage_start", "stage": "INGESTING"},
            {"type": "stage_start", "stage": "INSPECTING"},
            {"type": "stage_start", "stage": "CALIBRATING"},
            {"type": "stage_start", "stage": "PROFILING", "message": "Host CPU performance benchmark"}
        ]
        res = compute_stage_states(events)
        self.assertEqual(res["current_stage"], "PROFILING")
        self.assertEqual(res["stage_states"]["PROFILING"], "ACTIVE")
        self.assertEqual(res["operation_detail"], "Host CPU performance benchmark")

    def test_06_searching_stage_6_active(self):
        events = [
            {"type": "stage_start", "stage": "PROFILING"},
            {"type": "fp32_baseline", "accuracy": 0.94, "latency_ms": 10.0},
            {"type": "stage_start", "stage": "SEARCHING", "message": "Candidate exploration"}
        ]
        res = compute_stage_states(events)
        self.assertEqual(res["current_stage"], "SEARCHING")
        self.assertEqual(res["stage_states"]["SEARCHING"], "ACTIVE")

    def test_07_evaluating_stage_7_active(self):
        events = [
            {"type": "stage_start", "stage": "SEARCHING"},
            {"type": "candidate_start", "candidate_id": "cand_001", "candidate_name": "W8A8_PTQ"}
        ]
        res = compute_stage_states(events)
        self.assertEqual(res["current_stage"], "EVALUATING")
        self.assertEqual(res["stage_states"]["EVALUATING"], "ACTIVE")
        self.assertEqual(res["operation_detail"], "Candidate cand_001")

    def test_08_selecting_stage_8_active(self):
        events = [
            {"type": "candidate_done", "candidate_id": "cand_001"},
            {"type": "best_candidate_selected", "candidate": {"candidate_id": "cand_001"}, "message": "Candidate cand_001"}
        ]
        res = compute_stage_states(events)
        self.assertEqual(res["current_stage"], "SELECTING")
        self.assertEqual(res["stage_states"]["SELECTING"], "ACTIVE")

    def test_09_validating_stage_9_active(self):
        events = [
            {"type": "best_candidate_selected", "candidate": {"candidate_id": "cand_001"}},
            {"type": "stage_start", "stage": "VALIDATING", "message": "Validating constraints"}
        ]
        res = compute_stage_states(events)
        self.assertEqual(res["current_stage"], "VALIDATING")
        self.assertEqual(res["stage_states"]["VALIDATING"], "ACTIVE")

    def test_10_packaging_stage_10_active(self):
        events = [
            {"type": "stage_start", "stage": "VALIDATING"},
            {"type": "stage_start", "stage": "PACKAGING", "message": "Packaging deployment artifacts"}
        ]
        res = compute_stage_states(events)
        self.assertEqual(res["current_stage"], "PACKAGING")
        self.assertEqual(res["stage_states"]["PACKAGING"], "ACTIVE")

    def test_11_completed_final_state_completed(self):
        events = [
            {"type": "stage_start", "stage": "PACKAGING"},
            {"type": "complete", "stage": "COMPLETED", "message": "Autonomous optimization complete"}
        ]
        res = compute_stage_states(events, is_complete=True)
        self.assertEqual(res["current_stage"], "COMPLETED")
        for stage in PIPELINE_STAGES:
            self.assertEqual(res["stage_states"][stage], "COMPLETED")

    def test_12_failed_current_stage_failed(self):
        events = [
            {"type": "stage_start", "stage": "INITIALIZING"},
            {"type": "stage_start", "stage": "INGESTING"},
            {"type": "stage_start", "stage": "PROFILING"},
            {"type": "baseline_invalid", "stage": "FAILED", "error": "INVALID_BASELINE: Reference accuracy too low"}
        ]
        res = compute_stage_states(events, is_failed=True)
        self.assertEqual(res["current_stage"], "FAILED")
        self.assertEqual(res["stage_states"]["INITIALIZING"], "COMPLETED")
        self.assertEqual(res["stage_states"]["INGESTING"], "COMPLETED")
        self.assertEqual(res["stage_states"]["PROFILING"], "FAILED")
        self.assertEqual(res["stage_states"]["SEARCHING"], "PENDING")
        self.assertIn("INVALID_BASELINE", res["operation_detail"])

    def test_13_job_isolation(self):
        events = [
            {"job_id": "JOB-A", "type": "stage_start", "stage": "PACKAGING"},
            {"job_id": "JOB-B", "type": "stage_start", "stage": "INITIALIZING"}
        ]
        res_b = compute_stage_states(events, job_id="JOB-B")
        self.assertEqual(res_b["current_stage"], "INITIALIZING")
        self.assertEqual(res_b["stage_states"]["INITIALIZING"], "ACTIVE")
        self.assertEqual(res_b["stage_states"]["PACKAGING"], "PENDING")

    def test_14_sse_reconnect_preserves_stage(self):
        # Simulating reconnect: the event stream re-emits accumulated events up to EVALUATING
        events = [
            {"type": "stage_start", "stage": "INITIALIZING"},
            {"type": "stage_start", "stage": "INGESTING"},
            {"type": "stage_start", "stage": "INSPECTING"},
            {"type": "stage_start", "stage": "CALIBRATING"},
            {"type": "stage_start", "stage": "PROFILING"},
            {"type": "stage_start", "stage": "SEARCHING"},
            {"type": "candidate_start", "candidate_id": "cand_001"},
            {"type": "telemetry", "cpu_system_pct": 25.0}  # Reconnect ping/telemetry
        ]
        res = compute_stage_states(events)
        # Should stay at EVALUATING and NOT reset to INITIALIZING
        self.assertEqual(res["current_stage"], "EVALUATING")
        self.assertEqual(res["stage_states"]["INITIALIZING"], "COMPLETED")
        self.assertEqual(res["stage_states"]["PROFILING"], "COMPLETED")
        self.assertEqual(res["stage_states"]["EVALUATING"], "ACTIVE")


if __name__ == "__main__":
    unittest.main()
