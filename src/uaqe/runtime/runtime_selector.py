from __future__ import annotations
from typing import Optional, List
from uaqe.domain.hardware_manager import HardwareProfile
from uaqe.common.exceptions import ExportError

class RuntimeSelector:
    """Selects and validates the target runtime for a run, based on hardware constraints
    and explicit user overrides.
    """
    
    def select_runtime(self, profile: HardwareProfile, requested_runtime: Optional[str] = None) -> str:
        supported = getattr(profile, "supported_runtimes", [])
        default = getattr(profile, "default_runtime", None)
        
        if not default:
            raise ExportError(
                "HardwareProfile does not define a default_runtime.", 
                code="INVALID_HARDWARE_PROFILE"
            )
        if default not in supported:
            raise ExportError(
                f"HardwareProfile default_runtime '{default}' is not in supported_runtimes {supported}.",
                code="INVALID_HARDWARE_PROFILE"
            )
            
        if requested_runtime is None:
            print(f"No runtime specified. Using default runtime '{default}'. Specify --runtime to override.")
            runtime = default
        else:
            if requested_runtime not in supported:
                raise ExportError(
                    f"Requested runtime '{requested_runtime}' is not supported by hardware profile '{profile.profile_id}'. "
                    f"Supported runtimes: {supported}.",
                    code="UNSUPPORTED_RUNTIME"
                )
            runtime = requested_runtime
            
        return runtime
