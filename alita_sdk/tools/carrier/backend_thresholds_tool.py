import logging
import json
from typing import Type, Optional, List, Dict, Tuple

from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from .api_wrapper import CarrierAPIWrapper
from .backend_reports_tool import BaseCarrierTool
from .backend_tests_tool import GetBackendTestsTool

logger = logging.getLogger(__name__)


# =============================
# Input Schemas
# =============================
class ShowTestsInput(BaseModel):
    pass


class GetRequestsInput(BaseModel):
    test_name: str = Field(..., description="Backend test name")
    environment: str = Field(..., description="Environment name")


class ThresholdConfig(BaseModel):
    test: Optional[str] = Field(None, description="Backend test name")
    environment: Optional[str] = Field(None, description="Environment name")
    scope: str = Field(..., description="'all', 'every', or specific request name")
    target: str = Field(..., description="'response_time' | 'throughput' | 'error_rate'")
    aggregation: str = Field(..., description="'max' | 'min' | 'avg' | 'pct95' | 'pct50'")
    comparison: str = Field(..., description=">, >=, <, <=, == or API style gt/gte/lt/lte/eq")
    value: float = Field(..., description="Threshold numeric value")


class CreateThresholdInput(BaseModel):
    test_name: Optional[str] = Field(None, description="Backend test name (optional if provided in threshold_config)")
    environment: Optional[str] = Field(None, description="Environment (optional if provided in threshold_config)")
    threshold_config: ThresholdConfig = Field(..., description="Threshold configuration body")


# =============================
# Tools
# =============================
class ShowBackendTestsAndEnvsTool(BaseTool):
    api_wrapper: CarrierAPIWrapper = Field(...)
    name: str = "show_backend_tests_and_envs"
    description: str = "List available backend tests with their environments to guide threshold creation."
    args_schema: Type[BaseModel] = ShowTestsInput

    def _run(self) -> str:
        """Use the extended GetBackendTestsTool to show tests with environments."""
        try:
            # Create GetBackendTestsTool instance and use its extended method
            backend_tests_tool = GetBackendTestsTool(api_wrapper=self.api_wrapper)
            return backend_tests_tool.get_tests_with_environments()
        except Exception as e:
            logger.exception("Failed to list tests and envs")
            raise ToolException(str(e))


class GetBackendRequestsTool(BaseTool):
    api_wrapper: CarrierAPIWrapper = Field(...)
    name: str = "get_backend_requests"
    description: str = "List request names for a backend test/environment to help scope thresholds."
    args_schema: Type[BaseModel] = GetRequestsInput

    def _run(self, test_name: str, environment: str) -> str:
        try:
            requests = self.api_wrapper.get_backend_requests(test_name, environment)
            parts = [f"📋 Requests for '{test_name}' in '{environment}':", ""]
            if requests:
                parts.extend([f"  - {r}" for r in requests])
            else:
                parts.append("  - No specific requests found; you can still use scope 'all' or 'every', or a known request name.")

            parts.append("")
            parts += ThresholdsHelper.get_configuration_guide(include_template=True, test_name=test_name, environment=environment)
            parts += ThresholdsHelper.quick_examples()
            return "\n".join(parts)
        except Exception as e:
            # Graceful fallback: provide guidance even if the API call fails
            logger.warning(f"Failed to fetch requests for {test_name}/{environment}: {e}")
            parts = [
                f"⚠️ Couldn't retrieve request names for '{test_name}' in '{environment}'.",
                "You can still create a threshold:",
                " - Use scope 'all' to apply to the entire test",
                " - Use scope 'every' for response_time across all requests",
                " - Or provide a specific request name if you know it.",
                "",
            ]
            parts += ThresholdsHelper.get_configuration_guide(include_template=True, test_name=test_name, environment=environment)
            parts += ThresholdsHelper.quick_examples()
            return "\n".join(parts)


class CreateBackendThresholdTool(BaseCarrierTool):
    api_wrapper: CarrierAPIWrapper = Field(...)
    name: str = "create_backend_threshold"
    description: str = "Create a backend performance threshold for a test/environment."
    args_schema: Type[BaseModel] = CreateThresholdInput

    def _run(self, threshold_config: Dict, test_name: Optional[str] = None, environment: Optional[str] = None) -> str:
        try:
            cfg = dict(threshold_config)
            if test_name and not cfg.get("test"):
                cfg["test"] = test_name
            if environment and not cfg.get("environment"):
                cfg["environment"] = environment

            # Validate and normalize
            ok, err = ThresholdsHelper.validate_and_normalize(cfg)
            if not ok:
                raise ToolException(err)

            resp = self.api_wrapper.create_backend_threshold(cfg)

            # Human friendly comparison
            cmp_disp = ThresholdsHelper.api_to_user_comparison(cfg.get("comparison"))
            lines = [
                "✅ Threshold created",
                "",
                "Details:",
                f"  - Test: {cfg['test']}",
                f"  - Environment: {cfg['environment']}",
                f"  - Scope: {cfg['scope']}",
                f"  - Target: {cfg['target']}",
                f"  - Aggregation: {cfg['aggregation']}",
                f"  - Comparison: {cmp_disp}",
                f"  - Value: {cfg['value']}",
                "",
                f"Response: {json.dumps(resp, indent=2) if not isinstance(resp, str) else resp}"
            ]
            return "\n".join(lines)
        except Exception as e:
            logger.exception("Failed to create backend threshold")
            raise ToolException(str(e))


class ListBackendThresholdsTool(BaseTool):
    api_wrapper: CarrierAPIWrapper = Field(...)
    name: str = "list_backend_thresholds"
    description: str = "List existing backend thresholds."
    args_schema: Type[BaseModel] = ShowTestsInput

    def _run(self) -> str:
        try:
            data = self.api_wrapper.get_backend_thresholds()
            rows = data.get("rows") if isinstance(data, dict) else data
            if not rows:
                return "No backend thresholds found."
            
            lines = [f"Found {len(rows)} threshold{'s' if len(rows) != 1 else ''}:", ""]
            for t in rows:
                cmp_disp = ThresholdsHelper.api_to_user_comparison(t.get("comparison"))
                lines.append(
                    f"ID: {t.get('id')} - Test: {t.get('test')} - Env: {t.get('environment')} - "
                    f"Scope: {t.get('scope')} - Target: {t.get('target')} ({t.get('aggregation')}) {cmp_disp} {t.get('value')}"
                )
            return "\n".join(lines)
        except Exception as e:
            logger.exception("Failed to list backend thresholds")
            raise ToolException(str(e))


class DeleteBackendThresholdTool(BaseTool):
    api_wrapper: CarrierAPIWrapper = Field(...)
    name: str = "delete_backend_threshold"
    description: str = "Delete a backend threshold by id, or list available thresholds if id not provided."
    class _Args(BaseModel):
        threshold_id: Optional[str] = Field(None, description="Threshold ID to delete")
    args_schema: Type[BaseModel] = _Args

    def _run(self, threshold_id: Optional[str] = None) -> str:
        try:
            if not threshold_id:
                # Reuse the same logic as ListBackendThresholdsTool
                data = self.api_wrapper.get_backend_thresholds()
                rows = data.get("rows") if isinstance(data, dict) else data
                if not rows:
                    return "No backend thresholds found."
                lines = [f"Found {len(rows)} thresholds. Provide threshold_id to delete:", ""]
                for t in rows:
                    cmp_disp = ThresholdsHelper.api_to_user_comparison(t.get("comparison"))
                    lines.append(
                        f"ID: {t.get('id')} - Test: {t.get('test')} - Env: {t.get('environment')} - "
                        f"Scope: {t.get('scope')} - Target: {t.get('target')} ({t.get('aggregation')}) {cmp_disp} {t.get('value')}"
                    )
                return "\n".join(lines)

            self.api_wrapper.delete_backend_threshold(threshold_id)
            return f"✅ Deleted backend threshold {threshold_id}"
        except Exception as e:
            logger.exception("Failed to delete backend threshold")
            raise ToolException(str(e))


class UpdateBackendThresholdTool(BaseTool):
    api_wrapper: CarrierAPIWrapper = Field(...)
    name: str = "update_backend_threshold"
    description: str = "Update an existing backend threshold by id."
    class _Args(BaseModel):
        threshold_id: Optional[str] = Field(None, description="Threshold ID to update")
        threshold_config: Optional[Dict] = Field(None, description="Fields to update")
    args_schema: Type[BaseModel] = _Args

    def _run(self, threshold_id: Optional[str] = None, threshold_config: Optional[Dict] = None) -> str:
        try:
            if not threshold_id:
                # Show available thresholds
                data = self.api_wrapper.get_backend_thresholds()
                rows = data.get("rows") if isinstance(data, dict) else data
                if not rows:
                    return "No backend thresholds found."
                parts = [f"🔧 Found {len(rows)} thresholds:", ""]
                for t in rows:
                    parts += [
                        f"ID: {t.get('id')}",
                        f"  - Test: {t.get('test')}",
                        f"  - Environment: {t.get('environment')}",
                        f"  - Scope: {t.get('scope')}",
                        f"  - Target: {t.get('target')}",
                        f"  - Aggregation: {t.get('aggregation')}",
                        f"  - Comparison: {t.get('comparison')}",
                        f"  - Value: {t.get('value')}",
                        ""
                    ]
                parts += [
                    "Provide: threshold_id and threshold_config with fields to update"
                ]
                return "\n".join(parts)

            if not threshold_config:
                # Show specific threshold for guidance
                data = self.api_wrapper.get_backend_thresholds()
                rows = data.get("rows") if isinstance(data, dict) else data
                cur = next((t for t in rows or [] if str(t.get('id')) == str(threshold_id)), None)
                if not cur:
                    return f"❌ Threshold {threshold_id} not found."
                cmp_disp = ThresholdsHelper.api_to_user_comparison(cur.get("comparison"))
                parts = [
                    f"Current configuration for {threshold_id}:",
                    f"  - Test: {cur.get('test')}",
                    f"  - Environment: {cur.get('environment')}",
                    f"  - Scope: {cur.get('scope')}",
                    f"  - Target: {cur.get('target')}",
                    f"  - Aggregation: {cur.get('aggregation')}",
                    f"  - Comparison: {cmp_disp}",
                    f"  - Value: {cur.get('value')}"
                ]
                parts += ThresholdsHelper.get_configuration_guide(include_template=False)
                return "\n".join(parts)

            cfg = dict(threshold_config)
            ok, err = ThresholdsHelper.validate_and_normalize(cfg)
            if not ok:
                return f"❌ {err}"

            # Merge: API expects full object; fetch existing to fill required fields
            data = self.api_wrapper.get_backend_thresholds()
            rows = data.get("rows") if isinstance(data, dict) else data
            cur = next((t for t in rows or [] if str(t.get('id')) == str(threshold_id)), None)
            if not cur:
                return f"❌ Threshold {threshold_id} not found."

            required = ["test", "environment", "scope", "target", "aggregation", "comparison", "value"]
            body = {k: cur[k] for k in required if k in cur}
            body.update(cfg)
            body["id"] = int(threshold_id)

            # Make the API call and get response
            resp = self.api_wrapper.update_backend_threshold(threshold_id, body)
            
            # Verify the update by fetching the current state
            updated_data = self.api_wrapper.get_backend_thresholds()
            updated_rows = updated_data.get("rows") if isinstance(updated_data, dict) else updated_data
            updated_threshold = next((t for t in updated_rows or [] if str(t.get('id')) == str(threshold_id)), None)
            
            if not updated_threshold:
                return f"❌ Failed to verify update for threshold {threshold_id}"
            
            # Display the actual updated values from the API
            cmp_disp = ThresholdsHelper.api_to_user_comparison(updated_threshold.get("comparison"))
            out = [
                f"✅ Updated threshold {threshold_id}",
                "",
                "Verified updated configuration:",
                f"  - Test: {updated_threshold.get('test')}",
                f"  - Environment: {updated_threshold.get('environment')}",
                f"  - Scope: {updated_threshold.get('scope')}",
                f"  - Target: {updated_threshold.get('target')}",
                f"  - Aggregation: {updated_threshold.get('aggregation')}",
                f"  - Comparison: {cmp_disp}",
                f"  - Value: {updated_threshold.get('value')}"
            ]
            return "\n".join(out)
        except Exception as e:
            logger.exception("Failed to update backend threshold")
            raise ToolException(str(e))


# =============================
# Helper
# =============================
class ThresholdsHelper:
    valid_targets = {"response_time", "throughput", "error_rate"}
    valid_aggs = {"max", "min", "avg", "pct95", "pct50"}
    user_to_api_cmp = {">": "gt", ">=": "gte", "<": "lt", "<=": "lte", "==": "eq"}
    api_to_user_cmp = {v: k for k, v in user_to_api_cmp.items()}

    @classmethod
    def validate_and_normalize(cls, cfg: Dict) -> Tuple[bool, str]:
        # Required fields
        required = ["test", "environment", "scope", "target", "aggregation", "comparison", "value"]
        missing = [f for f in required if f not in cfg]
        if missing:
            return False, f"Missing required fields: {missing}"

        # target
        if cfg["target"] not in cls.valid_targets:
            return False, f"Invalid target '{cfg['target']}'. Valid: {sorted(cls.valid_targets)}"
        # aggregation
        if cfg["aggregation"] not in cls.valid_aggs:
            return False, f"Invalid aggregation '{cfg['aggregation']}'. Valid: {sorted(cls.valid_aggs)}"
        # comparison
        comp = cfg.get("comparison")
        if comp in cls.user_to_api_cmp:
            cfg["comparison"] = cls.user_to_api_cmp[comp]
        elif comp not in cls.api_to_user_cmp:
            return False, "Invalid comparison. Use one of >, >=, <, <=, == or gt/gte/lt/lte/eq"

        # value
        try:
            cfg["value"] = float(cfg["value"])
        except Exception:
            return False, "'value' must be numeric"

        return True, ""

    @classmethod
    def api_to_user_comparison(cls, cmp_val: Optional[str]) -> str:
        if not cmp_val:
            return ""
        return cls.api_to_user_cmp.get(cmp_val, str(cmp_val))

    @classmethod
    def get_configuration_guide(cls, include_template: bool = True, test_name: Optional[str] = None,
                                environment: Optional[str] = None) -> List[str]:
        parts: List[str] = []
        if include_template:
            parts += [
                "🛠️ Threshold template:",
                "{",
                f'  "test": "{test_name or "<TEST_NAME>"}",',
                f'  "environment": "{environment or "<ENV_NAME>"}",',
                '  "scope": "all",',
                '  "target": "response_time",',
                '  "aggregation": "pct95",',
                '  "comparison": ">=",',
                '  "value": 1000',
                "}",
                ""
            ]
        parts += [
            "🔍 Comparison types: >, >=, <, <=, ==",
            "Targets: response_time, throughput, error_rate",
            "Aggregations: max, min, avg, pct95, pct50"
        ]
        return parts

    @classmethod
    def quick_examples(cls) -> List[str]:
        return [
            "💡 Examples:",
            '- Error rate: {"scope": "all", "target": "error_rate", "comparison": ">", "value": 10}',
            '- Throughput: {"scope": "all", "target": "throughput", "comparison": "<", "value": 3}',
            '- Response time: {"scope": "every", "target": "response_time", "comparison": ">", "value": 3000}',
            'Note: "every" scope applies only to response_time.'
        ]


__all__ = [
    "ShowBackendTestsAndEnvsTool",
    "GetBackendRequestsTool",
    "CreateBackendThresholdTool",
    "ListBackendThresholdsTool",
    "DeleteBackendThresholdTool",
    "UpdateBackendThresholdTool",
]
