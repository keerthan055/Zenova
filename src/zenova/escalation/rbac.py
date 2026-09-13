"""Role-Based Access Control (RBAC) and data sanitization for clinical oversight."""
from typing import Dict, Any, Optional
from zenova.schemas.escalation import UserRole
from zenova.core.logging import get_logger

logger = get_logger("zenova.escalation.rbac")


class AccessControlManager:
    """Manages role-based permissions and sensitive health information masking."""

    PERMISSIONS: Dict[UserRole, Dict[str, bool]] = {
        UserRole.CLINICIAN: {
            "can_view_queue": True,
            "can_view_full_context": True,
            "can_view_cues": True,
            "can_acknowledge": True,
            "can_resolve": True,
            "can_reassign": False,
            "can_view_audit": True,
            "can_edit_rules": False,
        },
        UserRole.TRIAGE_SUPERVISOR: {
            "can_view_queue": True,
            "can_view_full_context": True,
            "can_view_cues": True,
            "can_acknowledge": True,
            "can_resolve": True,
            "can_reassign": True,
            "can_view_audit": True,
            "can_edit_rules": True,
        },
        UserRole.SYSTEM_ADMIN: {
            "can_view_queue": True,
            "can_view_full_context": False,  # Masks raw patient dialogue
            "can_view_cues": False,
            "can_acknowledge": False,
            "can_resolve": False,
            "can_reassign": True,
            "can_view_audit": True,
            "can_edit_rules": True,
        },
        UserRole.AUDITOR: {
            "can_view_queue": True,
            "can_view_full_context": False,  # Sanitized for compliance
            "can_view_cues": False,
            "can_acknowledge": False,
            "can_resolve": False,
            "can_reassign": False,
            "can_view_audit": True,
            "can_edit_rules": False,
        },
        UserRole.PATIENT: {
            "can_view_queue": False,
            "can_view_full_context": False,
            "can_view_cues": False,
            "can_acknowledge": False,
            "can_resolve": False,
            "can_reassign": False,
            "can_view_audit": False,
            "can_edit_rules": False,
        },
    }

    @classmethod
    def check_permission(cls, role: UserRole, permission: str) -> bool:
        role_perms = cls.PERMISSIONS.get(role, {})
        return role_perms.get(permission, False)

    @classmethod
    def sanitize_alert_data(cls, alert_data: Dict[str, Any], role: UserRole) -> Dict[str, Any]:
        """Sanitizes patient dialogue and sensitive clinical cues based on role permissions."""
        if not cls.check_permission(role, "can_view_queue"):
            raise PermissionError(f"Role '{role.value}' is not authorized to view clinical escalation alerts.")

        sanitized = dict(alert_data)

        # Mask trigger cues if not authorized
        if not cls.check_permission(role, "can_view_cues"):
            if "trigger_cues" in sanitized:
                sanitized["trigger_cues"] = ["[REDACTED_CLINICAL_CUES]"]
            if "reason" in sanitized and isinstance(sanitized["reason"], dict):
                sanitized["reason"] = dict(sanitized["reason"])
                sanitized["reason"]["trigger_cues"] = ["[REDACTED_CLINICAL_CUES]"]

        # Mask sensitive user dialogue & context snapshot if not authorized
        if not cls.check_permission(role, "can_view_full_context"):
            if "context_snapshot" in sanitized and isinstance(sanitized["context_snapshot"], dict):
                ctx = dict(sanitized["context_snapshot"])
                ctx["last_user_message"] = "[PROTECTED_HEALTH_INFORMATION]"
                sanitized["context_snapshot"] = ctx
            if "notes" in sanitized and sanitized["notes"]:
                sanitized["notes"] = "[CONFIDENTIAL_CLINICAL_NOTES]"
            if "resolution_notes" in sanitized and sanitized["resolution_notes"]:
                sanitized["resolution_notes"] = "[CONFIDENTIAL_RESOLUTION_NOTES]"

        return sanitized
