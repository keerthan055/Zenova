"""ZENOVA Safety Policy Verification Rules Subpackage."""
from zenova.safety.rules.base import BaseSafetyRule
from zenova.safety.rules.harmful import HarmfulInstructionsRule
from zenova.safety.rules.medical import MedicalClaimsRule
from zenova.safety.rules.diagnosis import UnsupportedDiagnosisRule
from zenova.safety.rules.crisis import CrisisMishandlingRule
from zenova.safety.rules.advice import UnsafeAdviceRule
from zenova.safety.rules.delusions import DelusionReinforcementRule
from zenova.safety.rules.dependency import InappropriateDependencyRule
from zenova.safety.rules.manipulation import ManipulativeLanguageRule
from zenova.safety.rules.authority import ProfessionalAuthorityRule
from zenova.safety.rules.privacy import PrivacyViolationRule
from zenova.safety.rules.resources import HallucinatedResourcesRule
from zenova.safety.rules.certainty import DangerousCertaintyRule

ALL_SAFETY_RULES = [
    HarmfulInstructionsRule,
    MedicalClaimsRule,
    UnsupportedDiagnosisRule,
    CrisisMishandlingRule,
    UnsafeAdviceRule,
    DelusionReinforcementRule,
    InappropriateDependencyRule,
    ManipulativeLanguageRule,
    ProfessionalAuthorityRule,
    PrivacyViolationRule,
    HallucinatedResourcesRule,
    DangerousCertaintyRule
]

__all__ = [
    "BaseSafetyRule",
    "HarmfulInstructionsRule",
    "MedicalClaimsRule",
    "UnsupportedDiagnosisRule",
    "CrisisMishandlingRule",
    "UnsafeAdviceRule",
    "DelusionReinforcementRule",
    "InappropriateDependencyRule",
    "ManipulativeLanguageRule",
    "ProfessionalAuthorityRule",
    "PrivacyViolationRule",
    "HallucinatedResourcesRule",
    "DangerousCertaintyRule",
    "ALL_SAFETY_RULES",
]
