import re
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class RuleViolation:
    rule_id: str
    category: str
    severity: str  # "high", "medium", "low"
    matched_text: str
    description: str


# ---------------------------------------------------------------------------
# Rule Definitions
# ---------------------------------------------------------------------------

COMPLIANCE_RULES = [
    # --- Health & Medical Claims ---
    {
        "rule_id": "HEALTH_001",
        "category": "Unsubstantiated Health Claims",
        "severity": "high",
        "patterns": [
            r"\bcures?\b.{0,30}\b(disease|cancer|diabetes|covid|virus|infection)\b",
            r"\b(guaranteed|100%)\s+(results?|cure|recovery|healing)\b",
            r"\bclinically\s+proven\b",
            r"\bFDA\s+approved\b",
            r"\blose\s+\d+\s+(pounds?|kgs?)\s+in\s+\d+\s+(days?|weeks?)\b",
            r"\bno\s+side\s+effects?\b",
            r"\bmiracl(e|ulous)\b.{0,20}\b(cure|result|remedy|treatment)\b",
        ],
        "description": "Contains unsubstantiated health or medical claims",
    },
    # --- Financial Claims ---
    {
        "rule_id": "FIN_001",
        "category": "Misleading Financial Claims",
        "severity": "high",
        "patterns": [
            r"\bguaranteed\s+(returns?|profits?|income|earnings?)\b",
            r"\bzero\s+risk\b",
            r"\b(double|triple)\s+your\s+(money|investment|income)\b",
            r"\bmake\s+\$\d+\s+(per|a)\s+(day|week|month)\b",
            r"\bget\s+rich\s+quick\b",
            r"\bpassive\s+income\s+guaranteed\b",
            r"\b100%\s+(safe|secure)\s+investment\b",
        ],
        "description": "Contains misleading financial claims or guarantees",
    },
    # --- False Advertising ---
    {
        "rule_id": "AD_001",
        "category": "False Advertising",
        "severity": "medium",
        "patterns": [
            r"\b#\s*1\s+(in\s+the\s+world|globally|worldwide)\b",
            r"\bbest\s+in\s+the\s+world\b",
            r"\b(only|first)\s+product\s+(that|to)\b",
            r"\b100%\s+(natural|organic|pure)\b",
            r"\binstant\s+(results?|cure|fix|solution)\b",
            r"\bworks?\s+(instantly|immediately|overnight)\b",
        ],
        "description": "Contains potentially false advertising claims",
    },
    # --- Missing Disclaimers ---
    {
        "rule_id": "DISC_001",
        "category": "Missing Disclaimers",
        "severity": "medium",
        "patterns": [
            r"\bresults?\s+may\s+vary\b",
            r"\bindividual\s+results?\b",
            r"\bnot\s+a\s+substitute\s+for\s+(professional|medical)\b",
            r"\bconsult\s+(your\s+)?(doctor|physician|professional)\b",
        ],
        "description": "Content appears to lack required disclaimers",
        "inverted": True,  # flag if these patterns are ABSENT
    },
    # --- Privacy Concerns ---
    {
        "rule_id": "PRIV_001",
        "category": "Privacy Concerns",
        "severity": "high",
        "patterns": [
            r"\bwe\s+(sell|share|trade)\s+(your\s+)?(personal\s+)?data\b",
            r"\btrack(ing)?\s+your\s+(location|activity|behavior)\b",
            r"\bcollect(ing)?\s+personal\s+information\s+without\b",
        ],
        "description": "Contains potential privacy policy violations",
    },
    # --- Urgency & Pressure Tactics ---
    {
        "rule_id": "PRES_001",
        "category": "High Pressure Sales Tactics",
        "severity": "low",
        "patterns": [
            r"\blimited\s+time\s+offer\b",
            r"\bact\s+now\b",
            r"\bonly\s+\d+\s+(left|remaining|available)\b",
            r"\boffer\s+expires?\b",
            r"\btoday\s+only\b",
            r"\blast\s+chance\b",
        ],
        "description": "Uses high-pressure sales tactics",
    },
]


# ---------------------------------------------------------------------------
# Rule Engine
# ---------------------------------------------------------------------------

def check_compliance_rules(text: str) -> List[RuleViolation]:
    """
    Runs all regex rules against the provided text.
    Returns list of violations found.
    """
    violations = []
    text_lower = text.lower()

    for rule in COMPLIANCE_RULES:
        is_inverted = rule.get("inverted", False)

        if is_inverted:
            # For disclaimer rules: flag if NONE of the patterns are found
            found_any = any(
                re.search(pattern, text_lower, re.IGNORECASE)
                for pattern in rule["patterns"]
            )
            if not found_any and len(text.split()) > 50:
                violations.append(RuleViolation(
                    rule_id=rule["rule_id"],
                    category=rule["category"],
                    severity=rule["severity"],
                    matched_text="[No disclaimer found]",
                    description=rule["description"],
                ))
        else:
            for pattern in rule["patterns"]:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    violations.append(RuleViolation(
                        rule_id=rule["rule_id"],
                        category=rule["category"],
                        severity=rule["severity"],
                        matched_text=match.group(0),
                        description=rule["description"],
                    ))
                    break  # one violation per rule is enough

    return violations


def format_violations_for_llm(violations: List[RuleViolation]) -> str:
    """Formats rule violations into a string for the LLM prompt."""
    if not violations:
        return "No rule-based violations detected."

    lines = []
    for v in violations:
        lines.append(
            f"[{v.severity.upper()}] {v.rule_id} - {v.category}: "
            f"{v.description} (matched: '{v.matched_text}')"
        )
    return "\n".join(lines)