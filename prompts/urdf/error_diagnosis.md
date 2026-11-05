Diagnose the failed operation using only the supplied evidence. Choose one of the allowed actions. Use retry only for a plausibly transient state; use refine only when an available repairer can address the evidence within its remaining budget. Otherwise choose the supplied fallback action. You cannot change acceptance thresholds, declare damaged evidence valid, edit program code, or infer an environment failure without verified evidence. Scene recovery cannot modify object assets.

Return JSON only: {"action": "allowed action", "reason": "brief evidence-based reason", "object_ids": []}.
