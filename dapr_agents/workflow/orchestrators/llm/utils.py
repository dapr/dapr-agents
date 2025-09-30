from typing import List, Dict, Any, Optional


def update_step_statuses(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensures step and sub-step statuses follow logical progression:
      • Parent completes if all substeps complete.
      • Parent goes in_progress if any substep is in_progress.
      • If substeps start completing, parent moves in_progress.
      • If parent was completed but a substep reverts to in_progress, parent downgrades.
      • Standalone steps (no substeps) are only updated via explicit status_updates.
    """
    for step in plan:
        subs = step.get("substeps", None)

        # --- NO substeps: do nothing here (explicit updates only) ---
        if subs is None:
            continue

        # If substeps is not a list or is an empty list, treat as no‐substeps too:
        if not isinstance(subs, list) or len(subs) == 0:
            continue

        # Collect child statuses
        statuses = {ss["status"] for ss in subs}

        # 1. All done → parent done
        if statuses == {"completed"}:
            step["status"] = "completed"

        # 2. Any in_progress → parent in_progress
        elif "in_progress" in statuses:
            step["status"] = "in_progress"

        # 3. Some done, parent not yet started → bump to in_progress
        elif "completed" in statuses and step["status"] == "not_started":
            step["status"] = "in_progress"

        # 4. If parent was completed but a child is in_progress, downgrade
        elif step["status"] == "completed" and any(s != "completed" for s in statuses):
            step["status"] = "in_progress"

    return plan


def validate_plan_structure(plan: List[Dict[str, Any]]) -> bool:
    """
    Validates if the plan structure follows the correct schema.

    Args:
        plan (List[Dict[str, Any]]): The execution plan.

    Returns:
        bool: True if the plan structure is valid, False otherwise.
    """
    required_keys = {"step", "description", "status"}
    for step in plan:
        if not required_keys.issubset(step.keys()):
            return False
        if "substeps" in step:
            for substep in step["substeps"]:
                if not {"substep", "description", "status"}.issubset(substep.keys()):
                    return False
    return True


def find_step_in_plan(
    plan: List[Dict[str, Any]], step: int, substep: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """
    Finds a specific step or substep in a plan.

    Args:
        plan (List[Dict[str, Any]]): The execution plan.
        step (int): The step number to find.
        substep (Optional[float]): The substep number (if applicable).

    Returns:
        Dict[str, Any] | None: The found step/substep dictionary or None if not found.
    """
    for step_entry in plan:
        if step_entry["step"] == step:
            if substep is None:
                return step_entry

            for sub in step_entry.get("substeps", []):
                if sub["substep"] == substep:
                    return sub
    return None


def restructure_plan(
    plan: List[Dict[str, Any]], updates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Applies restructuring updates to the task execution plan.

    Args:
        plan (List[Dict[str, Any]]): The current execution plan.
        updates (List[Dict[str, Any]]): A list of updates to apply.

    Returns:
        List[Dict[str, Any]]: The updated execution plan.
    """
    for update in updates:
        step_id = update["step"]
        step_entry = find_step_in_plan(plan, step_id)
        if step_entry:
            # Preserve existing substeps when updating step
            existing_substeps = step_entry.get("substeps")
            step_entry.update(update)
            # Restore substeps if they were present and not explicitly updated
            if existing_substeps is not None and "substeps" not in update:
                step_entry["substeps"] = existing_substeps

    return plan
