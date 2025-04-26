from typing import List, Dict, Any, Optional


def update_step_statuses(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensures step and sub-step statuses follow logical progression:
    - A step is marked "completed" if all substeps are "completed".
    - If any sub-step is "in_progress", the parent step must also be "in_progress".
    - If a sub-step is "completed" but the parent step is "not_started", update it to "in_progress".
    - If a parent step is "completed" but a substep is still "in_progress", downgrade it to "in_progress".
    - Steps without substeps should still progress logically.

    Args:
        plan (List[Dict[str, Any]]): The current execution plan.

    Returns:
        List[Dict[str, Any]]: The updated execution plan with correct statuses.
    """
    for step in plan:
        # Case 0: Handle steps that have NO substeps
        if "substeps" not in step or not step["substeps"]:
            if step["status"] == "not_started":
                step[
                    "status"
                ] = "in_progress"  # Independent steps should start when execution begins
            continue  # Skip further processing if no substeps exist

        substep_statuses = {ss["status"] for ss in step["substeps"]}

        # Case 1: If ALL substeps are "completed", parent step must be "completed".
        if all(status == "completed" for status in substep_statuses):
            step["status"] = "completed"

        # Case 2: If ANY substep is "in_progress", parent step must also be "in_progress".
        elif "in_progress" in substep_statuses:
            step["status"] = "in_progress"

        # Case 3: If a sub-step was completed but the step is still "not_started", update it.
        elif "completed" in substep_statuses and step["status"] == "not_started":
            step["status"] = "in_progress"

        # Case 4: If the step is already marked as "completed" but a substep is still "in_progress", downgrade it.
        elif step["status"] == "completed" and "in_progress" in substep_statuses:
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
            step_entry.update(update)

    return plan
