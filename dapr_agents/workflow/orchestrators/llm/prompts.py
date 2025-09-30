TASK_PLANNING_PROMPT = """## Task Overview
We need to develop a structured, **logically ordered** plan to accomplish the following task:

{task}

### Team of Agents
{agents}

### Planning Instructions:
1. **Break the task into a structured, step-by-step plan** ensuring logical sequencing.
2. **Each step should be described in a neutral, objective manner**, focusing on the action required rather than specifying who performs it.
3. **The plan should be designed to be executed by the provided team**, but **DO NOT reference specific agents** in step descriptions.
4. **Do not introduce additional roles, skills, or external resources** outside of the given team.
5. **Use clear and precise descriptions** for each step, focusing on **what** needs to be done rather than **who** does it.
6. **Avoid unnecessary steps or excessive granularity**. Keep the breakdown clear and efficient.
7. **If a task involves both code generation and execution, structure the plan as follows:**
   - **Code Generation Step:** Describe what code needs to be generated, specifying language and functionality.
   - **Code Execution Step:** If execution is required, describe how it should be performed separately.
   - **Feedback Handling Step:** If execution results need analysis, include a step for refining the code.
8. **Maintain a natural task flow** where each step logically follows the previous one.
9. **Each step must have a `status` field** to track progress (Initially all set to `not_started`).
10. **Include sub-steps for complex steps that can be broken down into 2-4 distinct actions. This provides better granularity and tracking.**
11. **Examples of steps that should have substeps:**
    - Planning/analysis steps (e.g., "Assess situation" → "Gather intel", "Analyze data", "Identify risks")
    - Preparation steps (e.g., "Prepare supplies" → "Inventory needs", "Gather materials", "Pack equipment")
    - Execution steps with multiple phases (e.g., "Execute plan" → "Begin execution", "Monitor progress", "Adjust as needed")
12. **Focus only on structuring the task execution**, NOT on assigning agents at this stage.

### Expected Output Format (JSON Schema):
{plan_schema}

**IMPORTANT: Return ONLY the raw JSON object without any markdown code blocks, explanations, or additional text. The JSON must have an "objects" field containing the array of plan steps.**
"""

TASK_INITIAL_PROMPT = """## Mission Briefing

We have received the following task:

{task}

### Team of Agents
{agents}

### Execution Plan
Here is the structured approach the team will follow to accomplish the task:

{plan}
"""

NEXT_STEP_PROMPT = """## Task Context

The team is working on the following task:

{task}

## CRITICAL INSTRUCTION
**SUBSTEPS MUST USE DECIMAL NOTATION**: If you select a substep, it MUST be a decimal number like 1.2, 1.3, 2.1, 2.2, etc. NEVER use whole numbers like 1, 2, 3, 4 or 2.0, 3.0, 4.0 for substeps!

### Team of Agents (ONLY these agents are available):
{agents}

### Current Execution Plan:
{plan}

### Next Steps:
- **Select the next best-suited agent** from the team of agents list who should respond **based on the execution plan above**.
- **DO NOT select an agent that is not explicitly listed in `{agents}`**.
- **You must always provide a valid agent name** from the team**. DO NOT return `null` or an empty agent name**.
- Provide a **clear, actionable instruction** for the next agent.
- **You must ONLY select step and substep IDs that EXIST in the plan.**
  - **DO NOT select a `"completed"` step or substep.**
  - **If the main step is `"not_started"` but has `"completed"` substeps, you must correctly identify the next `"not_started"` substep.**
  - **DO NOT create or assume non-existent step/substep IDs.**

### Step Progression Rules:
- **SEQUENTIAL EXECUTION**: Execute steps in numerical order (1, 2, 3, etc.) and substeps in decimal order (1.1, 1.2, 1.3, etc.)
- **COMPLETE ALL SUBSTEPS**: Before moving to the next main step, ALL substeps of the current step must be completed
- **FIND NEXT AVAILABLE**: Look for the first "not_started" or "in_progress" step/substep in sequence
- **NO SKIPPING**: Only skip steps if they are explicitly marked as "completed"

### Examples:
- **Step 1 has substeps [1.1, 1.2, 1.3] and 1.1, 1.2 are completed:**
  - **CORRECT**: Next substep is 1.3
  - **WRONG**: Next substep is 2.1, 3.1, or 2.0
- **Step 1 has substeps [1.1, 1.2, 1.3] and ALL are completed:**
  - **CORRECT**: Next step is 2.1 (first substep of step 2)
  - **WRONG**: Next step is 3.1 or any other non-sequential step

### Expected Output Format (JSON Schema):
{next_step_schema}
"""

PROGRESS_CHECK_PROMPT = """## Progress Check

### Task Context
The team is working on the following task:

{task}

### Current Execution Plan:

{plan}

### Latest Execution Context:
- **Step ID:** {step}
- **Substep ID (if applicable):** {substep}
- **Step Execution Results:** "{results}"

### Task Evaluation:
Assess the task progress based on **conversation history**, execution results, and the structured plan.

**Key Point:** `plan_needs_update` should be `true` whenever you are making ANY changes to the plan (status updates OR restructuring).

1. **Determine Overall Task Verdict**
   - `"continue"` → **Use this if there are `"not_started"` or `"in_progress"` steps that still require execution.**
   - `"completed"` → The task is **done** (i.e., **all required steps and substeps have been completed**).
   - `"failed"` → The task cannot be completed due to an unresolved issue.

2. **Evaluate Step Completion**
   - If an **agent explicitly marks a step as `"completed"`**, then it **remains completed**, regardless of substeps.
   - If a **substep is completed**, check if **all** substeps are `"completed"` **before marking the parent step as "completed"**.
   - **If a step is "completed" but has "not_started" substeps, DO NOT modify those substeps.** They remain unchanged unless explicitly acted upon.

3. **Update Step & Sub-Step Status**
   - **Always update statuses based on the latest results**, regardless of whether the verdict is `"continue"` or `"completed"`.
   - If an **agent explicitly marks a step as `"completed"`**, then it **remains completed**, regardless of substeps.
   - If a **substep is completed**, check if **all** substeps are `"completed"` **before marking the parent step as "completed"**.
   - **If a step is "completed" but has "not_started" substeps, DO NOT modify those substeps.** They remain unchanged unless explicitly acted upon.
   - **IMPORTANT: If you are making ANY status updates (step or substep), set `plan_needs_update` to `true`.**

4. **Plan Adjustments (Only If Necessary)**
   - If the step descriptions are **unclear or incomplete**, update `"plan_restructure"` with a **single modified step**.
   - Do **not** introduce unnecessary modifications.
   - **IMPORTANT: If you are making ANY plan restructuring, set `plan_needs_update` to `true`.**

### Important:
- **Do NOT mark a step as `"completed"` unless explicitly confirmed based on execution results.**
- **Do NOT mark substeps as `"completed"` unless explicitly confirmed or all are already completed.**
- **Always apply step/substep status updates, even if the task is `"completed"`**.
- **Do not introduce unnecessary modifications to the plan.**

### Expected Output Format (JSON Schema):
{progress_check_schema}
"""

SUMMARY_GENERATION_PROMPT = """# Summary Generator

## Initial Task:
{task}

## Execution Overview:
- **Final Verdict:** {verdict}
  _(Possible values: `"continue"`, `"completed"`, `"failed"` `max_iterations_reached`)_
- **Execution Plan Status:**
  {plan}
- **Last Action Taken:**
  - **Step:** `{step}` (Sub-step `{substep}` if applicable)
  - **Executing Agent:** `{agent}`
  - **Execution Result:** {result}

## Instructions to Generate Best Summary
Based on the **conversation history** and **execution plan**, generate a **clear and structured** summary:

1. **If the task is `"completed"`**, provide a concise but complete final summary.
   - **Briefly describe the key steps taken** and the final outcome.
   - Ensure clarity, avoiding excessive details while **focusing on essential takeaways**.
   - Phrase it **as if reporting back to the user** rather than as a system log.

2. **If the task is `"failed"`**, explain why.
   - Summarize blockers, unresolved challenges, or missing steps.
   - If applicable, suggest potential next actions to resolve the issue.

3. **If the task is `"continue"`**, summarize progress so far.
   - **Highlight completed steps** and the current state of execution.
   - **Mention what remains unfinished** and the next logical step.
   - Keep it informative but **concise and forward-looking**.

4. **If the task is `"max_iterations_reached"`**, summarize the progress and note the limitation.
   - Highlight what has been **achieved so far** and what **remains unfinished**.
   - Clearly state that the **workflow reached its iteration limit** before completion.
   - If possible, **suggest next steps** (e.g., refining the execution plan, restarting the task, or adjusting agent strategies).

## Expected Output
A structured summary that is:
- **Clear and to the point**
- **Context-aware (includes results and execution progress)**
- **User-friendly (reads naturally rather than like system logs)**
- **Relevant (avoids unnecessary details while maintaining accuracy)**
"""
