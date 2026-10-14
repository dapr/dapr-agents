#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Send a human approval decision to a paused Claude agent workflow.

Run app.py with APPROVAL_MODE=manual. When Claude asks to call
transfer_money, app.py prints the workflow instance id and the approval
request id. Pass both here:

    python approval_sender.py <instance_id> <approval_request_id> [approve|deny]

The script needs a reachable Dapr sidecar (the one app.py runs with).
"""

import sys

from dapr.ext.workflow import DaprWorkflowClient

from dapr_agents.agents.schemas import ApprovalResponseEvent

DECISIONS = ("approve", "deny")


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    instance_id, approval_request_id = sys.argv[1], sys.argv[2]
    decision = sys.argv[3].lower() if len(sys.argv) > 3 else "approve"
    if decision not in DECISIONS:
        sys.exit(f"decision must be one of {', '.join(DECISIONS)}")

    response = ApprovalResponseEvent(
        approval_request_id=approval_request_id,
        approved=decision == "approve",
        reason=f"sent via approval_sender.py ({decision})",
    )
    DaprWorkflowClient().raise_workflow_event(
        instance_id=instance_id,
        event_name=f"approval_response_{approval_request_id}",
        data=response.model_dump(mode="json"),
    )
    print(f"Sent '{decision}' for request {approval_request_id} to {instance_id}.")


if __name__ == "__main__":
    main()
