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
Approval sender script for the HITL demo.

After running durable_agent_hitl.py, a workflow is waiting for an approval
event. This script sends that event to resume the workflow.

Usage
-----
Run with the instance_id printed by durable_agent_hitl.py and the
approval_request_id that appears in the agent logs when the workflow pauses:

    python approval_sender.py <instance_id> <approval_request_id> [approve|deny]

Example:

    python approval_sender.py abc123 550e8400-e29b-41d4-a716-446655440000 approve
"""

import sys
import logging

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    instance_id = sys.argv[1]
    approval_request_id = sys.argv[2]
    decision = sys.argv[3].lower() if len(sys.argv) > 3 else "approve"
    approved = decision == "approve"

    from dapr.ext.workflow import DaprWorkflowClient
    from dapr_agents.agents.schemas import ApprovalResponseEvent

    event_name = f"approval_response_{approval_request_id}"
    response = ApprovalResponseEvent(
        approval_request_id=approval_request_id,
        approved=approved,
        reason=f"sent via approval_sender.py (decision={decision})",
    )

    print(
        f"Sending approval decision: instance={instance_id}, "
        f"request={approval_request_id}, approved={approved}"
    )

    client = DaprWorkflowClient()
    client.raise_workflow_event(
        instance_id=instance_id,
        event_name=event_name,
        data=response.model_dump(mode="json"),
    )

    print("Approval event sent. The workflow will resume shortly.")


if __name__ == "__main__":
    main()
