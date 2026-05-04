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
HITL via direct workflow event: resume a paused workflow by raising an event
through the Dapr Workflow client, with no pub/sub or HTTP endpoint required.

How it works
------------
When a DurableAgent workflow pauses for approval, it is waiting for an external
event named approval_response_{approval_request_id}. This script sends that
event directly to the Dapr sidecar, which forwards it to the waiting workflow.

Use this when:
- You are operating entirely from the command line.
- The agent is not configured with a pub/sub approval topic.
- You know the instance_id and approval_request_id (printed by the agent).

Usage
-----
Run the agent first (e.g. durable_agent_hitl.py or hitl_pubsub.py), then:

    python hitl_wf_event.py <instance_id> <approval_request_id> [approve|deny]

Example:

    python hitl_wf_event.py abc123 550e8400-e29b-41d4-a716-446655440000 approve
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
        reason=f"Sent via hitl_wf_event.py (decision={decision})",
    )

    print(
        f"Sending decision: instance={instance_id}, "
        f"request={approval_request_id}, approved={approved}"
    )

    client = DaprWorkflowClient()
    client.raise_workflow_event(
        instance_id=instance_id,
        event_name=event_name,
        data=response.model_dump(mode="json"),
    )

    print("Event sent. The workflow will resume shortly.")


if __name__ == "__main__":
    main()
