# Shipment exception-routing policy

Use this read-only policy for a delayed shipment that may miss its promised delivery window.

- **Carrier escalation:** Choose this route when the carrier has not provided a current scan, the package is stalled in transit, or the carrier must confirm a new delivery estimate.
- **Warehouse review:** Choose this route when the shipment reached a facility but has a sorting, inventory, address, or handoff discrepancy that the warehouse must investigate.
- **Customer notification:** Choose this route when the next action is to tell the customer about a confirmed delay, revised delivery window, or available delivery choice.
- **Manual review:** Choose this route when the evidence is incomplete, routes conflict, the shipment is high value, or a supervisor must decide the exception.

For a delay, identify the missing evidence first and choose the narrowest route supported by the facts. A customer update may accompany another route, but it does not replace carrier escalation, warehouse review, or manual review when those actions are required. Record the decision only as a proposed rerouting (**réacheminement**); this policy never changes shipment data.
