---
name: shipment-exceptions
description: Route delayed shipment exceptions using an operational policy.
---

# Shipment exception routing

Use this skill to evaluate delayed shipments that may miss their delivery window.
Consult `references/exception-routing-policy.md` before making a recommendation.
Report the narrowest supported route, the evidence, and any missing facts.

The policy is advisory. Do not call external logistics systems, execute code, or mutate shipment data.
