"""Legacy aggregator that re-exports supervision, routing, and stage-selection tests."""

from tests.integration.routing.test_routing_decisions import *  # noqa: F401,F403
from tests.integration.stage_selection.test_select_stage_node import *  # noqa: F401,F403
from tests.integration.supervision.test_supervisor_triggers import *  # noqa: F401,F403
 
