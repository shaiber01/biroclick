"""Legacy aggregator that re-exports planning tests from the new subpackage."""

from tests.integration.planning.test_plan_node_contracts import *  # noqa: F401,F403
from tests.integration.planning.test_plan_reviewer_rules import *  # noqa: F401,F403
from tests.integration.planning.test_planning_edge_cases import *  # noqa: F401,F403
from tests.integration.planning.test_prompt_adaptation import *  # noqa: F401,F403

