"""Schema loading tests for `src.llm_client`."""

import pytest

from src.llm_client import get_agent_schema, load_schema


class TestSchemaLoading:
    """Tests for schema loading functions."""

    def test_load_schema_with_extension(self):
        """Loading schema with .json extension succeeds."""
        schema = load_schema("planner_output_schema.json")
        assert schema is not None
        assert "properties" in schema

    def test_load_schema_without_extension(self):
        """Loading schema without .json extension."""
        schema = load_schema("planner_output_schema")
        assert schema is not None
        assert "properties" in schema

    def test_load_schema_caching(self):
        """Schemas are cached so repeated loads stay consistent."""
        schema1 = load_schema("supervisor_output_schema.json")
        schema2 = load_schema("supervisor_output_schema.json")
        assert schema1 == schema2

    def test_load_nonexistent_schema(self):
        """Loading nonexistent schema raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_schema("nonexistent_schema.json")

    def test_get_agent_schema_planner(self):
        """Planner agent schema exists and contains properties."""
        schema = get_agent_schema("planner")
        assert schema is not None
        assert "properties" in schema

    def test_get_agent_schema_supervisor(self):
        """Supervisor schema includes verdict property."""
        schema = get_agent_schema("supervisor")
        assert schema is not None
        assert "verdict" in schema.get("properties", {})

    def test_get_agent_schema_unknown(self):
        """Unknown agent raises ValueError."""
        with pytest.raises(ValueError, match="Unknown agent"):
            get_agent_schema("unknown_agent")

    def test_get_agent_schema_auto_discovery(self):
        """Auto-discovery works for standard agent schemas."""
        auto_discovered_agents = [
            "planner",
            "plan_reviewer",
            "simulation_designer",
            "design_reviewer",
            "code_generator",
            "code_reviewer",
            "execution_validator",
            "physics_sanity",
            "results_analyzer",
            "comparison_validator",
            "supervisor",
            "prompt_adaptor",
        ]

        for agent_name in auto_discovered_agents:
            schema = get_agent_schema(agent_name)
            assert schema is not None, f"Failed to auto-discover schema for {agent_name}"
            assert isinstance(schema, dict), f"Schema for {agent_name} should be dict"

    def test_get_agent_schema_special_case_report(self):
        """Special-case mapping for report agent."""
        schema = get_agent_schema("report")
        assert schema is not None
        assert isinstance(schema, dict)

    def test_get_agent_schema_error_includes_path(self):
        """Error path mentions expected schema filename."""
        with pytest.raises(ValueError) as exc_info:
            get_agent_schema("nonexistent_agent")

        assert "nonexistent_agent_output_schema" in str(exc_info.value)


