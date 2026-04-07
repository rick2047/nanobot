from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.subagent import SubagentManager, SubagentRuntimeDefaults
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config, ExecToolConfig, SubagentConfig, WebToolsConfig
from nanobot.providers.base import GenerationSettings
from nanobot.providers.provider import ProviderFactory, ProviderRequest


def _make_provider(model: str = "anthropic/claude-sonnet-4-5") -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = model
    provider.generation = GenerationSettings(
        temperature=0.2,
        max_tokens=321,
        reasoning_effort="medium",
    )
    return provider


class _FactoryStub:
    def __init__(self, request: ProviderRequest, created_provider: object | None = None):
        self.request = request
        self.created_provider = created_provider or object()
        self.resolve_calls: list[dict[str, object]] = []
        self.create_calls: list[ProviderRequest] = []

    def resolve_request(self, **kwargs) -> ProviderRequest:
        self.resolve_calls.append(kwargs)
        return self.request

    def create(self, request: ProviderRequest | None = None):
        assert request is not None
        self.create_calls.append(request)
        return self.created_provider


def _make_manager(
    tmp_path: Path,
    *,
    provider_factory: _FactoryStub | ProviderFactory | None = None,
    profiles: dict[str, SubagentConfig] | None = None,
    exec_config: ExecToolConfig | None = None,
    web_config: WebToolsConfig | None = None,
) -> SubagentManager:
    provider = _make_provider()
    return SubagentManager(
        provider=provider,
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=4096,
        model=provider.get_default_model(),
        exec_config=exec_config,
        web_config=web_config,
        provider_factory=provider_factory,
        main_defaults=SubagentRuntimeDefaults(
            provider_name="anthropic",
            model=provider.get_default_model(),
            generation=provider.generation,
            max_iterations=9,
        ),
        subagent_profiles=profiles,
    )


def test_subagent_config_parses_camel_case_and_validates() -> None:
    config = Config.model_validate(
        {
            "providers": {
                "anthropic": {"apiKey": "ant-key"},
                "githubCopilot": {},
            },
            "agents": {
                "defaults": {
                    "model": "anthropic/claude-sonnet-4-5",
                    "temperature": 0.3,
                    "maxTokens": 800,
                    "reasoningEffort": "low",
                },
                "subagents": {
                    "reviewer": {
                        "description": "Review code changes",
                        "provider": "github-copilot",
                        "model": "github-copilot/gpt-4.1",
                        "maxIterations": 4,
                        "tools": ["read_file", "web-search"],
                        "skills": ["lint", "tests"],
                    }
                },
            },
        }
    )

    reviewer = config.agents.subagents["reviewer"]
    assert reviewer.provider == "github_copilot"
    assert reviewer.max_iterations == 4
    assert reviewer.tools == ["read_file", "web_search"]
    assert reviewer.skills == ["lint", "tests"]

    with pytest.raises(ValueError, match="requires model"):
        Config.model_validate(
            {
                "agents": {
                    "subagents": {
                        "bad": {
                            "description": "Bad profile",
                            "provider": "anthropic",
                        }
                    }
                }
            }
        )

    with pytest.raises(ValueError, match="Unknown subagent tools"):
        Config.model_validate(
            {
                "agents": {
                    "subagents": {
                        "bad": {
                            "description": "Bad tool",
                            "tools": ["message"],
                        }
                    }
                }
            }
        )

    with pytest.raises(ValueError, match="greater than or equal to 1"):
        Config.model_validate(
            {
                "agents": {
                    "subagents": {
                        "bad": {
                            "description": "Bad iterations",
                            "maxIterations": 0,
                        }
                    }
                }
            }
        )


def test_spawn_tool_exposes_profile_parameter_and_lists_profiles(tmp_path: Path) -> None:
    manager = _make_manager(
        tmp_path,
        profiles={
            "reviewer": SubagentConfig(
                description="Review code changes",
                tools=["read_file"],
            )
        },
    )

    tool = SpawnTool(manager)

    assert "profile" in tool.parameters["properties"]
    assert "Configured profiles:" in tool.description
    assert "- reviewer: Review code changes" in tool.description


@pytest.mark.asyncio
async def test_spawn_unknown_profile_returns_clear_error(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)

    result = await manager.spawn("inspect repo", profile="missing")

    assert result.startswith("Error: Unknown subagent profile 'missing'")


def test_profile_tool_allowlist_respects_global_disables(tmp_path: Path) -> None:
    factory = _FactoryStub(
        ProviderRequest(
            provider_name="anthropic",
            model="anthropic/claude-sonnet-4-5",
            generation=GenerationSettings(temperature=0.2, max_tokens=321, reasoning_effort="medium"),
        )
    )
    manager = _make_manager(
        tmp_path,
        provider_factory=factory,
        profiles={
            "worker": SubagentConfig(
                description="Limited worker",
                tools=["read_file", "exec", "web_search", "web_fetch"],
            )
        },
        exec_config=ExecToolConfig(enable=False),
        web_config=WebToolsConfig(enable=False),
    )

    resolved = manager._resolve_profile("worker")
    tools = manager._build_subagent_tools(resolved)

    assert set(tools.tool_names) == {"read_file"}


@pytest.mark.asyncio
async def test_profile_skill_allowlist_preloads_and_blocks_other_skills(tmp_path: Path) -> None:
    (tmp_path / "skills" / "allowed").mkdir(parents=True)
    (tmp_path / "skills" / "blocked").mkdir(parents=True)
    (tmp_path / "skills" / "allowed" / "SKILL.md").write_text(
        "---\ndescription: Allowed skill\n---\nallowed instructions\n",
        encoding="utf-8",
    )
    (tmp_path / "skills" / "blocked" / "SKILL.md").write_text(
        "---\ndescription: Blocked skill\n---\nblocked instructions\n",
        encoding="utf-8",
    )

    factory = _FactoryStub(
        ProviderRequest(
            provider_name="anthropic",
            model="anthropic/claude-sonnet-4-5",
            generation=GenerationSettings(temperature=0.2, max_tokens=321, reasoning_effort="medium"),
        )
    )
    manager = _make_manager(
        tmp_path,
        provider_factory=factory,
        profiles={
            "writer": SubagentConfig(
                description="Write with one skill",
                skills=["allowed"],
            )
        },
    )

    resolved = manager._resolve_profile("writer")
    tools = manager._build_subagent_tools(resolved)
    prompt = manager._build_subagent_prompt(resolved, tools)

    assert "### Skill: allowed" in prompt
    assert "blocked instructions" not in prompt
    assert "<name>blocked</name>" not in prompt

    blocked_result = await tools.execute("read_file", {"path": "skills/blocked/SKILL.md"})
    allowed_result = await tools.execute("read_file", {"path": "skills/allowed/SKILL.md"})
    list_result = await tools.execute("list_dir", {"path": "skills"})

    assert blocked_result.startswith("Error:")
    assert "allowed instructions" in allowed_result
    assert "allowed" in list_result
    assert "blocked" not in list_result


def test_provider_factory_inherits_defaults_and_normalizes_provider_names() -> None:
    config = Config.model_validate(
        {
            "providers": {
                "anthropic": {"apiKey": "ant-key"},
                "githubCopilot": {},
            },
            "agents": {
                "defaults": {
                    "model": "anthropic/claude-sonnet-4-5",
                    "temperature": 0.3,
                    "maxTokens": 800,
                    "reasoningEffort": "low",
                }
            },
        }
    )
    factory = ProviderFactory(config)

    inherited = factory.resolve_request(model="anthropic/claude-haiku-4-5")
    shifted = factory.resolve_request(
        provider_name="github-copilot",
        model="github-copilot/gpt-4.1",
        temperature=0.6,
    )

    assert inherited.provider_name == "anthropic"
    assert inherited.generation.temperature == 0.3
    assert inherited.generation.max_tokens == 800
    assert inherited.generation.reasoning_effort == "low"
    assert shifted.provider_name == "github_copilot"
    assert shifted.generation.temperature == 0.6
    assert shifted.generation.max_tokens == 800


@pytest.mark.asyncio
async def test_profiled_subagent_uses_fresh_provider_and_reports_iteration_limit_as_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_provider = MagicMock()
    request = ProviderRequest(
        provider_name="github_copilot",
        model="github-copilot/gpt-4.1",
        generation=GenerationSettings(
            temperature=0.6,
            max_tokens=777,
            reasoning_effort="high",
        ),
    )
    factory = _FactoryStub(request, created_provider=created_provider)
    manager = _make_manager(
        tmp_path,
        provider_factory=factory,
        profiles={
            "reviewer": SubagentConfig(
                description="Review with Copilot",
                provider="github-copilot",
                model="github-copilot/gpt-4.1",
                temperature=0.6,
                max_tokens=777,
                reasoning_effort="high",
                max_iterations=6,
            )
        },
    )
    manager._announce_result = AsyncMock()

    captured: dict[str, object] = {}

    async def fake_run(self, spec):
        captured["provider"] = self.provider
        captured["model"] = spec.model
        captured["temperature"] = spec.temperature
        captured["max_tokens"] = spec.max_tokens
        captured["reasoning_effort"] = spec.reasoning_effort
        captured["max_iterations"] = spec.max_iterations
        return SimpleNamespace(
            stop_reason="max_iterations",
            final_content="subagent hit limit",
            error=None,
            tool_events=[],
        )

    monkeypatch.setattr("nanobot.agent.subagent.AgentRunner.run", fake_run)

    await manager._run_subagent(
        "sub-1",
        "review code",
        "review",
        {"channel": "cli", "chat_id": "direct"},
        manager._resolve_profile("reviewer"),
    )

    assert captured["provider"] is created_provider
    assert captured["model"] == "github-copilot/gpt-4.1"
    assert captured["temperature"] == 0.6
    assert captured["max_tokens"] == 777
    assert captured["reasoning_effort"] == "high"
    assert captured["max_iterations"] == 6
    args = manager._announce_result.await_args.args
    assert args[3] == "subagent hit limit"
    assert args[5] == "error"
