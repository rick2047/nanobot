"""Subagent manager for background task execution."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.runner import AgentRunSpec, AgentRunner
from nanobot.agent.skills import BUILTIN_SKILLS_DIR, SkillsLoader
from nanobot.agent.tools.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
    _is_under,
)
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.search import GlobTool, GrepTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import (
    AgentDefaults,
    ExecToolConfig,
    SUBAGENT_SAFE_TOOL_NAMES,
    SubagentConfig,
    WebToolsConfig,
)
from nanobot.providers.base import GenerationSettings, LLMProvider
from nanobot.providers.provider import ProviderFactory, ProviderRequest
from nanobot.utils.prompt_templates import render_template

_LEGACY_SUBAGENT_MAX_ITERATIONS = 15


class _SubagentHook(AgentHook):
    """Logging-only hook for subagent execution."""

    def __init__(self, task_id: str) -> None:
        self._task_id = task_id

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        for tool_call in context.tool_calls:
            args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
            logger.debug(
                "Subagent [{}] executing: {} with arguments: {}",
                self._task_id,
                tool_call.name,
                args_str,
            )


@dataclass(frozen=True)
class SubagentRuntimeDefaults:
    """Inherited defaults used when resolving configured subagent profiles."""

    provider_name: str | None
    model: str
    generation: GenerationSettings
    max_iterations: int


@dataclass(frozen=True)
class ResolvedSubagentProfile:
    """Resolved runtime profile for one subagent invocation."""

    provider_request: ProviderRequest
    max_iterations: int
    profile_name: str | None = None
    description: str | None = None
    tools: tuple[str, ...] | None = None
    skills: tuple[str, ...] | None = None
    use_fresh_provider: bool = False


class _SkillPathGuard:
    """Restrict skill-path access to an allowlisted subset."""

    def __init__(self, workspace: Path, allowed_skill_dirs: list[Path]):
        self._workspace = workspace.resolve()
        self._allowed_skill_dirs = [path.resolve() for path in allowed_skill_dirs]
        self._protected_roots = [
            root.resolve()
            for root in (workspace / "skills", BUILTIN_SKILLS_DIR)
            if root.exists()
        ]

    @property
    def extra_allowed_dirs(self) -> list[Path]:
        return [
            path for path in self._allowed_skill_dirs
            if not _is_under(path, self._workspace)
        ]

    def allows(self, path: Path) -> bool:
        resolved = path.resolve()
        if not any(_is_under(resolved, root) for root in self._protected_roots):
            return True
        return any(_is_under(resolved, allowed) for allowed in self._allowed_skill_dirs)

    def ensure_allowed(self, path: Path) -> Path:
        if not self.allows(path):
            raise PermissionError(
                f"Path {path} is outside the allowed subagent skill set"
            )
        return path


class _SkillGuardedMixin:
    def __init__(self, *args: Any, skill_guard: _SkillPathGuard, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._skill_guard = skill_guard

    def _resolve(self, path: str) -> Path:
        return self._skill_guard.ensure_allowed(super()._resolve(path))


class _SkillGuardedReadFileTool(_SkillGuardedMixin, ReadFileTool):
    pass


class _SkillGuardedWriteFileTool(_SkillGuardedMixin, WriteFileTool):
    pass


class _SkillGuardedEditFileTool(_SkillGuardedMixin, EditFileTool):
    pass


class _SkillGuardedGlobTool(_SkillGuardedMixin, GlobTool):
    def _iter_files(self, root: Path):
        for path in super()._iter_files(root):
            if self._skill_guard.allows(path):
                yield path

    def _iter_entries(self, root: Path, *, include_files: bool, include_dirs: bool):
        for path in super()._iter_entries(
            root,
            include_files=include_files,
            include_dirs=include_dirs,
        ):
            if self._skill_guard.allows(path):
                yield path


class _SkillGuardedGrepTool(_SkillGuardedGlobTool, GrepTool):
    pass


class _SkillGuardedListDirTool(_SkillGuardedMixin, ListDirTool):
    async def execute(
        self,
        path: str | None = None,
        recursive: bool = False,
        max_entries: int | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            if path is None:
                raise ValueError("Unknown path")
            dp = self._resolve(path)
            if not dp.exists():
                return f"Error: Directory not found: {path}"
            if not dp.is_dir():
                return f"Error: Not a directory: {path}"

            cap = max_entries or self._DEFAULT_MAX
            items: list[str] = []
            total = 0

            if recursive:
                for item in sorted(dp.rglob("*")):
                    if any(part in self._IGNORE_DIRS for part in item.parts):
                        continue
                    if not self._skill_guard.allows(item):
                        continue
                    total += 1
                    if len(items) < cap:
                        rel = item.relative_to(dp)
                        items.append(f"{rel}/" if item.is_dir() else str(rel))
            else:
                for item in sorted(dp.iterdir()):
                    if item.name in self._IGNORE_DIRS:
                        continue
                    if not self._skill_guard.allows(item):
                        continue
                    total += 1
                    if len(items) < cap:
                        pfx = "\U0001f4c1 " if item.is_dir() else "\U0001f4c4 "
                        items.append(f"{pfx}{item.name}")

            if not items and total == 0:
                return f"Directory {path} is empty"

            result = "\n".join(items)
            if total > cap:
                result += f"\n\n(truncated, showing first {cap} of {total} entries)"
            return result
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error listing directory: {e}"


class SubagentManager:
    """Manages background subagent execution."""

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        max_tool_result_chars: int,
        model: str | None = None,
        web_config: WebToolsConfig | None = None,
        exec_config: ExecToolConfig | None = None,
        restrict_to_workspace: bool = False,
        provider_factory: ProviderFactory | None = None,
        main_defaults: SubagentRuntimeDefaults | None = None,
        subagent_profiles: dict[str, SubagentConfig] | None = None,
    ):
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.web_config = web_config or WebToolsConfig()
        self.max_tool_result_chars = max_tool_result_chars
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self.runner = AgentRunner(provider)
        self._provider_factory = provider_factory
        defaults = main_defaults or SubagentRuntimeDefaults(
            provider_name=None,
            model=self.model,
            generation=provider.generation,
            max_iterations=AgentDefaults().max_tool_iterations,
        )
        self._main_defaults = defaults
        self._subagent_profiles = subagent_profiles or {}
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._session_tasks: dict[str, set[str]] = {}

    def get_profile_descriptions(self) -> list[tuple[str, str]]:
        """Return configured subagent profile ids and descriptions."""
        return [
            (profile_id, profile.description)
            for profile_id, profile in sorted(self._subagent_profiles.items())
        ]

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
        profile: str | None = None,
    ) -> str:
        """Spawn a subagent to execute a task in the background."""
        try:
            resolved_profile = self._resolve_profile(profile)
        except ValueError as e:
            return f"Error: {e}"

        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        origin = {"channel": origin_channel, "chat_id": origin_chat_id}

        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin, resolved_profile)
        )
        self._running_tasks[task_id] = bg_task
        if session_key:
            self._session_tasks.setdefault(session_key, set()).add(task_id)

        def _cleanup(_: asyncio.Task) -> None:
            self._running_tasks.pop(task_id, None)
            if session_key and (ids := self._session_tasks.get(session_key)):
                ids.discard(task_id)
                if not ids:
                    del self._session_tasks[session_key]

        bg_task.add_done_callback(_cleanup)

        logger.info(
            "Spawned subagent [{}]: {}{}",
            task_id,
            display_label,
            f" (profile: {resolved_profile.profile_name})" if resolved_profile.profile_name else "",
        )
        return (
            f"Subagent [{display_label}] started (id: {task_id}). "
            "I'll notify you when it completes."
        )

    def _resolve_profile(self, profile_name: str | None) -> ResolvedSubagentProfile:
        if not profile_name:
            return ResolvedSubagentProfile(
                provider_request=ProviderRequest(
                    provider_name=self._main_defaults.provider_name,
                    model=self.model,
                    generation=self.provider.generation,
                ),
                max_iterations=_LEGACY_SUBAGENT_MAX_ITERATIONS,
            )

        profile = self._subagent_profiles.get(profile_name)
        if profile is None:
            available = ", ".join(sorted(self._subagent_profiles)) or "(none configured)"
            raise ValueError(
                f"Unknown subagent profile '{profile_name}'. Available profiles: {available}"
            )
        if self._provider_factory is None:
            raise ValueError("Subagent profiles require a configured provider factory")

        skill_names = tuple(profile.skills or ())
        if skill_names:
            available_skill_paths = self._filter_skill_paths(
                SkillsLoader(self.workspace).get_skill_paths(),
                skill_names,
            )
            missing = [name for name in skill_names if name not in available_skill_paths]
            if missing:
                raise ValueError(
                    f"Subagent profile '{profile_name}' references unknown skills: "
                    f"{', '.join(missing)}"
                )

        provider_request = self._provider_factory.resolve_request(
            provider_name=profile.provider,
            model=profile.model or self._main_defaults.model,
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
            reasoning_effort=profile.reasoning_effort,
        )

        return ResolvedSubagentProfile(
            profile_name=profile_name,
            description=profile.description,
            provider_request=provider_request,
            max_iterations=profile.max_iterations or self._main_defaults.max_iterations,
            tools=tuple(profile.tools) if profile.tools else None,
            skills=skill_names or None,
            use_fresh_provider=True,
        )

    @staticmethod
    def _filter_skill_paths(
        skill_paths: dict[str, Path],
        allowed_skills: tuple[str, ...] | None,
    ) -> dict[str, Path]:
        if not allowed_skills:
            return skill_paths
        allowed = set(allowed_skills)
        return {name: path for name, path in skill_paths.items() if name in allowed}

    @staticmethod
    def _filter_skills_summary(skills_summary: str, allowed_skills: tuple[str, ...] | None) -> str:
        if not allowed_skills or not skills_summary:
            return skills_summary

        allowed = set(allowed_skills)
        blocks: list[str] = []
        current: list[str] = []
        inside = False

        for line in skills_summary.splitlines():
            stripped = line.strip()
            if stripped.startswith("<skill "):
                current = [line]
                inside = True
                continue
            if inside:
                current.append(line)
                if stripped == "</skill>":
                    block = "\n".join(current)
                    if any(f"<name>{name}</name>" in block for name in allowed):
                        blocks.append(block)
                    current = []
                    inside = False

        if not blocks:
            return ""
        return "<skills>\n" + "\n".join(blocks) + "\n</skills>"

    def _build_skill_guard(self, resolved_profile: ResolvedSubagentProfile) -> _SkillPathGuard | None:
        if not resolved_profile.skills:
            return None
        loader = SkillsLoader(self.workspace)
        allowed_skill_dirs = list(
            self._filter_skill_paths(loader.get_skill_paths(), resolved_profile.skills).values()
        )
        return _SkillPathGuard(self.workspace, allowed_skill_dirs)

    def _tool_enabled(self, tool_name: str) -> bool:
        if tool_name == "exec":
            return self.exec_config.enable
        if tool_name in {"web_search", "web_fetch"}:
            return self.web_config.enable
        return True

    def _effective_tool_names(self, resolved_profile: ResolvedSubagentProfile) -> list[str]:
        requested = (
            list(resolved_profile.tools)
            if resolved_profile.tools is not None
            else list(SUBAGENT_SAFE_TOOL_NAMES)
        )
        return [tool_name for tool_name in requested if self._tool_enabled(tool_name)]

    def _build_subagent_tools(self, resolved_profile: ResolvedSubagentProfile) -> ToolRegistry:
        tools = ToolRegistry()
        allowed_tools = set(self._effective_tool_names(resolved_profile))
        allowed_dir = self.workspace if (self.restrict_to_workspace or self.exec_config.sandbox) else None
        skill_guard = self._build_skill_guard(resolved_profile)
        extra_read = (
            skill_guard.extra_allowed_dirs
            if skill_guard is not None
            else ([BUILTIN_SKILLS_DIR] if allowed_dir else None)
        )

        read_cls = _SkillGuardedReadFileTool if skill_guard else ReadFileTool
        write_cls = _SkillGuardedWriteFileTool if skill_guard else WriteFileTool
        edit_cls = _SkillGuardedEditFileTool if skill_guard else EditFileTool
        list_cls = _SkillGuardedListDirTool if skill_guard else ListDirTool
        glob_cls = _SkillGuardedGlobTool if skill_guard else GlobTool
        grep_cls = _SkillGuardedGrepTool if skill_guard else GrepTool

        constructor_kwargs = {
            "workspace": self.workspace,
            "allowed_dir": allowed_dir,
        }

        if "read_file" in allowed_tools:
            read_kwargs = dict(constructor_kwargs)
            if extra_read:
                read_kwargs["extra_allowed_dirs"] = extra_read
            if skill_guard:
                read_kwargs["skill_guard"] = skill_guard
            tools.register(read_cls(**read_kwargs))

        for tool_name, tool_cls in (
            ("write_file", write_cls),
            ("edit_file", edit_cls),
            ("list_dir", list_cls),
            ("glob", glob_cls),
            ("grep", grep_cls),
        ):
            if tool_name not in allowed_tools:
                continue
            kwargs = dict(constructor_kwargs)
            if skill_guard:
                kwargs["skill_guard"] = skill_guard
            tools.register(tool_cls(**kwargs))

        if "exec" in allowed_tools:
            tools.register(ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
                sandbox=self.exec_config.sandbox,
                path_append=self.exec_config.path_append,
            ))
        if "web_search" in allowed_tools:
            tools.register(
                WebSearchTool(config=self.web_config.search, proxy=self.web_config.proxy)
            )
        if "web_fetch" in allowed_tools:
            tools.register(WebFetchTool(proxy=self.web_config.proxy))
        return tools

    def _build_available_tools_summary(self, tools: ToolRegistry) -> str:
        lines: list[str] = []
        for name in sorted(tools.tool_names):
            tool = tools.get(name)
            if tool is None:
                continue
            lines.append(f"- {name}: {tool.description}")
        return "\n".join(lines)

    def _build_subagent_prompt(
        self,
        resolved_profile: ResolvedSubagentProfile,
        tools: ToolRegistry,
    ) -> str:
        from nanobot.agent.context import ContextBuilder

        time_ctx = ContextBuilder._build_runtime_context(None, None)
        skills_loader = SkillsLoader(self.workspace)
        skills_summary = self._filter_skills_summary(
            skills_loader.build_skills_summary(),
            resolved_profile.skills,
        )
        preloaded_skills = (
            skills_loader.load_skills_for_context(list(resolved_profile.skills))
            if resolved_profile.skills
            else ""
        )
        return render_template(
            "agent/subagent_system.md",
            time_ctx=time_ctx,
            workspace=str(self.workspace),
            profile_name=resolved_profile.profile_name or "",
            profile_description=resolved_profile.description or "",
            preloaded_skills=preloaded_skills,
            skills_summary=skills_summary or "",
            available_tools=self._build_available_tools_summary(tools),
        )

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
        resolved_profile: ResolvedSubagentProfile | None = None,
    ) -> None:
        """Execute the subagent task and announce the result."""
        resolved_profile = resolved_profile or self._resolve_profile(None)
        logger.info(
            "Subagent [{}] starting task: {}{}",
            task_id,
            label,
            f" (profile: {resolved_profile.profile_name})" if resolved_profile.profile_name else "",
        )

        try:
            provider = (
                self._provider_factory.create(resolved_profile.provider_request)
                if resolved_profile.use_fresh_provider and self._provider_factory is not None
                else self.provider
            )
            runner = AgentRunner(provider) if provider is not self.provider else self.runner
            tools = self._build_subagent_tools(resolved_profile)
            system_prompt = self._build_subagent_prompt(resolved_profile, tools)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            result = await runner.run(AgentRunSpec(
                initial_messages=messages,
                tools=tools,
                model=resolved_profile.provider_request.model,
                max_iterations=resolved_profile.max_iterations,
                max_tool_result_chars=self.max_tool_result_chars,
                temperature=resolved_profile.provider_request.generation.temperature,
                max_tokens=resolved_profile.provider_request.generation.max_tokens,
                reasoning_effort=resolved_profile.provider_request.generation.reasoning_effort,
                hook=_SubagentHook(task_id),
                max_iterations_message="Task completed but no final response was generated.",
                error_message=None,
                fail_on_tool_error=True,
            ))
            if result.stop_reason == "tool_error":
                await self._announce_result(
                    task_id,
                    label,
                    task,
                    self._format_partial_progress(result),
                    origin,
                    "error",
                )
                return
            if result.stop_reason == "error":
                await self._announce_result(
                    task_id,
                    label,
                    task,
                    result.error or "Error: subagent execution failed.",
                    origin,
                    "error",
                )
                return
            if result.stop_reason == "max_iterations" and resolved_profile.profile_name:
                await self._announce_result(
                    task_id,
                    label,
                    task,
                    result.final_content or "Error: subagent hit its iteration limit.",
                    origin,
                    "error",
                )
                return

            final_result = (
                result.final_content or "Task completed but no final response was generated."
            )
            logger.info("Subagent [{}] completed successfully", task_id)
            await self._announce_result(task_id, label, task, final_result, origin, "ok")

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error("Subagent [{}] failed: {}", task_id, e)
            await self._announce_result(task_id, label, task, error_msg, origin, "error")

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
    ) -> None:
        """Announce the subagent result to the main agent via the message bus."""
        status_text = "completed successfully" if status == "ok" else "failed"

        announce_content = render_template(
            "agent/subagent_announce.md",
            label=label,
            status_text=status_text,
            task=task,
            result=result,
        )

        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
        )

        await self.bus.publish_inbound(msg)
        logger.debug(
            "Subagent [{}] announced result to {}:{}",
            task_id,
            origin["channel"],
            origin["chat_id"],
        )

    @staticmethod
    def _format_partial_progress(result) -> str:
        completed = [e for e in result.tool_events if e["status"] == "ok"]
        failure = next((e for e in reversed(result.tool_events) if e["status"] == "error"), None)
        lines: list[str] = []
        if completed:
            lines.append("Completed steps:")
            for event in completed[-3:]:
                lines.append(f"- {event['name']}: {event['detail']}")
        if failure:
            if lines:
                lines.append("")
            lines.append("Failure:")
            lines.append(f"- {failure['name']}: {failure['detail']}")
        if result.error and not failure:
            if lines:
                lines.append("")
            lines.append("Failure:")
            lines.append(f"- {result.error}")
        return "\n".join(lines) or (result.error or "Error: subagent execution failed.")

    async def cancel_by_session(self, session_key: str) -> int:
        """Cancel all subagents for the given session. Returns count cancelled."""
        tasks = [
            self._running_tasks[tid]
            for tid in self._session_tasks.get(session_key, [])
            if tid in self._running_tasks and not self._running_tasks[tid].done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)
