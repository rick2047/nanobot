"""Shared provider factory used by the main agent and subagents."""

from __future__ import annotations

from dataclasses import dataclass

from nanobot.config.schema import Config
from nanobot.providers.base import GenerationSettings, LLMProvider
from nanobot.providers.registry import ProviderSpec, find_by_name


class ProviderConfigurationError(ValueError):
    """Raised when the configured provider cannot be constructed."""


@dataclass(frozen=True)
class ProviderRequest:
    """Resolved provider/model/generation inputs for one provider instance."""

    provider_name: str | None
    model: str
    generation: GenerationSettings


class ProviderFactory:
    """Instantiate configured LLM providers from the shared config model."""

    def __init__(self, config: Config):
        self._config = config

    @property
    def defaults(self) -> ProviderRequest:
        defaults = self._config.agents.defaults
        return self.resolve_request(
            model=defaults.model,
            temperature=defaults.temperature,
            max_tokens=defaults.max_tokens,
            reasoning_effort=defaults.reasoning_effort,
        )

    def resolve_request(
        self,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> ProviderRequest:
        defaults = self._config.agents.defaults
        resolved_model = model or defaults.model
        resolved_provider_name = self._resolve_provider_name(provider_name, resolved_model)
        generation = GenerationSettings(
            temperature=defaults.temperature if temperature is None else temperature,
            max_tokens=defaults.max_tokens if max_tokens is None else max_tokens,
            reasoning_effort=(
                defaults.reasoning_effort if reasoning_effort is None else reasoning_effort
            ),
        )
        return ProviderRequest(
            provider_name=resolved_provider_name,
            model=resolved_model,
            generation=generation,
        )

    def create(
        self,
        request: ProviderRequest | None = None,
    ) -> LLMProvider:
        resolved = request or self.defaults
        provider_config, spec = self._resolve_provider_config(resolved.provider_name, resolved.model)
        backend = spec.backend if spec else "openai_compat"

        self._validate_provider_config(
            backend=backend,
            provider_name=resolved.provider_name,
            provider_config=provider_config,
            model=resolved.model,
            spec=spec,
        )

        provider = self._instantiate_provider(
            backend=backend,
            provider_config=provider_config,
            model=resolved.model,
            spec=spec,
        )
        provider.generation = resolved.generation
        return provider

    def _resolve_provider_name(self, provider_name: str | None, model: str) -> str | None:
        if provider_name:
            spec = find_by_name(provider_name)
            if spec is None:
                raise ProviderConfigurationError(f"Unknown provider: {provider_name}")
            return spec.name
        return self._config.get_provider_name(model)

    def _resolve_provider_config(
        self,
        provider_name: str | None,
        model: str,
    ):
        if provider_name:
            spec = find_by_name(provider_name)
            provider_config = getattr(self._config.providers, spec.name, None) if spec else None
            return provider_config, spec
        provider_config = self._config.get_provider(model)
        detected_name = self._config.get_provider_name(model)
        return provider_config, find_by_name(detected_name) if detected_name else None

    @staticmethod
    def _validate_provider_config(
        *,
        backend: str,
        provider_name: str | None,
        provider_config,
        model: str,
        spec: ProviderSpec | None,
    ) -> None:
        if backend == "azure_openai":
            if not provider_config or not provider_config.api_key or not provider_config.api_base:
                raise ProviderConfigurationError(
                    "Azure OpenAI requires api_key and api_base in config."
                )
            return

        if backend == "openai_compat" and not model.startswith("bedrock/"):
            needs_key = not (provider_config and provider_config.api_key)
            exempt = spec and (spec.is_oauth or spec.is_local or spec.is_direct)
            if needs_key and not exempt:
                provider_label = provider_name or (spec.name if spec else "auto")
                raise ProviderConfigurationError(
                    f"No API key configured for provider '{provider_label}'."
                )

    def _instantiate_provider(
        self,
        *,
        backend: str,
        provider_config,
        model: str,
        spec: ProviderSpec | None,
    ) -> LLMProvider:
        if backend == "openai_codex":
            from nanobot.providers.openai_codex_provider import OpenAICodexProvider

            return OpenAICodexProvider(default_model=model)

        if backend == "github_copilot":
            from nanobot.providers.github_copilot_provider import GitHubCopilotProvider

            return GitHubCopilotProvider(default_model=model)

        if backend == "azure_openai":
            from nanobot.providers.azure_openai_provider import AzureOpenAIProvider

            return AzureOpenAIProvider(
                api_key=provider_config.api_key,
                api_base=provider_config.api_base,
                default_model=model,
            )

        if backend == "anthropic":
            from nanobot.providers.anthropic_provider import AnthropicProvider

            return AnthropicProvider(
                api_key=provider_config.api_key if provider_config else None,
                api_base=self._config.get_api_base(model),
                default_model=model,
                extra_headers=provider_config.extra_headers if provider_config else None,
            )

        from nanobot.providers.openai_compat_provider import OpenAICompatProvider

        return OpenAICompatProvider(
            api_key=provider_config.api_key if provider_config else None,
            api_base=self._config.get_api_base(model),
            default_model=model,
            extra_headers=provider_config.extra_headers if provider_config else None,
            spec=spec,
        )


def create_provider(config: Config, request: ProviderRequest | None = None) -> LLMProvider:
    """Compatibility helper for callers that only need a provider instance."""

    return ProviderFactory(config).create(request)
