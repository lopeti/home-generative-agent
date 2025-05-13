"""Config flow for Home Generative Agent integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import callback
from homeassistant.const import (
    CONF_API_KEY,
    CONF_LLM_HASS_API,
)
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TemplateSelector,
)
from langchain_openai import ChatOpenAI

from .const import (
    CONF_CHAT_MODEL,
    CONF_CHAT_MODEL_LOCATION,
    CONF_CHAT_MODEL_TEMPERATURE,
    CONF_EDGE_CHAT_MODEL,
    CONF_EDGE_CHAT_MODEL_TEMPERATURE,
    CONF_EDGE_CHAT_MODEL_TOP_P,
    CONF_EMBEDDING_MODEL,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_SUMMARIZATION_MODEL,
    CONF_SUMMARIZATION_MODEL_TEMPERATURE,
    CONF_SUMMARIZATION_MODEL_TOP_P,
    CONF_VIDEO_ANALYZER_MODE,
    CONF_VLM,
    CONF_VLM_TEMPERATURE,
    CONF_VLM_TOP_P,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_CHAT_MODEL_LOCATION,
    RECOMMENDED_CHAT_MODEL_TEMPERATURE,
    RECOMMENDED_EDGE_CHAT_MODEL,
    RECOMMENDED_EDGE_CHAT_MODEL_TEMPERATURE,
    RECOMMENDED_EDGE_CHAT_MODEL_TOP_P,
    RECOMMENDED_EMBEDDING_MODEL,
    RECOMMENDED_SUMMARIZATION_MODEL,
    RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
    RECOMMENDED_SUMMARIZATION_MODEL_TOP_P,
    RECOMMENDED_VIDEO_ANALYZER_MODE,
    RECOMMENDED_VLM,
    RECOMMENDED_VLM_TEMPERATURE,
    RECOMMENDED_VLM_TOP_P,
    # --- ezek az új DB-konstansok ---
    CONF_DB_HOST,
    CONF_DB_PORT,
    CONF_DB_NAME,
    CONF_DB_USER,
    CONF_DB_PASSWORD,
    CONF_DB_SSLMODE,
    DEFAULT_DB_HOST,
    DEFAULT_DB_PORT,
    DEFAULT_DB_NAME,
    DEFAULT_DB_USER,
    DEFAULT_DB_PASSWORD,
    DEFAULT_DB_SSLMODE,
)

if TYPE_CHECKING:
    from types import MappingProxyType
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.typing import VolDictType

LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema({vol.Required(CONF_API_KEY): str})

RECOMMENDED_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: llm.LLM_API_ASSIST,
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
    CONF_VIDEO_ANALYZER_MODE: RECOMMENDED_VIDEO_ANALYZER_MODE,
}

async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, Any]:
    """Validate connectivity with provided input."""
    client = ChatOpenAI(api_key=data[CONF_API_KEY], async_client=get_async_client(hass))
    await hass.async_add_executor_job(client.bind(timeout=10).get_name)

class HomeGenerativeAgentConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Home Generative Agent."""
    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors: dict[str, str] = {}
        try:
            await validate_input(self.hass, user_input)
        except CannotConnectError:
            errors["base"] = "cannot_connect"
        except InvalidAuthError:
            errors["base"] = "invalid_auth"
        except Exception:
            LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            return self.async_create_entry(
                title="HGA", data=user_input, options=RECOMMENDED_OPTIONS
            )

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> HomeGenerativeAgentOptionsFlow:
        """Create the options flow."""
        return HomeGenerativeAgentOptionsFlow(config_entry)

class HomeGenerativeAgentOptionsFlow(OptionsFlow):
    """Options flow handler."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        self.config_entry = config_entry
        self.last_rendered_recommended = config_entry.options.get(
            CONF_RECOMMENDED, False
        )

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options flow."""
        options = self.config_entry.options

        # --- chat/LLM séma (eredeti) ---
        schema: VolDictType = {
            vol.Optional(
                CONF_PROMPT,
                description={"suggested_value": options.get(CONF_PROMPT)},
                default=llm.DEFAULT_INSTRUCTIONS_PROMPT,
            ): TemplateSelector(),
            vol.Optional(
                CONF_LLM_HASS_API,
                description={"suggested_value": options.get(CONF_LLM_HASS_API)},
                default="none",
            ): SelectSelector(SelectSelectorConfig(options=[
                SelectOptionDict(label="No control", value="none"),
                *[
                    SelectOptionDict(label=api.name, value=api.id)
                    for api in llm.async_get_apis(self.hass)
                ]
            ])),
            vol.Optional(
                CONF_VIDEO_ANALYZER_MODE,
                description={"suggested_value": options.get(CONF_VIDEO_ANALYZER_MODE)},
                default=RECOMMENDED_VIDEO_ANALYZER_MODE,
            ): SelectSelector(SelectSelectorConfig(options=[
                SelectOptionDict(label="Disable", value="disable"),
                SelectOptionDict(label="Notify on anomaly", value="notify_on_anomaly"),
                SelectOptionDict(label="Always notify", value="always_notify"),
            ])),
            vol.Required(
                CONF_RECOMMENDED,
                description={"suggested_value": options.get(CONF_RECOMMENDED)},
                default=options.get(CONF_RECOMMENDED, False),
            ): bool,
        }
        if not options.get(CONF_RECOMMENDED):
            schema.update({
                vol.Optional(
                    CONF_CHAT_MODEL_LOCATION,
                    description={"suggested_value": options.get(CONF_CHAT_MODEL_LOCATION)},
                    default=RECOMMENDED_CHAT_MODEL_LOCATION,
                ): SelectSelector(SelectSelectorConfig(options=[
                    SelectOptionDict(label="cloud", value="cloud"),
                    SelectOptionDict(label="edge", value="edge"),
                ])),
                vol.Optional(
                    CONF_CHAT_MODEL,
                    description={"suggested_value": options.get(CONF_CHAT_MODEL)},
                    default=RECOMMENDED_CHAT_MODEL,
                ): str,
                vol.Optional(
                    CONF_CHAT_MODEL_TEMPERATURE,
                    description={"suggested_value": options.get(CONF_CHAT_MODEL_TEMPERATURE)},
                    default=RECOMMENDED_CHAT_MODEL_TEMPERATURE,
                ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
                # … (a többi eredeti mező változatlanul)
            })

        # --- ide illesztjük be az adatbázis-kulcsokat ---
        schema.update({
            vol.Optional(
                CONF_DB_HOST,
                default=options.get(CONF_DB_HOST, DEFAULT_DB_HOST),
            ): str,
            vol.Optional(
                CONF_DB_PORT,
                default=options.get(CONF_DB_PORT, DEFAULT_DB_PORT),
            ): int,
            vol.Optional(
                CONF_DB_NAME,
                default=options.get(CONF_DB_NAME, DEFAULT_DB_NAME),
            ): str,
            vol.Optional(
                CONF_DB_USER,
                default=options.get(CONF_DB_USER, DEFAULT_DB_USER),
            ): str,
            vol.Optional(
                CONF_DB_PASSWORD,
                default=options.get(CONF_DB_PASSWORD, DEFAULT_DB_PASSWORD),
            ): str,
            vol.Optional(
                CONF_DB_SSLMODE,
                default=options.get(CONF_DB_SSLMODE, DEFAULT_DB_SSLMODE),
            ): bool,
        })

        if user_input is not None:
            return self.async_create_entry(title="Database settings", data=user_input)

        return self.async_show_form(step_id="init", data_schema=vol.Schema(schema))

class CannotConnectError(HomeAssistantError):
    """Error to indicate we cannot connect."""

class InvalidAuthError(HomeAssistantError):
    """Error to indicate invalid auth."""
