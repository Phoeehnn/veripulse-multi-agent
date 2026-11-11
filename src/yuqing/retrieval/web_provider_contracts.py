"""Independent web-provider request and receipt types."""
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, TypeAdapter, model_validator


class WebRetrievalRequest(BaseModel):
    """Allow only explicitly supplied search/extract options, never credentials."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    operation: Literal["search", "extract"]
    payload: dict[str, Any]

    @model_validator(mode="after")
    def check_options(self) -> "WebRetrievalRequest":
        """Reject unknown request fields before they can reach the network."""
        common = {"include_usage"}
        allowed = ({"query", "max_results", "search_depth", "topic", "include_domains", "exclude_domains"}
                   if self.operation == "search" else {"urls", "extract_depth", "format"})
        if set(self.payload) - allowed - common:
            raise ValueError("unsupported_web_request_option")
        required = "query" if self.operation == "search" else "urls"
        value = self.payload.get(required)
        if self.operation == "search":
            if not isinstance(value, str) or not value.strip():
                raise ValueError("web_query_required")
        elif not isinstance(value, list) or not value or any(not isinstance(url, str) for url in value):
            raise ValueError("web_urls_required")
        if self.operation == "extract":
            TypeAdapter(list[HttpUrl]).validate_python(value)
        if "max_results" in self.payload:
            count = self.payload["max_results"]
            if type(count) is not int or count <= 0:
                raise ValueError("invalid_max_results")
        if "include_usage" in self.payload and type(self.payload["include_usage"]) is not bool:
            raise ValueError("invalid_include_usage")
        for name in ("include_domains", "exclude_domains"):
            if name in self.payload and (not isinstance(self.payload[name], list)
                    or any(not isinstance(item, str) or not item.strip() for item in self.payload[name])):
                raise ValueError("invalid_domain_list")
        choices = {"search_depth": {"basic", "advanced"}, "topic": {"general", "news", "finance"},
                   "extract_depth": {"basic", "advanced"}, "format": {"markdown", "text"}}
        for name, allowed_values in choices.items():
            if name in self.payload and (not isinstance(self.payload[name], str)
                                        or self.payload[name] not in allowed_values):
                raise ValueError("invalid_web_option_value")
        return self


class WebRetrievalReceipt(BaseModel):
    """Report invocation facts without treating unknown charges as free."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    operation: Literal["search", "extract"]
    status: Literal["success", "not_authorized", "http_error", "timeout", "transport_error", "invalid_response"]
    billing_status: Literal["not_sent", "unknown", "usage_reported"]
    http_status: int | None = None
    provider_request_id: str | None = None
    usage: dict[str, Any] | None = None
    response: dict[str, Any] | None = Field(default=None, repr=False)
