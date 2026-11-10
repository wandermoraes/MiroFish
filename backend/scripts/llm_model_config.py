"""
Configuração de modelo para os agentes da simulação (camel-ai / OASIS).

Os agentes usam function tools (postar, curtir, comentar...). Em modelos
GPT-6, a Chat Completions só aceita function tools com reasoning_effort
igual a 'none'; com o padrão do modelo, cada ação falha com HTTP 400.

A variável de ambiente LLM_REASONING_EFFORT, se definida, tem prioridade.
"""

import os
import re
from typing import Any, Dict, Optional

_NEEDS_NO_REASONING = re.compile(r"^(?:[\w.-]+/)?gpt-(?:[6-9]|\d{2,})")


def camel_model_config(model: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return a camel model_config_dict for the model, or None to keep defaults."""
    override = os.environ.get("LLM_REASONING_EFFORT", "").strip()
    if override:
        return {"reasoning_effort": override}
    if model and _NEEDS_NO_REASONING.match(model.strip().lower()):
        return {"reasoning_effort": "none"}
    return None
