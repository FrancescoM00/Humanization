"""Profili espliciti dei giudici, senza cambiare le richieste già congelate."""
from __future__ import annotations

from .naturalness import GroqClient


NATIVE_NON_REASONING_MODELS = frozenset({"llama-3.3-70b-versatile"})


def _native_profile(plan):
    control = plan.get("reasoning_control")
    if control is None:
        return False
    if control != "native_non_reasoning":
        raise ValueError("Profilo di ragionamento del giudice non riconosciuto")
    if plan.get("model") not in NATIVE_NON_REASONING_MODELS:
        raise ValueError("Il profilo nativo richiede un modello non reasoning esplicitamente ammesso")
    if "reasoning_effort" not in plan or plan["reasoning_effort"] is not None:
        raise ValueError("Il profilo nativo deve dichiarare reasoning_effort nullo nel piano")
    return True


def validate_judge_payload(plan, payload):
    """Verifica il profilo dichiarato senza aggiungere o rimuovere parametri."""
    native = _native_profile(plan)
    if not isinstance(payload, dict) or payload.get("model") != plan.get("model"):
        raise ValueError("Il modello della richiesta non coincide con il piano")
    if native:
        if "reasoning_effort" in payload:
            raise ValueError("Il modello non reasoning nativo richiede di omettere reasoning_effort")
    elif payload.get("reasoning_effort") != "none":
        raise ValueError("L'opzione no-thinking non coincide con il piano storico")


class NativeNonReasoningGroqClient(GroqClient):
    """Usa soltanto il modello nativo dichiarato e ne verifica il catalogo."""

    def check_model(self, model):
        if model not in NATIVE_NON_REASONING_MODELS:
            raise ValueError("Questo client richiede un modello non reasoning esplicitamente ammesso")
        result = super().check_model(model)
        result.update(
            reasoning_control="native_non_reasoning",
            no_thinking_option=(
                "Modello non reasoning nativo: reasoning_effort non viene inviato; "
                "il parser rifiuta comunque eventuale ragionamento nella risposta"
            ),
        )
        return result


def client_for_plan(plan):
    """Costruisce il client del profilo esplicito, senza fallback fra modelli."""
    if _native_profile(plan):
        return NativeNonReasoningGroqClient()
    return GroqClient()
