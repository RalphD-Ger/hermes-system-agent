"""
deep_coding.py — Hermes Deep Coding Skill
=========================================
Leitet schwere Coding-Anfragen an Claude weiter.
Agent preprocesst das Problem mit Grok, dann ruft er
dieses Modul auf um die Rechenpower von Opus 4.8 oder
Fable 5 zu nutzen.

Verwendung durch Agent:
    from deep_coding import deep_code

    result = await deep_code(problem="...", heavy=False)
    print(result["result"])
    print(f"Kosten: ${result['cost']:.4f}")
"""

import os
from anthropic import AsyncAnthropic

# ════════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════════

MODEL_STANDARD = "claude-opus-4-8"    # Stark, effizient — Standard-Coding
MODEL_HEAVY    = "claude-fable-5"     # Mythos-Klasse — komplexe Architektur, grosse Codebases

# Preise pro 1 Million Tokens (Stand Juni 2026)
RATES = {
    MODEL_STANDARD: {"input": 5.00,  "output": 25.00},
    MODEL_HEAVY:    {"input": 10.00, "output": 50.00},
}

# Maximale Output-Tokens pro Modell
MAX_TOKENS = {
    MODEL_STANDARD: 8192,
    MODEL_HEAVY:    16384,   # Fable 5 unterstuetzt bis 128k, 16k als vernuenftiger Default
}

SYSTEM_PROMPT = """\
Du bist ein Elite-Software-Architekt und Coding-Spezialist mit Expertise in:
- C++ und Low-Level-Optimierung / High-Performance Computing
- Python, Rust, Go und modernen Backend-Sprachen
- Systemarchitektur, Concurrency, Speicherverwaltung
- AI/ML Engineering, Inference-Pipelines, Edge-Deployment

Dein Auftrag: Setze den folgenden Architektur-Plan oder die Coding-Anfrage
perfekt in fehlerfreien, hochoptimierten Code um.
Halte Erklaerungen extrem kurz — Code hat Prioritaet.
Achte penibel auf: Korrektheit, Performance, Lesbarkeit.\
"""

# ════════════════════════════════════════════════════════════════════════
#  HAUPTFUNKTION
# ════════════════════════════════════════════════════════════════════════

async def deep_code(
    problem: str,
    heavy: bool = False,
    system_prompt: str = None,
    api_key: str = None,
) -> dict:
    """
    Sendet ein Coding-Problem an Claude und gibt ein strukturiertes Ergebnis zurueck.

    Args:
        problem:       Die (von Hermes/Grok vorverarbeitete) Coding-Anfrage.
        heavy:         True  -> Claude Fable 5  (komplex, teuer)
                       False -> Claude Opus 4.8 (Standard, effizient)
        system_prompt: Optionaler Override des System-Prompts.
        api_key:       Optionaler Override des API-Keys (sonst ANTHROPIC_API_KEY aus Env).

    Returns:
        {
            "result":        str,   # Generierter Code / Antwort
            "model":         str,   # Genutztes Modell
            "input_tokens":  int,
            "output_tokens": int,
            "cost":          float, # USD
        }

    Raises:
        EnvironmentError:  API-Key nicht gesetzt.
        RuntimeError:      API-Fehler.
    """
    model = MODEL_HEAVY if heavy else MODEL_STANDARD

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY nicht in den Umgebungsvariablen gefunden. "
            "Bitte in der .env oder Systemumgebung setzen."
        )

    client = AsyncAnthropic(api_key=key)

    try:
        message = await client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS[model],
            system=[{
                "type": "text",
                "text": system_prompt or SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},   # 90% Rabatt auf gecachte Tokens
            }],
            messages=[{"role": "user", "content": problem}],
        )
    except Exception as e:
        if "rate_limit" in str(e).lower():
            raise RuntimeError("Anthropic Rate Limit erreicht. Warte kurz oder nutze normales Coding.") from e
        raise RuntimeError(f"Anthropic API-Fehler: {e}") from e

    input_tokens  = message.usage.input_tokens
    output_tokens = message.usage.output_tokens
    cache_read    = getattr(message.usage, "cache_read_input_tokens", 0) or 0
    cache_write   = getattr(message.usage, "cache_creation_input_tokens", 0) or 0    
    rate  = RATES[model]
    # Cache-Write kostet 125% des normalen Input-Preises, Cache-Read nur 10%
    cost = (
        ((input_tokens - cache_read - cache_write) / 1_000_000) * rate["input"]
        + (cache_write / 1_000_000) * rate["input"] * 1.25
        + (cache_read  / 1_000_000) * rate["input"] * 0.10
        + (output_tokens / 1_000_000) * rate["output"]
    )
 
    return {
        "result":               message.content[0].text,
        "model":                model,
        "input_tokens":         input_tokens,
        "output_tokens":        output_tokens,
        "cache_read_tokens":    cache_read,
        "cache_write_tokens":   cache_write,
        "cost":                 cost,
    }
 


def format_cost_report(result: dict) -> str:
    """Gibt einen lesbaren Kosten-Report als String zurueck (fuer Discord / Terminal)."""
    cache_info = ""
    if result.get("cache_read_tokens"):
        cache_info = f"\n💾 Cache-Hit: {result['cache_read_tokens']:,} Tokens (90% Rabatt)"
    elif result.get("cache_write_tokens"):
        cache_info = f"\n💾 Cache-Write: {result['cache_write_tokens']:,} Tokens (naechster Call guenstiger)"
    return (
        f"🤖 **Modell:** {result['model']}\n"
        f"📥 Input:  {result['input_tokens']:,} Tokens\n"
        f"📤 Output: {result['output_tokens']:,} Tokens"
        f"{cache_info}\n"
        f"💰 Kosten: ${result['cost']:.4f} USD"
    )


# ════════════════════════════════════════════════════════════════════════
#  CLI — zum direkten Testen ohne Hermes Agent
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import asyncio
    import sys

    if len(sys.argv) < 2:
        print("Usage: python deep_coding.py '<problem>' [--heavy]")
        sys.exit(0)

    problem_input = sys.argv[1]
    use_heavy     = len(sys.argv) > 2 and sys.argv[2] == "--heavy"

    print(f"🚀 Deep Coding — {'HEAVY (Fable 5)' if use_heavy else 'Standard (Opus 4.8)'}")
    print("=" * 50)

    async def _run():
        result = await deep_code(problem_input, heavy=use_heavy)
        print(result["result"])
        print("\n" + "=" * 50)
        print(format_cost_report(result))

    asyncio.run(_run())