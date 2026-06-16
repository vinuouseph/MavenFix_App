"""
Routers and terminal nodes  (Project Fix agent)
────────────────────────────────────────────────
"""
import logging
import asyncio
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from app.agents.project_fix_agent.state import AgentState

logger = logging.getLogger(__name__)

_STUCK_THRESHOLD = 10    # how many times the same fingerprint before "stuck"


# ── Conditional router: after parse_errors ─────────────────────────────────────

def route_check_success(state: AgentState) -> str:
    """Route: success → build_context → abort."""
    if state.get("compiler_success"):
        return "success"
    if state.get("iteration", 0) >= state.get("max_iterations", 20):
        return "abort"
    errors = state.get("errors", [])
    if not errors:
        return "success"
    return "build_context"


# ── Conditional router: after pre_compile_check ────────────────────────────────

def route_pre_compile(state: AgentState) -> str:
    if state.get("status") == "pre_compile_failed":
        logger.warning("[router:pre_compile] Pre-existing errors — routing to escalate")
        return "pre_compile_escalate"
    return "pom_update"


# ── Conditional router: after build_context ────────────────────────────────────

def route_check_stuck(state: AgentState) -> str:
    """Escalate if the same fingerprint keeps repeating past the threshold."""
    counts: dict = state.get("fingerprint_counts") or {}
    if any(v >= _STUCK_THRESHOLD for v in counts.values()):
        return "escalate"
    return "llm_fix"


# ── Terminal nodes ─────────────────────────────────────────────────────────────

def _score_and_store_fixes(state: AgentState, config: RunnableConfig) -> None:
    """
    Ask the LLM to rate confidence for the fix applied in this session,
    then store any fix rated >= 90 in the VectorDB.

    This runs synchronously in the success_node (wrapped with asyncio.run if needed).
    """
    errors          = state.get("all_errors_fixed") or []
    full_diff       = state.get("full_diff", "")
    java_version    = state.get("java_version") or "unknown"

    if not errors:
        logger.info("[success_node] No errors recorded — skipping VectorDB store")
        return

    # Use diff if available, otherwise note it was unavailable
    diff_text = full_diff.strip() if full_diff.strip() else "(diff not captured — fixes were applied via tool calls)"
        
    dispatch_custom_event(
        "project_fix_trace",
        {"id": "vdb_store", "status": "running", "title": "Updating Knowledge Base", "detail": "Evaluating applied fixes for knowledge retention..."},
        config=config,
    )
        
    matched_errors = state.get("matched_known_errors") or []

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        from app.llm.llm_registry import build_llm_model
        from app.llm.fix_knowledge_store import store_fix_if_confident

        llm = build_llm_model()

        error_texts = "\n".join(
            f"[{i}] {e.get('error_code', '')[:200]}" for i, e in enumerate(errors[:10])
        )
        sys_msg = SystemMessage(
            content=(
                "You are a Java compiler error analyst. "
                "Given a list of compiler errors and the diff of fixes applied, "
                "rate your confidence (0-100) that the fix is correct and complete. "
                "Reply ONLY with a JSON array of objects: "
                '[{{"error_index": <int>, "fix_summary": "<one sentence>", "confidence": <int>}}]. '
                "No prose, no markdown."
            )
        )
        human_msg = HumanMessage(
            content=(
                f"Compiler errors that were fixed:\n{error_texts}\n\n"
                f"Diff applied:\n{diff_text[:3000]}\n\n"
                "Rate each fix and reply in the required JSON format."
            )
        )

        response = llm.invoke([sys_msg, human_msg])
        content  = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            content = " ".join(
                item["text"] if isinstance(item, dict) and "text" in item else str(item)
                for item in content
            )

        import re, json
        json_match = re.search(r"\[.*\]", content, re.DOTALL)
        if not json_match:
            logger.warning("[success_node] Could not parse confidence JSON from LLM")
            dispatch_custom_event(
                "project_fix_trace",
                {"id": "vdb_store", "status": "completed", "title": "Knowledge Base Checked", "detail": "No valid fixes parsed for storage."},
                config=config,
            )
            return

        ratings: list[dict] = json.loads(json_match.group(0))
        stored_count = 0
        for rating in ratings:
            idx = rating.get("error_index")
            if idx is None or not isinstance(idx, int) or idx < 0 or idx >= len(errors):
                continue
                
            error_text = errors[idx].get("error_code", "")
            fix_text   = rating.get("fix_summary", "")
            confidence = int(rating.get("confidence", 0))
            
            # Skip if we already used a known fix from VectorDB for this error
            is_known = False
            for me in matched_errors:
                if error_text.strip() and (error_text in me or me in error_text):
                    is_known = True
                    break
            
            if is_known:
                logger.info(f"[success_node] Skipping VectorDB store; error already known: {error_text[:60]}")
                continue

            store_fix_if_confident(
                java_version=java_version,
                error=error_text,
                fix=fix_text,
                confidence=confidence,
            )
            if confidence >= 90:
                stored_count += 1
            
        dispatch_custom_event(
            "project_fix_trace",
            {"id": "vdb_store", "status": "completed", "title": "Knowledge Base Updated", "detail": f"Successfully learned {stored_count} new fix(es)."},
            config=config,
        )

    except Exception as exc:
        logger.warning(f"[success_node] _score_and_store_fixes failed: {exc}", exc_info=True)
        dispatch_custom_event(
            "project_fix_trace",
            {"id": "vdb_store", "status": "error", "title": "Knowledge Base Update Failed", "detail": "Failed to store learned fixes."},
            config=config,
        )


def success_node(state: AgentState, config: RunnableConfig) -> AgentState:
    logger.info("[project_fix] SUCCESS — compilation succeeded.")

    # Store high-confidence fixes in VectorDB (best-effort, non-blocking)
    try:
        _score_and_store_fixes(state, config)
    except Exception as exc:
        logger.warning(f"[success_node] Knowledge store step failed: {exc}")

    dispatch_custom_event(
        "project_fix_trace",
        {"id": "terminal", "status": "completed",
         "title": "Compilation Successful",
         "detail": f"All errors resolved after {state.get('iteration', 1)} iteration(s)."},
        config=config,
    )

    return {
        **state,
        "status":        "success",
        "final_message": (
            f"✅ Compilation succeeded after {state.get('iteration', 1)} iteration(s). "
            "Your fixed project is ready for download."
        ),
    }


def abort_node(state: AgentState, config: RunnableConfig) -> AgentState:
    remaining = len(state.get("errors", []))
    logger.warning(f"[project_fix] ABORT — max iterations reached, {remaining} error(s) left.")
    dispatch_custom_event(
        "project_fix_trace",
        {"id": "terminal", "status": "error",
         "title": "Max Iterations Reached",
         "detail": f"{remaining} error(s) still unresolved."},
        config=config,
    )
    return {
        **state,
        "status":        "abort",
        "final_message": (
            f"⚠️ Max iterations ({state.get('max_iterations', 20)}) reached. "
            f"{remaining} error(s) remain. Partial fixes applied — download available."
        ),
    }


def escalate_node(state: AgentState, config: RunnableConfig) -> AgentState:
    logger.warning("[project_fix] ESCALATE — agent stuck on repeating errors.")
    dispatch_custom_event(
        "project_fix_trace",
        {"id": "terminal", "status": "error",
         "title": "Agent Escalating",
         "detail": "Repeating errors detected — manual review required."},
        config=config,
    )
    return {
        **state,
        "status":        "escalate",
        "final_message": (
            "⚠️ Agent escalating — the same errors repeated too many times. "
            "Manual intervention required. Partial fixes applied."
        ),
    }


def pre_compile_escalate_node(state: AgentState, config: RunnableConfig) -> AgentState:
    logger.warning("[project_fix] PRE-COMPILE ESCALATE — pre-existing errors detected.")
    dispatch_custom_event(
        "project_fix_trace",
        {"id": "terminal", "status": "error",
         "title": "Agent Escalating",
         "detail": "Pre-existing errors detected before any changes. Manual review required."},
        config=config,
    )
    return {
        **state,
        "status": "pre_compile_failed",
    }

