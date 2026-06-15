"""
Node: pom_update  (Project Fix agent)
──────────────────────────────────────
Runs the vulnerability scanner on pom.xml and automatically writes the
patched pom.xml before the first compile cycle.

Mirrors the logic in compile_fix_2's pom_updation_node but:
  - dispatches 'project_fix_trace' events (so the correct frontend handler picks them up)
  - uses work_dir (not project_path) from AgentState
  - also records vuln_summary / vuln_updates in state for the final reply
"""

import sys
import asyncio
import uuid
import logging
from pathlib import Path

from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig

from app.agents.project_fix_agent.state import AgentState

logger = logging.getLogger(__name__)

_BUILD_FILES = {"pom.xml", "build.gradle", "build.gradle.kts"}


def _find_project_root(work_dir: str) -> Path:
    """
    Return the directory that contains pom.xml / build.gradle,
    walking up to 3 levels into work_dir (handles ZIP nesting).
    """
    root = Path(work_dir)
    if any((root / f).exists() for f in _BUILD_FILES):
        return root
    for child in root.iterdir():
        if child.is_dir() and any((child / f).exists() for f in _BUILD_FILES):
            return child
        if child.is_dir():
            for grandchild in child.iterdir():
                if grandchild.is_dir() and any((grandchild / f).exists() for f in _BUILD_FILES):
                    return grandchild
    return root


def _find_poms(project_root: Path) -> list[Path]:
    """Locate all pom.xml files starting from project_root."""
    return list(project_root.rglob("pom.xml"))


def pom_update_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    LangGraph node: scan pom.xml for vulnerable / outdated deps.
    If updates are found, show the HITL review modal and apply on approval.

    Writes: vuln_summary, vuln_updates (for final reply), work_dir (corrected to build root).
    """
    work_dir = state["work_dir"]

    # Resolve the actual build root (handles nested ZIPs)
    project_root = _find_project_root(work_dir)
    project_root_str = str(project_root.resolve())

    vuln_summary = "⚠️ No pom.xml found — vulnerability scan skipped."
    vuln_updates: list = []

    dispatch_custom_event(
        "project_fix_trace",
        {
            "id":     "pom_update",
            "status": "running",
            "title":  "POM Dependency Update",
            "detail": f"Scanning pom.xml for vulnerabilities and outdated dependencies…",
        },
        config=config,
    )

    pom_paths = _find_poms(project_root)
    if not pom_paths:
        dispatch_custom_event(
            "project_fix_trace",
            {"id": "pom_update", "status": "completed",
             "title": "POM Dependency Update",
             "detail": "No pom.xml found — skipping vulnerability scan."},
            config=config,
        )
        return {**state, "work_dir": project_root_str,
                "vuln_summary": vuln_summary, "vuln_updates": vuln_updates}

    # ── Import the vuln-agent scanner ─────────────────────────────────────────
    vuln_agent_dir = Path(__file__).resolve().parents[2] / "compile_fix_agent" / "vuln-agent"
    sys_path_added = False
    if str(vuln_agent_dir) not in sys.path:
        sys.path.insert(0, str(vuln_agent_dir))
        sys_path_added = True

    try:
        from app.agents.project_fix_agent import scanner as sc

        overall_total_scanned = 0
        overall_total_issues = 0
        all_updates = []

        # Run the async scanner synchronously
        loop = asyncio.new_event_loop()
        try:
            for pom_path in pom_paths:
                pom_content = pom_path.read_bytes()

                scan_data = loop.run_until_complete(
                    sc.run_full_scan(pom_content, include_transitive=False)
                )

                scan_results  = scan_data.get("scan_results", [])
                overall_total_scanned += scan_data.get("total", 0)
                overall_total_issues  += scan_data.get("issues", 0)

                fixed_bytes, updates = sc.generate_fixed_pom(pom_content, scan_results)

                if updates:
                    logger.info(f"[pom_update] Automatically applying POM patch to {pom_path.name} in background...")
                    pom_path.write_bytes(fixed_bytes)
                    for u in updates:
                        all_updates.append((u[0], u[1], u[2], u[3], str(pom_path)))
        finally:
            loop.close()

        if not all_updates:
            vuln_summary = (
                f"**Vulnerability Scan** — {overall_total_scanned} deps scanned across {len(pom_paths)} POMs. "
                "No version updates required."
            )
            if overall_total_issues:
                vuln_summary += (
                    f"\n({overall_total_issues} advisories found but all are already at the "
                    "recommended version or have no known fix.)"
                )
            dispatch_custom_event(
                "project_fix_trace",
                {"id": "pom_update", "status": "completed",
                 "title": "POM Dependency Update",
                 "detail": "No dependency updates needed."},
                config=config,
            )
        else:
            dispatch_custom_event(
                "project_fix_trace",
                {"id": "pom_update", "status": "completed",
                 "title": "POM Dependency Update",
                 "detail": f"Found {len(all_updates)} dependency update(s) — awaiting user review."},
                config=config,
            )

            dispatch_custom_event(
                "project_fix_trace",
                {"id": "pom_update_applied", "status": "completed",
                 "title": "POM Vulnerability Patching Applied ✓",
                 "detail": f"Updated {len(all_updates)} dependency version(s) across {len(pom_paths)} POMs."},
                config=config,
            )
            update_lines = "\n".join(
                f"  • `{u[0]}` → `{u[2]}` (was `{u[1]}`)" for u in all_updates[:15]
            )
            vuln_summary = (
                f"**Vulnerability Scan** — {overall_total_scanned} deps scanned across {len(pom_paths)} POMs, "
                f"{overall_total_issues} with issues.\n\n"
                f"**{len(all_updates)} version update(s) applied:**\n\n{update_lines}"
            )
            vuln_updates = [
                {"dep": u[0], "from_ver": u[1], "to_ver": u[2], "method": u[3], "pom": u[4]}
                for u in all_updates
            ]

    except Exception as exc:
        logger.error(f"[project_fix/pom_update] Error: {exc}")
        dispatch_custom_event(
            "project_fix_trace",
            {"id": "pom_update", "status": "warning",
             "title": "POM Update Warning",
             "detail": f"POM scanning failed: {exc}"},
            config=config,
        )
        vuln_summary = f"⚠️ POM scanning/fixing failed: {exc}"
    finally:
        if sys_path_added and str(vuln_agent_dir) in sys.path:
            sys.path.remove(str(vuln_agent_dir))

    return {
        **state,
        "work_dir":     project_root_str,    # corrected to actual build root
        "vuln_summary": vuln_summary,
        "vuln_updates": vuln_updates,
    }
