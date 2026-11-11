"""Lightweight UI-to-core JSON bridge; never imports Agent or writable wallet dependencies."""
from pathlib import Path
import json
import os
import subprocess
from src.yuqing.serve.research_live_ui import LiveUIPlan, LiveUIBinding


class LiveProcessBridge:
    """Bind server-selected worktree, interpreter, root and pinned checklist, without execution."""

    def __init__(self, worktree: Path, core_python: Path, root: Path, checklist: Path,
                 *, allow_live: bool = False) -> None:
        """Accept only D-drive server configuration; no keys enter arguments or return values."""
        paths=[p.resolve() for p in (worktree,core_python,root,checklist)]
        if any(p.drive.lower()!="d:" for p in paths):
            raise ValueError("live_bridge_requires_d_paths")
        self.worktree,self.core_python,self.root,self.checklist=paths
        if not self.root.is_relative_to(self.worktree/"traces") or not self.checklist.is_relative_to(self.worktree):
            raise ValueError("live_bridge_outside_worktree")
        self.allow_live=allow_live
        self.used=False

    def _request(self, identity: str, action: str) -> dict[str, object]:
        """Use a fixed core module with inherited credentials, D temp paths and safe error messages."""
        if action=="execute" and not self.allow_live:
            raise ValueError("live_bridge_execution_not_authorized")
        command=[str(self.core_python),"-B","-m","src.yuqing.serve.research_live_worker",
            "--root",str(self.root),"--checklist",str(self.checklist),"--source-id",identity,"--action",action]
        if action=="execute":
            command.append("--allow-live")
        temp=str(self.worktree/"traces/tmp")
        env=dict(os.environ,PYTHONPATH=str(self.worktree),PYTHONDONTWRITEBYTECODE="1",PYTHONIOENCODING="utf-8",
            HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TEMP=temp,TMP=temp,
            PIP_CACHE_DIR=str(self.worktree/"traces/ui-install/pip-cache"),HF_HOME=str(self.worktree/"traces/ui-install/hf"))
        result=subprocess.run(command,cwd=self.worktree,env=env,capture_output=True,text=True,encoding="utf-8",
            creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0),check=False)
        if result.returncode:
            raise ValueError("live_bridge_core_process_failed")
        try:
            data=json.loads(result.stdout)
        except ValueError as exc:
            raise ValueError("live_bridge_invalid_response") from exc
        if data.get("ok") is not True:
            raise ValueError("live_bridge_preflight_or_execution_failed")
        return data["result"]

    def preview(self, identity: str) -> LiveUIPlan:
        """Request read-only core metadata only; never grant or execute a live action."""
        data=self._request(identity,"preview")
        data["material_urls"]=tuple(data["material_urls"])
        return LiveUIPlan(**data)

    def execute(self, identity: str) -> str:
        """Delegate only a server-enabled, explicitly confirmed UI action to the core executor."""
        if self.used or not self.allow_live:
            raise ValueError("live_bridge_consumed_or_not_authorized")
        self.used=True
        return self._request(identity,"execute")["research_id"]

    def binding(self) -> LiveUIBinding | None:
        """Keep the default page's execution APIs absent until backend installation is authorized."""
        return LiveUIBinding(self.preview,self.execute) if self.allow_live else None
