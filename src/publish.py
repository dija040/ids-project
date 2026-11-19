"""
Publish the small result files (results/tables, results/figures) from Colab to GitHub.

Why: the notebooks run on Colab, so their tables and figures are written to the
Google Drive copy of the repository. Committing them makes them available on
GitHub (and on the PC after `git pull`) for the report.

Security:
- The GitHub token is read from Colab's *Secrets* (key icon in the left panel),
  never written in a notebook, a file or the git configuration.
- It is passed to git as a one-off HTTP header for the push only, and masked
  in any message that is printed.
- Only the results/ folder is committed (data/ is git-ignored anyway).

Commits are attributed to the author of the latest commit of the repository
(i.e. the repository owner), so no new contributor appears on GitHub.
"""
from __future__ import annotations

import base64
import subprocess

from . import config

SETUP_HELP = """\
To enable publishing (one-time setup):
  1. GitHub > Settings > Developer settings > Personal access tokens > Fine-grained tokens
     > Generate new token.
     Repository access: "Only select repositories" -> ids-project.
     Permissions: Repository permissions -> Contents: "Read and write".
  2. In Colab: click the key icon (Secrets) in the left panel > Add new secret.
     Name: GITHUB_TOKEN   Value: the token   Then switch on "Notebook access".
  3. Re-run this cell.
Results were NOT published; nothing else is affected."""


def publish_results(message: str, secret_name: str = "GITHUB_TOKEN", max_attempts: int = 3) -> bool:
    """Commit results/ and push it to GitHub. Returns True if something was published."""
    try:
        from google.colab import userdata
    except ImportError:
        print("Not running on Colab: commit results/ from your own computer instead.")
        return False
    try:
        token = userdata.get(secret_name)
    except Exception:  # secret missing or notebook access not granted
        token = None
    if not token:
        print(f"No '{secret_name}' secret available.\n{SETUP_HELP}")
        return False

    auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    auth_cfg = ["-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {auth}"]
    root = str(config.PROJECT_ROOT)

    def git(*args, with_auth=False):
        cmd = ["git", "-C", root, *(auth_cfg if with_auth else []), *args]
        r = subprocess.run(cmd, capture_output=True, text=True)
        out = (r.stdout + r.stderr).replace(token, "***").replace(auth, "***").strip()
        return r.returncode, out

    # Attribute the commit to the repository owner (author of the latest commit).
    _, name = git("log", "-1", "--format=%an")
    _, email = git("log", "-1", "--format=%ae")
    ident = ["-c", f"user.name={name}", "-c", f"user.email={email}"]

    git("add", "results")
    staged, _ = git("diff", "--cached", "--quiet")
    if staged == 0:
        print("No new or changed result files: nothing to publish.")
        return False
    rc, out = git(*ident, "commit", "-m", message)
    if rc != 0:
        raise RuntimeError(f"git commit failed: {out}")

    error = ""
    for attempt in range(1, max_attempts + 1):
        rc, out = git("push", "origin", "HEAD:main", with_auth=True)
        if rc == 0:
            _, sha = git("log", "-1", "--format=%h")
            print(f"✅ Results published to GitHub (commit {sha}).")
            return True
        error = out
        # Most likely the code was updated on GitHub during the run: replay our
        # results commit on top of the latest version and try again.
        git("fetch", "origin", with_auth=True)
        rc, out2 = git(*ident, "rebase", "origin/main")
        if rc != 0:
            git("rebase", "--abort")
            error = out2
            break

    # Failure: undo the local commit but KEEP the result files on disk. Otherwise the
    # `reset --hard` done by the next notebook's setup cell would delete them.
    git("reset", "--mixed", "origin/main")
    raise RuntimeError(f"Publishing failed (result files are kept on Drive): {error}")
