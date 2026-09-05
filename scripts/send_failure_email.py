"""
send_failure_email.py
Runs only when a prior step in the daily GitHub Actions pipeline fails
(daily_run.yml calls this with `if: failure()`, which can trigger even if
an earlier step's `pip install` itself failed) -- so this uses only the
stdlib (smtplib/email), nothing from requirements.txt, to guarantee it can
still send the alert regardless of what broke upstream.

Usage: python scripts/send_failure_email.py <workflow_name> <run_id>
"""

import os
import sys
import smtplib
from email.mime.text import MIMEText

EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER")

def main():
    workflow = sys.argv[1] if len(sys.argv) > 1 else "unknown workflow"
    run_id = sys.argv[2] if len(sys.argv) > 2 else "unknown"
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    run_url = f"{server}/{repo}/actions/runs/{run_id}" if repo else "(no run URL available)"

    subject = f"[Pipeline FAILED] {workflow}"
    body = (
        f"The daily pipeline failed.\n\n"
        f"Workflow: {workflow}\n"
        f"Run: {run_url}\n\n"
        f"Check the run log for which step failed and why."
    )

    # EMAIL_RECEIVER may hold several comma-separated addresses; smtplib
    # treats a bare string as ONE recipient, so split it into a real list.
    recipients = [a.strip() for a in (EMAIL_RECEIVER or "").split(",") if a.strip()]

    if not EMAIL_SENDER or not EMAIL_PASSWORD or not recipients:
        print("Email credentials missing -- printing failure notice instead:")
        print(body)
        return

    try:
        msg = MIMEText(body, "plain")
        msg["From"] = EMAIL_SENDER
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server_conn:
            server_conn.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server_conn.sendmail(EMAIL_SENDER, recipients, msg.as_string())
        print(f"Failure email sent to {len(recipients)} recipient(s): {subject}")
    except Exception as e:
        print(f"Failure email itself failed to send: {e}")

if __name__ == "__main__":
    main()
