# Single source of truth for the Groq model id.
#
# Groq retires models on a schedule (llama-3.3-70b-versatile was deprecated
# 2026-06-17 for free/developer tiers), and the id was previously hardcoded in
# three modules, so a retirement broke the app in three places at once.
# It is env-overridable: set GROQ_MODEL on the host to switch models without a
# code change or redeploy of the image.
#
# Needs tool-calling support: the agent binds tools to this model.

import os

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
