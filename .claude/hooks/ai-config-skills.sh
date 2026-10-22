#!/bin/bash
# SessionStart hook: install the ai-config plugin in cloud sessions.
#
# A cloud session does not install the plugins `.claude/settings.json` enables
# (https://code.claude.com/docs/en/cloud-environments.md, "What carries over").
# Installed while SessionStart hooks run, the plugin still loads for the session's
# first prompt once `reloadSkills` has Claude Code re-scan skills:
# its own hooks run, and its skills appear in `~/.claude/skills` as links named
# without the prefix (`/ardia`), except where another skill has the name (`/ai-config:steward`).
# Neither this hook nor the plugin makes those links (tested 2026-10-02).
#
# Local sessions are left alone: there, `enabledPlugins` installs the plugin itself.

set -u

[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0

plugin="ai-config@Morrison-Lab"
installed="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/plugins/installed_plugins.json"

# A resumed container already has the plugin, and loads it at startup.
if jq -e --arg p "$plugin" '.plugins | has($p)' "$installed" >/dev/null 2>&1; then
  exit 0
fi

if {
  claude plugin marketplace add Morrison-Lab/ai-config &&
    claude plugin install "$plugin"
} >&2; then
  echo '{"hookSpecificOutput": {"hookEventName": "SessionStart", "reloadSkills": true}}'
else
  echo "ai-config-skills.sh: could not install $plugin; its skills are unavailable this session" >&2
fi
