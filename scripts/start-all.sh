#!/bin/zsh
set -e

# PATH para launchd (incluye /usr/local/bin donde está tu node)
/bin/launchctl setenv PATH "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"

export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"

NODE="/usr/local/bin/node"
if [ ! -x "$NODE" ]; then
  echo "❌ Node no ejecutable en $NODE" >&2
  exit 1
fi

# Backend
cd /Users/usuario/mayu-ai-backend
"$NODE" server.js > /Users/usuario/mayu-ai-backend/node.out.log 2> /Users/usuario/mayu-ai-backend/node.err.log &

# Espera un momento y valida puerto
sleep 1
if ! /usr/sbin/lsof -nP -iTCP:8787 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "❌ Backend no levantó en 8787" >&2
  exit 1
fi

# Tunnel (se queda en foreground para que launchd lo supervise)
exec /opt/homebrew/bin/cloudflared tunnel run mayu-ai
