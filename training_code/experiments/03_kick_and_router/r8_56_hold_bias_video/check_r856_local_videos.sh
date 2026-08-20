#!/usr/bin/env bash
set -Eeuo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; echo "BASE=$BASE"; FAIL=0
for item in '0.75:low' '1.0:mid' '1.25:high'; do strength="${item%%:*}"; label="${item##*:}"; video="$BASE/videos/R856_${label}_${strength}_kick.mp4"; meta="${video%.mp4}.json"; ready="${video}.ready"; if [[ -s "$video" && -s "$meta" && -s "$ready" ]]; then stat --printf='READY %n %s bytes %y\n' "$video"; python - "$meta" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8')); print('META strength',d['strength'],'frames',d['frames'],'real',d['real_contact'],'touchdown',d['touchdown'],'recovery3',d['recovery3'],'wrong',d['wrong'],'body',d['body'],'bias',d['bias'])
PY
 else echo "NOT_READY $video"; FAIL=1; fi; done
[[ "$FAIL" == 0 ]]||{ echo VERDICT=LOCAL_VIDEOS_INCOMPLETE; exit 4; }; [[ -s "$BASE/videos/SHA256SUMS" ]]&&(cd "$BASE/videos"&&sha256sum -c SHA256SUMS); echo VERDICT=THREE_VIDEOS_READY_FOR_HUMAN_REVIEW
