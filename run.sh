#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
export LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH
exec python3 -m whisper_input.main "$@"
