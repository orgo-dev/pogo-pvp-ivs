#!/usr/bin/env bash
docker run \
    --name pogo-pvp-ivs \
    --rm \
    -t \
    -p 8501:8501 \
    -v $PWD:/app \
    -e STREAMLIT_SERVER_FILE_WATCHER_TYPE=poll \
    pogo-pvp-ivs
