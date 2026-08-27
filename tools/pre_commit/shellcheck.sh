#!/bin/bash
set -euo pipefail

scversion="stable"

if [ -d "shellcheck-${scversion}" ]; then
    export PATH="$PATH:$(pwd)/shellcheck-${scversion}"
fi

if ! [ -x "$(command -v shellcheck)" ]; then
    if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
        echo "Please install shellcheck: https://github.com/koalaman/shellcheck?tab=readme-ov-file#installing"
        exit 1
    fi

    # automatic local install if linux x86_64
    wget -qO- "https://github.com/koalaman/shellcheck/releases/download/${scversion?}/shellcheck-${scversion?}.linux.x86_64.tar.xz" | tar -xJv
    export PATH="$PATH:$(pwd)/shellcheck-${scversion}"
fi

run_shellcheck() {
    for file in "$@"; do
        git check-ignore -q "$file" || shellcheck -s bash "$file"
    done
}

if [ "$#" -gt 0 ]; then
    run_shellcheck "$@"
else
    # TODO - fix warnings in .buildkite/scripts/hardware_ci/run-amd-test.sh
    while IFS= read -r -d '' file; do
        run_shellcheck "$file"
    done < <(
        find . -path ./.git -prune -o -name "*.sh" \
          -not -path "./.buildkite/scripts/hardware_ci/run-amd-test.sh" -print0
    )
fi
