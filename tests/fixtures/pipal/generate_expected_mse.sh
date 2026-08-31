#!/usr/bin/env bash

set -euo pipefail

fixture_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_file="${fixture_dir}/expected_mse.csv"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg is required to generate the expected MSE values." >&2
    exit 1
fi

if [[ ! -f "${fixture_dir}/ref.png" ]]; then
    echo "Reference image not found: ${fixture_dir}/ref.png" >&2
    exit 1
fi

distorted_images=("${fixture_dir}"/dist_*.png)

if [[ ! -e "${distorted_images[0]}" ]]; then
    echo "No distorted images matching dist_*.png were found." >&2
    exit 1
fi

printf "image,mse\n" > "${output_file}"

for distorted in "${distorted_images[@]}"; do
    stats="$(
        ffmpeg \
            -hide_banner \
            -loglevel error \
            -i "${fixture_dir}/ref.png" \
            -i "${distorted}" \
            -filter_complex \
                "[0:v]scale=in_range=full:out_range=full:out_color_matrix=bt601,format=yuv444p,extractplanes=y[ref]; \
                 [1:v]scale=in_range=full:out_range=full:out_color_matrix=bt601,format=yuv444p,extractplanes=y[dist]; \
                 [ref][dist]psnr=stats_file=-" \
            -frames:v 1 \
            -f null - \
            2>&1
    )"

    mse="$(
        awk '{
            for (i = 1; i <= NF; i++) {
                if ($i ~ /^mse_y:/) {
                    sub(/^mse_y:/, "", $i)
                    print $i
                }
            }
        }' <<< "${stats}"
    )"

    if [[ -z "${mse}" ]]; then
        echo "Unable to extract MSE for ${distorted}." >&2
        exit 1
    fi

    printf "%s,%s\n" "$(basename "${distorted}")" "${mse}" \
        >> "${output_file}"
done

echo "Wrote expected MSE values to ${output_file}"
