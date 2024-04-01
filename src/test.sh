#!/bin/bash
args1=("${@:2}")
args2=("${@}")

bash << EOT
# export NUM_GPUS=1
# echo NUM_GPUS=\${NUM_GPUS}
echo args1=${args1[@]}
echo args2=${args2[@]}
EOT