#! /bin/bash

usage() {
    echo "Usage: $0 <learning policy> <action policy>"
    echo "  <learning policy>: q_learning | sarsa | DQN | Dsarsa"
    echo "  <action policy>: epsilon_greedy | softmax_method"
    exit 1
}

if [ "$#" -ne 2 ]; then
    echo -e "Error: Missing arguments.\n"
    usage
fi

policy="$1"
apn="$2"
is_pkl=1

case "$policy" in
    q_learning | sarsa) ;;
    DQN | Dsarsa) 
        is_pkl=0
        ;;
    *)
        echo -e "Error: Invalid policy.\n"
        usage
        ;;
esac

case "$apn" in
    epsilon_greedy | softmax_method) ;;
    *)
        echo -e "Error: Invalid action policy.\n"
        usage
        ;;
esac

mkdir -p best_performance

img_file="images/${policy}_${apn}.png"
if [ -e "$img_file" ]; then
    cp "$img_file" best_performance/. 
    echo "[success] copied $img_file"
else 
    echo "[fail] Image file not found in images/"
fi

docs_file="documents/${policy}_${apn}.txt"
if [ -e "$docs_file" ]; then
    cp "$docs_file" best_performance/.
    echo "[success] copied $docs_file"
else 
    echo "[fail] Document file not found in documents/"
fi

if [ "$is_pkl" -eq 1 ]; then
    w_file="model_weights/${policy}_${apn}.pkl"
else 
    w_file="model_weights/${policy}_${apn}.weights.h5"
fi

if [ -e "$w_file" ]; then
    cp $w_file best_performance/.
    echo "[success] copied $w_file"
else 
    echo "[fail] File not found in model_weights/"
fi