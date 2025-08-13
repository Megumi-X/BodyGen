N="range(1)"
MODEL="BodyGen"
# EXPTS="crawler terraincrosser cheetah swimmer glider-regular glider-medium glider-hard walker-regular walker-medium walker-hard"
EXPTS="walker-hard_tunnel_neo_reconfig"
SEED=0

for EXPT in $EXPTS; do
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7 python -m design_opt.train -m cfg=$EXPT n="$N" \
        seed=$SEED \
        hydra.sweep.dir="multirun/$MODEL-$EXPT/$(date "+%Y-%m-%d-%H-%M-%S-%N")" \
        group=$MODEL-$EXPT
done