#!/bin/bash

#cd ../..

# custom config
DATA=./
TRAINER=ViP
Device_id=0
DATASET=Derm
DESCRIPTOR_PATH=descriptors/descriptors_skin.json  # descriptor path
CFG=('rn50x64_ep50' 'vit_b16_ep50' 'vit_L14_336_ep50')    # config file
CTP=end  # class token position (end or middle)
NCTX=4 # number of context tokens
SHOTS=10000  # full data
CSC=False  # class-specific context (False or True)


for CFG in ${CFG[@]}
do
    for SEED in 1 2 3
    do
        DIR=output/${DATASET}/${TRAINER}/${CFG}/${s}shots/nctx${NCTX}_csc_${CSC}_ctp_${CTP}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Run this job and save the output to ${DIR}"
        else
            echo "Run this job and save the output to ${DIR}"
            CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --descriptor-path ${DESCRIPTOR_PATH} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            TRAINER.VIP.N_CTX ${NCTX} \
            TRAINER.VIP.CSC ${CSC} \
            TRAINER.VIP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS}
        fi
    done
done