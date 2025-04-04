#!/bin/bash

#cd ../..

# custom config
DATA=./
TRAINER=CoOp
Device_id=0
DATASET=Pneumonia
DESCRIPTOR_PATH=descriptors/descriptors_pneumonia.json  # descriptor path
CFG=('rn50x64_ep50' 'vit_b16_ep50' 'vit_L14_336_ep50') 
CTP=end  # class token position (end)
NCTX=4  # number of context tokens
SHOTS=10000  # full data
CSC=False  # class-specific context (False or True)


for CFG in ${CFG[@]}
do
    for SEED in 1 2 3
    do
        DIR=output/${DATASET}/${TRAINER}/${CFG}/${s}shots/nctx${NCTX}_csc_${CSC}_ctp_${CTP}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Results are available in ${DIR}. Skip this job"
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
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS}
        fi
    done
done