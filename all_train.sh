#!/bin/bash

MODEL_PATH="./saved_models"

RELU_OPTION=("" "--use_leaky_relu=True")
RELU_ARG=("" "leaky_")
RELU_NAME=("" " leaky")

mkdir -p $MODEL_PATH

# Pretraining stage
for relu in {0..1}
do
    for norm in {1..2}
    do
        echo "Training with${RELU_NAME[relu]} activation decay=1e-4 with L${norm} act decay"
        python3 train.py --activation_decay=1e-4 --activation_decay_norm=${norm} \
            ${RELU_OPTION[relu]} --lr=0.1 \
            --save_path=$MODEL_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}.pth
    done
done

# Large activation decay stage
for relu in {0..1}
do
    for norm in {1..2}
    do
        echo "Training with${RELU_NAME[relu]} activation decay=1e-2 with L${norm} act decay"
        python3 train.py --activation_decay=1e-2 --activation_decay_norm=${norm} \
            ${RELU_OPTION[relu]} --total_epochs=10 --lr=0.001 --clip_before=True \
            --load_path=$MODEL_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}.pth \
            --save_path=$MODEL_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_act.pth
    done
done

# Fine-tuning stage
for relu in {0..1}
do
    for norm in {1..2}
    do
        for m in {5..15}
        do
            for n in {2..3}
            do
                echo "Fitting into Chebyshev ${m} degree with${RELU_NAME[relu]} 1e-${n} act decay in L${norm}"
                python3 train.py --activation_decay=1e-${n} --activation_decay_norm=${norm} \
                    ${RELU_OPTION[relu]} --total_epochs=10 --lr=0.001 \
                    --load_path=$MODEL_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_act.pth \
                    --save_path=$MODEL_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_act_tuned_chebyshev_${m}_1e-${n}.pth \
                    --block_type=chebyshev --poly_degree=${m} --clip_before=True
                python3 eval.py ${RELU_OPTION[relu]} \
                    --load_path=$MODEL_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_act_tuned_chebyshev_${m}_1e-${n}.pth \
                    --block_type=chebyshev --poly_degree=${m}

                echo "Fitting into Remez ${m} degree with${RELU_NAME[relu]} 1e-${n} act decay in L${norm}"
                python3 train.py --activation_decay=1e-${n} --activation_decay_norm=${norm} \
                    ${RELU_OPTION[relu]} --total_epochs=10 --lr=0.001 \
                    --load_path=$MODEL_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_act.pth \
                    --save_path=$MODEL_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_act_tuned_remez_${m}_1e-${n}.pth \
                    --block_type=remez --poly_degree=${m} --clip_before=True
                python3 eval.py ${RELU_OPTION[relu]} \
                    --load_path=$MODEL_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_act_tuned_remez_${m}_1e-${n}.pth \
                    --block_type=remez --poly_degree=${m}

                echo "Fitting into Chebyshev ${m} degree without prefit stage${RELU_NAME[relu]} 1e-${n} act decay in L${norm}"
                python3 train.py --activation_decay=1e-${n} --activation_decay_norm=${norm} \
                    ${RELU_OPTION[relu]} --total_epochs=10 --lr=0.001 \
                    --load_path=$MODEL_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}.pth \
                    --save_path=$MODEL_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_tuned_chebyshev_${m}_1e-${n}.pth \
                    --block_type=chebyshev --poly_degree=${m} --clip_before=True
                python3 eval.py ${RELU_OPTION[relu]} \
                    --load_path=$MODEL_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_tuned_chebyshev_${m}_1e-${n}.pth \
                    --block_type=chebyshev --poly_degree=${m}

                echo "Fitting into Remez ${m} degree without prefit stage${RELU_NAME[relu]} 1e-${n} act decay in L${norm}"
                python3 train.py --activation_decay=1e-${n} --activation_decay_norm=${norm} \
                    ${RELU_OPTION[relu]} --total_epochs=10 --lr=0.001 \
                    --load_path=$MODEL_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}.pth \
                    --save_path=$MODEL_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_tuned_remez_${m}_1e-${n}.pth \
                    --block_type=remez --poly_degree=${m} --clip_before=True
                python3 eval.py ${RELU_OPTION[relu]} \
                    --load_path=$MODEL_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_tuned_remez_${m}_1e-${n}.pth \
                    --block_type=remez --poly_degree=${m}
            done
        done
    done
done