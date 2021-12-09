#!/bin/bash

MODEL_PATH="./saved_models"
LOG_PATH="./logs"

RELU_OPTION=("" "--use_leaky_relu=True")
RELU_ARG=("" "leaky_")
RELU_NAME=("" " leaky")

mkdir -p $LOG_PATH

# Pretraining stage
for relu in {0..1}
do
    for norm in {1..2}
    do
        for m in {5..15}
        do
            echo "Evaluating${RELU_NAME[relu]} with Chebyshev's ${m} degree, learned with L${norm}-norm"
            python3 eval.py ${RELU_OPTION[relu]} \
                --load_path=$MODEL_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}.pth \
                --json_path=$LOG_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_chebyshev_${m}.json \
                --block_type=chebyshev --poly_degree=${m}
            
            echo "Evaluating${RELU_NAME[relu]} with Remez's ${m} degree, learned with L${norm}-norm"
            python3 eval.py ${RELU_OPTION[relu]} \
                --load_path=$MODEL_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}.pth \
                --json_path=$LOG_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_remez_${m}.json \
                --block_type=remez --poly_degree=${m}
        done
    done
done

# Large activation decay stage
for relu in {0..1}
do
    for norm in {1..2}
    do
        for m in {5..15}
        do
            echo "Evaluating after large act${RELU_NAME[relu]} with Chebyshev's ${m} degree, learned with L${norm}-norm"
            python3 eval.py ${RELU_OPTION[relu]} \
                --load_path=$MODEL_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_act.pth \
                --json_path=$LOG_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_act_chebyshev_${m}.json \
                --block_type=chebyshev --poly_degree=${m}
            
            echo "Evaluating after large act${RELU_NAME[relu]} with Remez's ${m} degree, learned with L${norm}-norm"
            python3 eval.py ${RELU_OPTION[relu]} \
                --load_path=$MODEL_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_act.pth \
                --json_path=$LOG_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_act_remez_${m}.json \
                --block_type=remez --poly_degree=${m}
        done
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
                echo "Fine-tuned Chebyshev ${m} degree with${RELU_NAME[relu]} 1e-${n} act decay in L${norm}"
                python3 eval.py ${RELU_OPTION[relu]} \
                    --load_path=$MODEL_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_act_tuned_chebyshev_${m}_1e-${n}.pth \
                    --json_path=$LOG_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_act_tuned_chebyshev_${m}_1e-${n}.json \
                    --block_type=chebyshev --poly_degree=${m}

                echo "Fine-tuned into Remez ${m} degree with${RELU_NAME[relu]} 1e-${n} act decay in L${norm}"
                python3 eval.py ${RELU_OPTION[relu]} \
                    --load_path=$MODEL_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_act_tuned_remez_${m}_1e-${n}.pth \
                    --json_path=$LOG_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_act_tuned_remez_${m}_1e-${n}.json \
                    --block_type=remez --poly_degree=${m}

                echo "Fine-tuned into Chebyshev ${m} degree without large act stage${RELU_NAME[relu]} 1e-${n} act decay in L${norm}"
                python3 eval.py ${RELU_OPTION[relu]} \
                    --load_path=$MODEL_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_tuned_chebyshev_${m}_1e-${n}.pth \
                    --json_path=$LOG_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_tuned_chebyshev_${m}_1e-${n}.json \
                    --block_type=chebyshev --poly_degree=${m}

                echo "Fine-tuned into Remez ${m} degree without large act stage${RELU_NAME[relu]} 1e-${n} act decay in L${norm}"
                python3 eval.py ${RELU_OPTION[relu]} \
                    --load_path=$MODEL_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_tuned_remez_${m}_1e-${n}.pth \
                    --json_path=$LOG_PATH/simple_mnist_${RELU_ARG[relu]}l${norm}_tuned_remez_${m}_1e-${n}.json \
                    --block_type=remez --poly_degree=${m}
            done
        done
    done
done