#!/bin/sh
source activate python39

temp=0.05
batch=64
lr=2e-5

languages=('python' 'java' 'go' 'php' 'javascript' 'ruby')
for lang in ${languages[@]}
    do
        echo $lang
        echo "*** pre-train 100K steps ***"

        output_dir=fine-tune/code_search/saved_models/computing_resource/100K_steps
        mkdir -p $output_dir
        saved_pretrained=pre-train/saved_models/computing_resource/100K_steps

        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:22302 --nproc_per_node=4 --nnodes=1 run_fsdp.py \
            --output_dir=$output_dir \
            --model_name_or_path=$saved_pretrained \
            --tokenizer_name=$saved_pretrained \
            --do_train \
            --train_data_file=dataset/CSN/$lang/train.jsonl \
            --eval_data_file=dataset/CSN/$lang/valid.jsonl \
            --test_data_file=dataset/CSN/$lang/test.jsonl \
            --codebase_file=dataset/CSN/$lang/codebase.jsonl \
            --language=$lang \
            --input_type=source_code \
            --num_train_epochs 10 \
            --code_length 512 \
            --nl_length 128 \
            --train_batch_size $batch \
            --eval_batch_size 128 \
            --learning_rate $lr \
            --temp $temp \
            --port 22302 \
            --seed 123456 2>&1| tee $output_dir/train.log
        wait


        echo "start test on lang"

        CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:22302 --nproc_per_node=2 --nnodes=1 run_fsdp.py \
            --output_dir=$output_dir \
            --model_name_or_path=$saved_pretrained \
            --tokenizer_name=$saved_pretrained \
            --do_eval \
            --do_test \
            --train_data_file=dataset/CSN/$lang/train.jsonl \
            --eval_data_file=dataset/CSN/$lang/valid.jsonl \
            --test_data_file=dataset/CSN/$lang/test.jsonl \
            --codebase_file=dataset/CSN/$lang/codebase.jsonl \
            --language=$lang \
            --input_type=source_code \
            --num_train_epochs 10 \
            --code_length 512 \
            --nl_length 128 \
            --train_batch_size $batch \
            --eval_batch_size 128 \
            --learning_rate $lr \
            --temp $temp \
            --port 22302 \
            --seed 123456 2>&1| tee $output_dir/test.log

        wait


        echo "*** pre-train 200K steps ***"

        output_dir=fine-tune/code_search/saved_models/computing_resource/200K_steps
        mkdir -p $output_dir
        saved_pretrained=pre-train/saved_models/computing_resource/200K_steps

        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:22302 --nproc_per_node=4 --nnodes=1 run_fsdp.py \
            --output_dir=$output_dir \
            --model_name_or_path=$saved_pretrained \
            --tokenizer_name=$saved_pretrained \
            --do_train \
            --train_data_file=dataset/CSN/$lang/train.jsonl \
            --eval_data_file=dataset/CSN/$lang/valid.jsonl \
            --test_data_file=dataset/CSN/$lang/test.jsonl \
            --codebase_file=dataset/CSN/$lang/codebase.jsonl \
            --language=$lang \
            --input_type=source_code \
            --num_train_epochs 10 \
            --code_length 512 \
            --nl_length 128 \
            --train_batch_size $batch \
            --eval_batch_size 128 \
            --learning_rate $lr \
            --temp $temp \
            --port 22302 \
            --seed 123456 2>&1| tee $output_dir/train.log
        wait


        echo "start test on lang"

        CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:22302 --nproc_per_node=2 --nnodes=1 run_fsdp.py \
            --output_dir=$output_dir \
            --model_name_or_path=$saved_pretrained \
            --tokenizer_name=$saved_pretrained \
            --do_eval \
            --do_test \
            --train_data_file=dataset/CSN/$lang/train.jsonl \
            --eval_data_file=dataset/CSN/$lang/valid.jsonl \
            --test_data_file=dataset/CSN/$lang/test.jsonl \
            --codebase_file=dataset/CSN/$lang/codebase.jsonl \
            --language=$lang \
            --input_type=source_code \
            --num_train_epochs 10 \
            --code_length 512 \
            --nl_length 128 \
            --train_batch_size $batch \
            --eval_batch_size 128 \
            --learning_rate $lr \
            --temp $temp \
            --port 22302 \
            --seed 123456 2>&1| tee $output_dir/test.log

        wait


        echo "*** pre-train 300K steps ***"

        output_dir=fine-tune/code_search/saved_models/computing_resource/300K_steps
        mkdir -p $output_dir
        saved_pretrained=pre-train/saved_models/computing_resource/300K_steps

        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:22303 --nproc_per_node=4 --nnodes=1 run_fsdp.py \
            --output_dir=$output_dir \
            --model_name_or_path=$saved_pretrained \
            --tokenizer_name=$saved_pretrained \
            --do_train \
            --train_data_file=dataset/CSN/$lang/train.jsonl \
            --eval_data_file=dataset/CSN/$lang/valid.jsonl \
            --test_data_file=dataset/CSN/$lang/test.jsonl \
            --codebase_file=dataset/CSN/$lang/codebase.jsonl \
            --language=$lang \
            --input_type=source_code \
            --num_train_epochs 10 \
            --code_length 512 \
            --nl_length 128 \
            --train_batch_size $batch \
            --eval_batch_size 128 \
            --learning_rate $lr \
            --temp $temp \
            --port 22302 \
            --seed 123456 2>&1| tee $output_dir/train.log
        wait


        echo "start test on lang"

        CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:22303 --nproc_per_node=2 --nnodes=1 run_fsdp.py \
            --output_dir=$output_dir \
            --model_name_or_path=$saved_pretrained \
            --tokenizer_name=$saved_pretrained \
            --do_eval \
            --do_test \
            --train_data_file=dataset/CSN/$lang/train.jsonl \
            --eval_data_file=dataset/CSN/$lang/valid.jsonl \
            --test_data_file=dataset/CSN/$lang/test.jsonl \
            --codebase_file=dataset/CSN/$lang/codebase.jsonl \
            --language=$lang \
            --input_type=source_code \
            --num_train_epochs 10 \
            --code_length 512 \
            --nl_length 128 \
            --train_batch_size $batch \
            --eval_batch_size 128 \
            --learning_rate $lr \
            --temp $temp \
            --port 22302 \
            --seed 123456 2>&1| tee $output_dir/test.log

        wait


        echo "*** pre-train 400K steps ***"

        output_dir=fine-tune/code_search/saved_models/computing_resource/400K_steps
        mkdir -p $output_dir
        saved_pretrained=pre-train/saved_models/computing_resource/400K_steps

        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:22303 --nproc_per_node=4 --nnodes=1 run_fsdp.py \
            --output_dir=$output_dir \
            --model_name_or_path=$saved_pretrained \
            --tokenizer_name=$saved_pretrained \
            --do_train \
            --train_data_file=dataset/CSN/$lang/train.jsonl \
            --eval_data_file=dataset/CSN/$lang/valid.jsonl \
            --test_data_file=dataset/CSN/$lang/test.jsonl \
            --codebase_file=dataset/CSN/$lang/codebase.jsonl \
            --language=$lang \
            --input_type=source_code \
            --num_train_epochs 10 \
            --code_length 512 \
            --nl_length 128 \
            --train_batch_size $batch \
            --eval_batch_size 128 \
            --learning_rate $lr \
            --temp $temp \
            --port 22302 \
            --seed 123456 2>&1| tee $output_dir/train.log
        wait


        echo "start test on lang"

        CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:22303 --nproc_per_node=2 --nnodes=1 run_fsdp.py \
            --output_dir=$output_dir \
            --model_name_or_path=$saved_pretrained \
            --tokenizer_name=$saved_pretrained \
            --do_eval \
            --do_test \
            --train_data_file=dataset/CSN/$lang/train.jsonl \
            --eval_data_file=dataset/CSN/$lang/valid.jsonl \
            --test_data_file=dataset/CSN/$lang/test.jsonl \
            --codebase_file=dataset/CSN/$lang/codebase.jsonl \
            --language=$lang \
            --input_type=source_code \
            --num_train_epochs 10 \
            --code_length 512 \
            --nl_length 128 \
            --train_batch_size $batch \
            --eval_batch_size 128 \
            --learning_rate $lr \
            --temp $temp \
            --port 22302 \
            --seed 123456 2>&1| tee $output_dir/test.log

        wait


        echo "*** pre-train 600K steps ***"

        output_dir=fine-tune/code_search/saved_models/computing_resource/600K_steps
        mkdir -p $output_dir
        saved_pretrained=pre-train/saved_models/computing_resource/600K_steps

        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:22303 --nproc_per_node=4 --nnodes=1 run_fsdp.py \
            --output_dir=$output_dir \
            --model_name_or_path=$saved_pretrained \
            --tokenizer_name=$saved_pretrained \
            --do_train \
            --train_data_file=dataset/CSN/$lang/train.jsonl \
            --eval_data_file=dataset/CSN/$lang/valid.jsonl \
            --test_data_file=dataset/CSN/$lang/test.jsonl \
            --codebase_file=dataset/CSN/$lang/codebase.jsonl \
            --language=$lang \
            --input_type=source_code \
            --num_train_epochs 10 \
            --code_length 512 \
            --nl_length 128 \
            --train_batch_size $batch \
            --eval_batch_size 128 \
            --learning_rate $lr \
            --temp $temp \
            --port 22302 \
            --seed 123456 2>&1| tee $output_dir/train.log
        wait


        echo "start test on lang"

        CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:22303 --nproc_per_node=2 --nnodes=1 run_fsdp.py \
            --output_dir=$output_dir \
            --model_name_or_path=$saved_pretrained \
            --tokenizer_name=$saved_pretrained \
            --do_eval \
            --do_test \
            --train_data_file=dataset/CSN/$lang/train.jsonl \
            --eval_data_file=dataset/CSN/$lang/valid.jsonl \
            --test_data_file=dataset/CSN/$lang/test.jsonl \
            --codebase_file=dataset/CSN/$lang/codebase.jsonl \
            --language=$lang \
            --input_type=source_code \
            --num_train_epochs 10 \
            --code_length 512 \
            --nl_length 128 \
            --train_batch_size $batch \
            --eval_batch_size 128 \
            --learning_rate $lr \
            --temp $temp \
            --port 22302 \
            --seed 123456 2>&1| tee $output_dir/test.log

        wait

    done

echo "end task"