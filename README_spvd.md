# Docker Prep
```bash
docker run --rm -d --runtime=habana --name="zhenzhong-xengine-spvd" -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host -v /home/zhenzhong/repo2025/pd_hccl_vllm_develop:/data -v /mnt/disk01/models/DeepSeek-R1-Distill-Qwen-32B:/models vault.habana.ai/gaudi-docker/1.21.2/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest  /bin/bash -c "ssh-keygen -A && service ssh start && tail -f /dev/null"
```



# XEngine-Vllm


```bash
curl -X POST -s http://localhost:8101/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "/models","messages": [{"role": "user", "content": "Tell me something about Intel"}],"max_completion_tokens": 32,"temperature": 0,"ignore_eos": true}'

```


# vLLMfrok
```bash
# P
python3 -m vllm.entrypoints.openai.api_server --model /models --port 8101 --dtype bfloat16 --block-size 32 --max-num-seqs 256 --max-num-batched-tokens 16384 --max-model-len 10240 --device hpu --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_producer"}'

# D
python3 -m vllm.entrypoints.openai.api_server --model /models --port 8200 --dtype bfloat16 --block-size 32 --max-num-seqs 256 --max-num-batched-tokens 16384 --max-model-len 10240 --device hpu --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer"}'

python3 /data/frameworks.ai.xengine-vllm/examples/online_serving/disagg_examples/disagg_proxy_demo.py     --model /models     --prefill 127.0.0.1:8101     --decode 127.0.0.1:8200     --port 8300

curl -X POST -s http://localhost:8300/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "/models","messages": [{"role": "user", "content": "Tell me something about Intel"}],"max_completion_tokens": 32,"temperature": 0,"ignore_eos": true}'
```

# XEngine-Vllm
```bash


# P
VLLM_SYNAPSELLM_ASYNC=0 on_device_sampling_disabled=True NUM_RESERVED_FREE_SLOTS=0 python3 -m vllm.entrypoints.openai.api_server --model /models --port 8101 --dtype bfloat16 --block-size 32 --max-num-seqs 256 --max-num-batched-tokens 16384 --max-model-len 10240 --device hpu --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_producer"}'

```


```bash
pip install lm-eval[api]
HF_ALLOW_CODE_EVAL=1 lm_eval --model local-completions --tasks humaneval --model_args model=/models,base_url=http://127.0.0.1:8300/v1/completions,num_concurrent=1 --batch_size 1 --confirm_run_unsafe_code --log_samples --output_path ./lm_eval_output_humaneval

lm_eval --model local-completions \
    --tasks gsm8k \
    --model_args model=/models,base_url=http://localhost:8300/v1/completions,num_concurrent=1 \
    --batch_size 1 \
    --log_samples \
    --output_path ./lm_eval_output \
    --num_fewshot 5


python benchmarks/benchmark_serving.py --backend vllm --model /models --trust-remote-code \
--port 8300 --dataset-path benchmarks/sonnet.txt  \
--dataset-name sonnet --sonnet-input-len 2048 --sonnet-output-len 64  \
--num-prompts 1 --request-rate inf --seed 0 --ignore_eos --max-concurrency 1 \
--save-result --result-filename synapsellm-sonnet.json
```