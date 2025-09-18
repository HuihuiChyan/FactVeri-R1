export CUDA_VISIBLE_DEVICES=2

echo "Running baseline fact-checking (without search)..."
MODEL=Qwen2.5-7B-Instruct
python fact_checker.py \
    --baseline \
    --input_file ./fact_checking_dataset/bamboogle_judged.jsonl \
    --output_file ./results/bamboogle_judged_${MODEL}_baseline.jsonl \
    --log_file ./results/bamboogle_judged_${MODEL}_baseline.log \
    --model_path /workspace/HFModels/${MODEL} \
    --evaluate

echo "Baseline run completed. Results saved to ./results/hotpotqa_subset_judged_${MODEL}_baseline.jsonl"