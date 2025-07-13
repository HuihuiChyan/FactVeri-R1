export CUDA_VISIBLE_DEVICES=6

echo "Running baseline fact-checking (without search)..."
YEAR=2017
MODEL=Qwen2.5-3B-Instruct
python fact_checker.py \
    --baseline \
    --input_file ./fact_checking_dataset/fact_checking_dataset_${YEAR}.jsonl \
    --output_file ./results/fact_checking_dataset_${YEAR}_${MODEL}_baseline.jsonl \
    --log_file ./results/fact_checking_dataset_${YEAR}_${MODEL}_baseline.log \
    --model_path /workspace/HFModels/${MODEL} \
    --evaluate

echo "Baseline run completed. Results saved to ./results/fact_checking_dataset_${YEAR}_${MODEL}_baseline.jsonl"