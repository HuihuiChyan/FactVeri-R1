export CUDA_VISIBLE_DEVICES=5
export SERPER_KEY_PRIVATE="a6783afd17eeaf93c04c0bb44a99038dd950165e"

echo "Running agentic fact-checking with search..."
YEAR=2025
MODEL=Qwen2.5-3B-Instruct
python fact_checker.py \
    --input_file ./fact_checking_dataset/fact_checking_dataset_${YEAR}.jsonl \
    --output_file ./results/fact_checking_dataset_${YEAR}_${MODEL}_agentic.jsonl \
    --log_file ./results/fact_checking_dataset_${YEAR}_${MODEL}_agentic.log \
    --model_path /workspace/HFModels/${MODEL} \
    --search_engine_url "https://google.serper.dev/search" \
    --search_api_type "serper" \
    --max_search_limit 20 \
    --evaluate

echo "Agentic run completed. Results saved to ./results/fact_checking_dataset_${YEAR}_${MODEL}_agentic.jsonl"