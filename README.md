# Dis-information Detection

This project is a dis-information detection that processes datasets to extract, search, and perform factuality verification. 

## Usage

The following is an example script for thinking-searching-verification:

```bash
export SERPER_KEY_PRIVATE="your-serper-key"
export OPENAI_API_BASE="https://api.shubiaobiao.cn/v1"
export OPENAI_API_KEY="your-openai-api-key"

INPUT="./fact_checking_dataset/fact_checking_dataset_2017.jsonl"
OUTPUT="./fact_checking_dataset/fact_checking_dataset_2017.json"

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

```

The following is an example script for direct-verification:

```bash
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
```


## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
